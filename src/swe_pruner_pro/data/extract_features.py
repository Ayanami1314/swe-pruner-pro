"""Extract per-token hidden states with SGLang for from-features training.

Runs the SGLang offline ``Engine`` on a labelled JSONL, saves one ``.npz`` per
sample with hidden states + per-token labels + per-token line ids (sliced to
the ``<tool_response>`` region). Pack the per-sample shards into memmap
``.bin`` files via :mod:`swe_pruner_pro.data.pack_features`.

SGLang must be launched with ``enable_return_hidden_states=True``. Long-sequence
hidden-state truncation under chunked prefill is mitigated by passing
``--disable-radix-cache`` (default) and optionally ``--disable-cuda-graph`` /
``--chunked-prefill-size -1``.

Each per-sample ``.npz`` contains::

    hidden_states    [T, D] float16 — slice over <tool_response> tokens
    token_labels     [T]   int16   — 0/1 per token (-100 for non-code gaps)
    token_line_ids   [T]   int16   — 1-based line id, 0 = non-code
"""
from __future__ import annotations

import base64
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

app = typer.Typer(add_completion=False, help="Hidden-state feature extraction (SGLang)")
console = Console()


# ---------------------------------------------------------------------------
# Tokenizer offsets
# ---------------------------------------------------------------------------

def _encode_with_offsets(tokenizer, text: str):
    """Tokenize and return (ids, offsets); falls back to a per-token decode walk
    for slow tokenizers that don't expose ``return_offsets_mapping``."""
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]
    except (NotImplementedError, ValueError, TypeError):
        ids = tokenizer.encode(text, add_special_tokens=False)
        offsets: list[tuple[int, int]] = []
        pos = 0
        n = len(text)
        for tid in ids:
            piece = tokenizer.decode([tid])
            if not piece:
                offsets.append((pos, pos))
                continue
            found = text.find(piece, pos)
            if found < 0:
                offsets.append((pos, min(pos + 1, n)))
                pos = min(pos + 1, n)
            else:
                offsets.append((found, found + len(piece)))
                pos = found + len(piece)
        return ids, offsets


# ---------------------------------------------------------------------------
# Prompt + label construction (Phase-2 / trajectory format only)
# ---------------------------------------------------------------------------

def _find_token_range(ids: list[int], start_tok: int, end_tok: int) -> tuple[int, int]:
    de = -1
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == end_tok:
            de = i
            break
    if de < 0:
        return 0, 0
    ds = -1
    for i in range(de - 1, -1, -1):
        if ids[i] == start_tok:
            ds = i + 1
            break
    if ds < 0:
        return 0, 0
    return ds, de


def _sanitize_messages(msgs: list[dict]) -> list[dict]:
    for m in msgs:
        if m.get("content") is None:
            m["content"] = ""
    return msgs


def _code_to_labels(code: str, kept_frags: list[int], tokenizer, doc_start: int, seq_len: int):
    kept = set(kept_frags or [])
    char_to_line: list[int] = []
    for idx, line in enumerate(code.splitlines(keepends=True), 1):
        char_to_line.extend([idx] * len(line))

    _ids, offsets = _encode_with_offsets(tokenizer, code)
    labels = np.full(seq_len, -100, dtype=np.int16)
    line_ids = np.zeros(seq_len, dtype=np.int16)
    for i, (tok_s, _) in enumerate(offsets):
        pos = doc_start + i
        if pos >= seq_len:
            break
        if tok_s < len(char_to_line):
            lid = char_to_line[tok_s]
            labels[pos] = 1 if lid in kept else 0
            line_ids[pos] = lid
    return labels, line_ids


def build_phase2(sample: dict, tokenizer) -> dict:
    """Build (input_ids, token_labels, token_line_ids, doc_start, doc_end)."""
    tc = sample.get("tool_call", {}) or {}
    name = tc.get("name", "tool") if isinstance(tc, dict) else "tool"
    args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
    tr = sample.get("tool_response") or ""

    msgs = _sanitize_messages(list(sample.get("history", [])) + [
        {"role": "assistant", "tool_calls": [{"function": {"name": name, "arguments": args}}]},
        {"role": "tool", "content": tr},
    ])
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    tr_start = tokenizer.convert_tokens_to_ids("<tool_response>")
    tr_end = tokenizer.convert_tokens_to_ids("</tool_response>")
    ds, de = _find_token_range(ids, tr_start, tr_end)

    # Fallback for chat templates without explicit <tool_response> markers.
    if (de - ds <= 0) and tr:
        char_start = prompt.rfind(tr)
        if char_start >= 0:
            char_end = char_start + len(tr)
            _ids2, offs = _encode_with_offsets(tokenizer, prompt)
            ds_fb = de_fb = 0
            for i, (s, e) in enumerate(offs):
                if e <= char_start:
                    continue
                if ds_fb == 0 and s < char_end:
                    ds_fb = i
                if s >= char_end:
                    de_fb = i
                    break
            if de_fb == 0 and ds_fb > 0:
                de_fb = len(offs)
            if de_fb > ds_fb:
                ds, de = ds_fb, de_fb

    labels, line_ids = _code_to_labels(tr, sample.get("kept_frags", []), tokenizer, ds, len(ids))
    return {
        "prompt": prompt, "input_ids": ids,
        "token_labels": labels, "token_line_ids": line_ids,
        "doc_start": ds, "doc_end": de,
    }


def inject_negatives(data: list[dict], ratio: float):
    """Append ``ratio * len(data)`` negative samples (random tool_response swaps)."""
    n_neg = int(len(data) * ratio)
    for k in range(n_neg):
        i = random.randint(0, len(data) - 1)
        j = (i + random.randint(1, len(data) - 1)) % len(data)
        if k % 2 == 0:
            data.append({**data[i], "tool_response": data[j].get("tool_response", ""),
                         "kept_frags": [], "total_lines": 0})
        else:
            data.append({**data[i], "history": data[j].get("history", []),
                         "kept_frags": [], "total_lines": 0})
    random.shuffle(data)


# ---------------------------------------------------------------------------
# extract — SGLang offline engine
# ---------------------------------------------------------------------------

@app.command()
def extract(
    input_jsonl: Path = typer.Argument(..., help="Labelled JSONL"),
    output_dir: Path = typer.Option(Path("features"), "-o"),
    model: str = typer.Option(..., "--model", help="Backbone model path or HF id"),
    tensor_parallel_size: int = typer.Option(8, "--tp"),
    data_parallel_size: int = typer.Option(1, "--dp"),
    batch_size: int = typer.Option(64, "-b"),
    max_model_len: int = typer.Option(16384),
    neg_ratio: float = typer.Option(0.0, "--neg-ratio"),
    seed: int = typer.Option(42),
    disable_cuda_graph: bool = typer.Option(False, "--disable-cuda-graph"),
    chunked_prefill_size: int = typer.Option(0, "--chunked-prefill-size",
        help="0 = SGLang default; -1 disables chunked prefill (recommended for long seqs)."),
    mem_fraction_static: float = typer.Option(0.80, "--mem-fraction-static"),
    attention_backend: str = typer.Option("", "--attention-backend"),
    enable_dp_attention: bool = typer.Option(False, "--enable-dp-attention"),
    save_compressed: bool = typer.Option(True, "--save-compressed/--save-fast",
        help="--save-fast skips zlib (much faster, ~3x more disk)."),
    async_save_workers: int = typer.Option(4, "--async-save-workers",
        help="Background threads for np.savez. 0 = synchronous."),
):
    """Extract hidden states via the SGLang offline ``Engine``.

    Tries ``from sglang import Engine`` first, falls back to
    ``sglang.srt.entrypoints.engine.Engine`` for older builds.
    """
    import torch
    try:
        import sglang as sgl  # type: ignore
        Engine = sgl.Engine  # noqa: N806
    except (ImportError, AttributeError):
        from sglang.srt.entrypoints.engine import Engine  # type: ignore
        import sglang as sgl  # type: ignore
        sgl.Engine = Engine  # type: ignore[attr-defined]

    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = [json.loads(line) for line in open(input_jsonl) if line.strip()]
    console.print(f"Loaded {len(data)} samples")
    if neg_ratio > 0:
        inject_negatives(data, neg_ratio)
        console.print(f"After negatives: {len(data)}")

    indexed = list(enumerate(data))
    existing = {f.stem for f in output_dir.glob("*.npz")}
    todo = [(i, d) for i, d in indexed if f"{i:06d}" not in existing]
    console.print(f"Existing: {len(existing)}, remaining: {len(todo)}")
    if not todo:
        return

    # Read config.json directly (avoids transformers version mismatches).
    model_path = Path(model)
    if (model_path / "config.json").exists():
        cfg = json.loads((model_path / "config.json").read_text())
    else:
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True).to_dict()
    if "text_config" in cfg and "num_hidden_layers" not in cfg:
        cfg = cfg["text_config"]
    n_layers = cfg["num_hidden_layers"]
    hidden_size = cfg["hidden_size"]
    console.print(f"Backbone: {n_layers} layers, hidden={hidden_size}")

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    (output_dir / "metadata.json").write_text(json.dumps({
        "model": model, "hidden_dim": hidden_size,
        "extract_layer": n_layers - 1, "n_samples": len(data),
    }, indent=2))

    engine_kwargs: dict = dict(
        model_path=model,
        tp_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        enable_return_hidden_states=True,
        disable_radix_cache=True,
        log_level="warning",
    )
    if attention_backend:
        engine_kwargs["attention_backend"] = attention_backend
    if chunked_prefill_size != 0:
        engine_kwargs["chunked_prefill_size"] = chunked_prefill_size
    if enable_dp_attention:
        engine_kwargs["enable_dp_attention"] = True
    if disable_cuda_graph:
        engine_kwargs["disable_cuda_graph"] = True
        engine_kwargs["disable_piecewise_cuda_graph"] = True
        engine_kwargs["disable_overlap_schedule"] = True
    if data_parallel_size > 1:
        engine_kwargs["dp_size"] = data_parallel_size

    console.print(f"[cyan]Starting SGLang Engine: tp={tensor_parallel_size} dp={data_parallel_size}[/cyan]")
    llm = Engine(**engine_kwargs)

    sampling_params = {"max_new_tokens": 1, "temperature": 0}
    save_fn = np.savez_compressed if save_compressed else np.savez
    save_pool: Optional[ThreadPoolExecutor] = (
        ThreadPoolExecutor(max_workers=async_save_workers) if async_save_workers > 0 else None
    )
    pending: list = []
    saved = skipped = 0
    char_limit = max_model_len * 4

    def _save(idx: int, built: dict, hs_list) -> bool:
        if isinstance(hs_list, dict) and hs_list.get("__binary__"):
            shape = tuple(hs_list["shape"])
            dtype = np.dtype(hs_list["dtype"])
            blob = base64.b64decode(hs_list["data"])
            hs = np.frombuffer(blob, dtype=dtype).reshape(shape).astype(np.float32)
        else:
            hs = torch.cat([
                t.unsqueeze(0) if t.ndim == 1 else t
                for t in [torch.tensor(h, dtype=torch.bfloat16) for h in hs_list]
            ]).float().numpy()
        full_len = len(built["input_ids"])
        ds, de = built["doc_start"], built["doc_end"]
        if de - ds <= 0:
            return False
        # Negative indexing — robust to SGLang prefix-cache truncation.
        hs_slice = hs[ds - full_len: de - full_len if de < full_len else None]
        if hs_slice.shape[0] == 0:
            return False
        save_fn(
            output_dir / f"{idx:06d}.npz",
            hidden_states=hs_slice.astype(np.float16),
            token_labels=built["token_labels"][ds:de],
            token_line_ids=built["token_line_ids"][ds:de],
        )
        return True

    batch_ids: list[list[int]] = []
    batch_meta: list[tuple[int, dict]] = []

    def _flush():
        nonlocal saved, skipped
        if not batch_ids:
            return
        outputs = llm.generate(
            input_ids=batch_ids,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )
        for (idx, built), out in zip(batch_meta, outputs):
            hs_list = out.get("meta_info", {}).get("hidden_states")
            if not hs_list:
                skipped += 1
                continue
            if save_pool is not None:
                pending.append(save_pool.submit(_save, idx, built, hs_list))
                if len(pending) > 4 * async_save_workers:
                    ok = pending.pop(0).result()
                    saved += int(ok); skipped += int(not ok)
            else:
                ok = _save(idx, built, hs_list)
                saved += int(ok); skipped += int(not ok)
        batch_ids.clear()
        batch_meta.clear()

    try:
        for idx, sample in tqdm(todo, desc="Extracting"):
            hist_chars = sum(len(m.get("content") or "") for m in sample.get("history", []))
            tr_chars = len(sample.get("tool_response") or "")
            if hist_chars + tr_chars > char_limit:
                skipped += 1
                continue
            built = build_phase2(sample, tokenizer)
            if len(built["input_ids"]) > max_model_len:
                skipped += 1
                continue
            batch_ids.append(built["input_ids"])
            batch_meta.append((idx, built))
            if len(batch_ids) >= batch_size:
                _flush()
        _flush()
        if pending:
            console.print(f"Draining {len(pending)} saves ...")
            for f in pending:
                ok = f.result()
                saved += int(ok); skipped += int(not ok)
    finally:
        if save_pool is not None:
            save_pool.shutdown(wait=True)
        llm.shutdown()

    console.print(f"[green]Done: saved={saved} skipped={skipped}[/green]")


if __name__ == "__main__":
    app()
