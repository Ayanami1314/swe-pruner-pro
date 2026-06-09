"""SWE-Pruner inference server.

Architecture (paper Section 5):
  SGLang serves the agent backbone in a separate process with
  ``--enable-return-hidden-states``; this FastAPI process loads only the
  ~18M FFN pruning head and POSTs to SGLang's ``/generate`` to extract
  hidden states for the latest tool_response region.

Endpoints:
  GET  /health  liveness + SGLang reachability
  POST /prune   {history, tool_call, tool_response, threshold}
                → {pruned_code, kept_lines, original_*, pruned_*, latency_ms}

Key implementation details from the paper:
  - Negative-indexing slice into the returned HS tensor: robust under
    SGLang's radix prefix cache where the response ``hidden_states`` may
    cover only the suffix that actually went through prefill.
  - ``_skip_template_whitespace``: Qwen3's chat template inserts ``\\n``
    around ``<tool_response>...</tool_response>``; skipping those bytes
    keeps conversation token IDs aligned with standalone tokenization so
    char→line mapping (via offset_mapping) is exact.
  - Binary HS envelope: server may send fp16 HS as base64-packed bytes for
    2× bandwidth savings. fp16→fp32 is lossless for finite values.
"""
from __future__ import annotations

import json as _json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybase64
import requests
import torch
import typer
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from ..model import PruningHead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class PruneRequest(BaseModel):
    history: List[dict]
    tool_call: dict
    tool_response: str
    threshold: float = 0.5
    output_format: str = "filtered_markers"


class PruneResponse(BaseModel):
    pruned_code: str
    kept_lines: List[int]
    original_lines: int
    kept_line_count: int
    original_chars: int
    pruned_chars: int
    original_tokens: int = 0
    pruned_tokens: int = 0
    latency_ms: float
    error_msg: Optional[str] = None


# Globals populated by serve(); module-private so the FastAPI handler closes
# over the post-init values rather than capturing None at import time.
_pruning_head: Optional[PruningHead] = None
_use_embedded_head: bool = False
_tokenizer = None
_sglang_url: str = ""
_sglang_client: Optional[requests.Session] = None
_max_length = 16384
_tool_response_start_id: Optional[int] = None
_tool_response_end_id: Optional[int] = None
_newline_token_id: Optional[int] = None

_TOOLS = [{
    "name": "bash",
    "description": "Execute a bash command in the repository.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The bash command"}},
        "required": ["command"],
    },
}]


def _apply_chat_template_ids(
    messages: List[dict], tools: Optional[List[dict]] = None,
) -> List[int]:
    """Return token ids regardless of transformers 4.x / 5.x behaviour.

    transformers 5.x's ``apply_chat_template(tokenize=True)`` returns a
    ``BatchEncoding`` whose ``len()`` is the number of keys (always 2),
    silently miscomparing against ``_max_length`` if not normalized.
    """
    out = _tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, tools=tools,
    )
    if isinstance(out, list):
        return out
    ids = out["input_ids"]
    if ids and isinstance(ids[0], list):
        return ids[0]
    return list(ids)


def _sglang_hidden_states(
    input_ids: List[int], hs_start_len: int = -1,
) -> Optional[torch.Tensor]:
    """POST ``/generate`` to SGLang and unpack returned hidden_states.

    ``hs_start_len`` (>=0) caps the radix prefix-cache match so positions
    [hs_start_len:] go through fresh prefill — cached prefix tokens carry
    KV but no HS, otherwise our slice would silently miss the region we
    need.
    """
    payload: dict = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_hidden_states": True,
    }
    if hs_start_len >= 0:
        payload["hidden_states_start_len"] = hs_start_len

    try:
        resp = _sglang_client.post(f"{_sglang_url}/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"[hs] SGLang request failed: {e}")
        return None

    hs_data = data.get("meta_info", {}).get("hidden_states")
    if not hs_data:
        return None

    if isinstance(hs_data, dict) and hs_data.get("__binary__"):
        buf = pybase64.b64decode(hs_data["data"])
        arr = np.frombuffer(buf, dtype=np.dtype(hs_data["dtype"])).reshape(hs_data["shape"])
        hs = torch.from_numpy(arr.copy())
        if hs.dtype != torch.float32:
            n_bad = int(torch.isinf(hs).sum() + torch.isnan(hs).sum())
            if n_bad:
                logger.warning(f"[hs] non-finite values in fp16 HS: count={n_bad}, shape={list(hs.shape)}")
            hs = hs.float()
    else:
        raw_hs = hs_data[0]
        hs = torch.tensor(raw_hs) if isinstance(raw_hs, list) else torch.as_tensor(raw_hs)

    if hs.dim() == 2:
        hs = hs.unsqueeze(0)
    return hs


def _flatten_content(content) -> str:
    """OpenAI-vision list-of-dicts → plain string. Qwen3's template only
    handles plain string content; image/audio parts are dropped (text-only)."""
    if content is None or isinstance(content, str):
        return content or ""
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return str(content)


def _normalize_history(history: List[dict]) -> List[dict]:
    """Coerce OpenAI-shaped history into the dict shape Qwen3's Jinja
    template accepts (string content; tool_calls.arguments as dict)."""
    out: List[dict] = []
    for m in history:
        m = dict(m)
        if "content" in m:
            m["content"] = _flatten_content(m["content"])
        if m.get("role") == "assistant" and m.get("tool_calls"):
            new_tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn = dict(fn)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = _json.loads(args)
                        except (ValueError, _json.JSONDecodeError):
                            fn["arguments"] = {}
                    tc["function"] = fn
                new_tcs.append(tc)
            m["tool_calls"] = new_tcs
        out.append(m)
    return out


def _build_tool_call_message(tool_call: dict) -> dict:
    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else tool_call
    name = fn.get("name", "tool")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = _json.loads(args)
        except (ValueError, _json.JSONDecodeError):
            args = {}
    if not isinstance(args, dict):
        args = {"input": str(args)[:500]}
    return {
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}],
    }


def _window_history(history: List[dict], budget: int) -> List[dict]:
    """Keep most recent complete (assistant, tool) turn pairs within budget.
    Never splits a pair — over-budget samples drop older turns whole."""
    def turn_len(msgs: List[dict]) -> int:
        return len(_apply_chat_template_ids(msgs)) if msgs else 0

    system_msgs = [m for m in history if m.get("role") == "system"]
    non_system = [m for m in history if m.get("role") != "system"]

    system_cost = turn_len(system_msgs)
    if system_cost >= budget:
        return []
    remaining = budget - system_cost

    turns: List[List[dict]] = []
    i = 0
    while i < len(non_system):
        msg = non_system[i]
        if (msg.get("role") == "assistant"
                and i + 1 < len(non_system)
                and non_system[i + 1].get("role") == "tool"):
            turns.append([msg, non_system[i + 1]])
            i += 2
        else:
            turns.append([msg])
            i += 1

    kept: List[List[dict]] = []
    for turn in reversed(turns):
        cost = turn_len(turn)
        if remaining - cost < 0:
            break
        kept.append(turn)
        remaining -= cost
    kept.reverse()

    result = list(system_msgs)
    for t in kept:
        result.extend(t)
    return result


def _skip_template_whitespace(full_ids: List[int], tr_start: int, tr_end: int) -> Tuple[int, int]:
    """Skip ``\\n`` tokens that Qwen3's chat template inserts inside
    ``<tool_response>...</tool_response>`` so subsequent BPE alignment with
    standalone tokenization holds."""
    while tr_start < tr_end and full_ids[tr_start] == _newline_token_id:
        tr_start += 1
    while tr_end > tr_start and full_ids[tr_end - 1] == _newline_token_id:
        tr_end -= 1
    return tr_start, tr_end


def _build_line_ids(tool_response: str) -> torch.Tensor:
    enc = _tokenizer(tool_response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    lines = tool_response.splitlines(keepends=True)
    char_to_line: List[int] = []
    for idx, line in enumerate(lines, 1):
        char_to_line.extend([idx] * len(line))
    line_ids = torch.zeros(len(offsets), dtype=torch.long)
    for i, (tok_s, _) in enumerate(offsets):
        if tok_s < len(char_to_line):
            line_ids[i] = char_to_line[tok_s]
    return line_ids


def _aggregate_to_lines(
    code: str, probs: torch.Tensor, line_ids: torch.Tensor, threshold: float,
) -> Tuple[str, List[int]]:
    """Token probs → kept lines via majority vote; format with filtered markers.

    Single-line gaps between two kept lines are filled to avoid orphans;
    removed runs surface as ``(filtered N lines)`` markers.
    """
    line_scores: Dict[int, List[float]] = defaultdict(list)
    for p, lid in zip(probs.tolist(), line_ids.tolist()):
        if lid > 0:
            line_scores[lid].append(p)

    agg_mode = os.environ.get("PRUNER_LINE_AGG", "vote").lower()
    kept: List[int] = []
    for lid in sorted(line_scores):
        scores = line_scores[lid]
        if agg_mode == "vote":
            keep = sum(1 for s in scores if s >= threshold) > len(scores) / 2
        else:
            keep = (sum(scores) / len(scores)) >= threshold
        if keep:
            kept.append(lid)

    extra = [kept[i] - 1 for i in range(1, len(kept)) if kept[i] - kept[i - 1] == 2]
    kept = sorted(set(kept + extra))

    src = code.splitlines()
    kept_set = set(kept)
    out: List[str] = []
    prev = 0
    for ln in range(1, len(src) + 1):
        if ln in kept_set:
            gap = ln - prev - 1
            if gap > 0:
                out.append(f"(filtered {gap} lines)")
            out.append(src[ln - 1])
            prev = ln
    trailing = len(src) - prev
    if trailing > 0 and prev > 0:
        out.append(f"(filtered {trailing} lines: {prev + 1}-{len(src)})")

    return "\n".join(out), kept


def _passthrough(tool_response: str, t0: float, error_msg: Optional[str] = None) -> PruneResponse:
    n_lines = len(tool_response.splitlines())
    return PruneResponse(
        pruned_code=tool_response,
        kept_lines=list(range(1, n_lines + 1)),
        original_lines=n_lines,
        kept_line_count=n_lines,
        original_chars=len(tool_response),
        pruned_chars=len(tool_response),
        latency_ms=round((time.time() - t0) * 1000, 1),
        error_msg=error_msg,
    )


def _prune(req: PruneRequest) -> PruneResponse:
    t0 = time.time()
    history = _normalize_history(req.history)
    tc_msg = _build_tool_call_message(req.tool_call)
    tr_msg = {"role": "tool", "content": req.tool_response}

    tail_ids = _apply_chat_template_ids([tc_msg, tr_msg])
    if len(tail_ids) >= _max_length:
        return _passthrough(req.tool_response, t0, error_msg="oversize_tail")

    full_ids = _apply_chat_template_ids(list(history) + [tc_msg, tr_msg], tools=_TOOLS)
    if len(full_ids) > _max_length:
        windowed = _window_history(history, _max_length - len(tail_ids))
        full_ids = _apply_chat_template_ids(list(windowed) + [tc_msg, tr_msg], tools=_TOOLS)

    total_len = len(full_ids)

    raw_tr_end = -1
    for i in range(total_len - 1, -1, -1):
        if full_ids[i] == _tool_response_end_id:
            raw_tr_end = i
            break
    raw_tr_start = -1
    if raw_tr_end >= 0:
        for i in range(raw_tr_end - 1, -1, -1):
            if full_ids[i] == _tool_response_start_id:
                raw_tr_start = i + 1
                break
    if raw_tr_start < 0 or raw_tr_end < 0:
        return _passthrough(req.tool_response, t0, error_msg="tr_markers_missing")

    tr_start, tr_end = _skip_template_whitespace(full_ids, raw_tr_start, raw_tr_end)
    if tr_end <= tr_start:
        return _passthrough(req.tool_response, t0, error_msg="empty_tr_region")

    all_hs = _sglang_hidden_states(full_ids, hs_start_len=tr_start)
    if all_hs is None or all_hs.numel() == 0 or all_hs.dim() < 3:
        return _passthrough(req.tool_response, t0, error_msg="hs_invalid")

    returned_len = all_hs.shape[1]
    # Negative indexing keeps the slice correct under prefix cache (returned_len
    # may be < total_len when an earlier prefix was cached).
    hs_start = returned_len - (total_len - tr_start)
    hs_end = returned_len - (total_len - tr_end)
    if hs_start < 0:
        del all_hs
        return _passthrough(req.tool_response, t0, error_msg="cache_overflow")

    tr_hs = all_hs[:, hs_start:hs_end, :]
    del all_hs
    if tr_hs.shape[1] == 0:
        return _passthrough(req.tool_response, t0, error_msg="empty_tr_hs")

    line_ids = _build_line_ids(req.tool_response)
    use_len = min(tr_hs.shape[1], len(line_ids))
    if use_len <= 0:
        return _passthrough(req.tool_response, t0, error_msg="zero_use_len")
    line_ids = line_ids[:use_len]

    logits = _pruning_head.forward(tr_hs[:, :use_len, :], line_ids=line_ids)
    del tr_hs
    probs = torch.sigmoid(logits)
    pruned_code, kept_lines = _aggregate_to_lines(req.tool_response, probs, line_ids, req.threshold)

    original_lines = len(req.tool_response.splitlines())
    kept_set = set(kept_lines)
    pruned_tokens = int(sum(1 for lid in line_ids.tolist() if lid in kept_set))
    latency = (time.time() - t0) * 1000
    logger.info(
        f"[prune] lines {original_lines}→{len(kept_lines)} | "
        f"tokens {use_len}→{pruned_tokens} | chars {len(req.tool_response)}→{len(pruned_code)} | "
        f"thr={req.threshold:.2f} | {latency:.0f}ms"
    )
    return PruneResponse(
        pruned_code=pruned_code,
        kept_lines=kept_lines,
        original_lines=original_lines,
        kept_line_count=len(kept_lines),
        original_chars=len(req.tool_response),
        pruned_chars=len(pruned_code),
        original_tokens=int(use_len),
        pruned_tokens=pruned_tokens,
        latency_ms=round(latency, 1),
    )


def create_app() -> FastAPI:
    app = FastAPI(title="SWE-Pruner Pro")

    @app.get("/health")
    async def health():
        sglang_ok = False
        try:
            r = _sglang_client.get(f"{_sglang_url}/health", timeout=5)
            sglang_ok = r.status_code == 200
        except Exception:
            pass
        return {
            "status": "healthy" if (_pruning_head is not None and sglang_ok) else "degraded",
            "head_loaded": _pruning_head is not None,
            "sglang_healthy": sglang_ok,
            "sglang_url": _sglang_url,
        }

    @app.post("/prune", response_model=PruneResponse)
    def prune(request: PruneRequest):
        if _pruning_head is None:
            raise HTTPException(500, "Pruning head not loaded")
        return _prune(request)

    return app


def main(
    checkpoint: str = typer.Option(
        os.environ.get("PRUNER_CHECKPOINT", ""),
        "--checkpoint",
        help="Checkpoint dir containing model_config.json + best_model.pt",
    ),
    backbone: str = typer.Option(
        os.environ.get("PRUNER_BACKBONE", ""),
        "--backbone",
        help="Path to the backbone model (tokenizer only; weights served by SGLang)",
    ),
    sglang_url: str = typer.Option(
        os.environ.get("SGLANG_URL", "http://localhost:30000"),
        "--sglang-url", "-s",
    ),
    port: int = typer.Option(int(os.environ.get("PRUNER_PORT", "8001")), "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host"),
    device: str = typer.Option(os.environ.get("PRUNER_DEVICE", "cuda:0"), "--device"),
    max_length: int = typer.Option(
        int(os.environ.get("PRUNER_MAX_LENGTH", "16384")), "--max-length",
    ),
    hidden_size: int = typer.Option(
        int(os.environ.get("PRUNER_HIDDEN_SIZE", "2048")),
        "--hidden-size",
        help="Backbone hidden size (e.g. 2048 for Qwen3-30B-A3B)",
    ),
):
    """Start the pruner inference server."""
    if not checkpoint:
        raise typer.BadParameter("--checkpoint is required (or set PRUNER_CHECKPOINT)")
    if not backbone:
        raise typer.BadParameter("--backbone is required (or set PRUNER_BACKBONE)")

    global _pruning_head, _tokenizer, _sglang_url, _sglang_client, _max_length
    global _tool_response_start_id, _tool_response_end_id, _newline_token_id

    _max_length = max_length
    _sglang_url = sglang_url.rstrip("/")
    _sglang_client = requests.Session()

    _tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    _tool_response_start_id = _tokenizer.convert_tokens_to_ids("<tool_response>")
    _tool_response_end_id = _tokenizer.convert_tokens_to_ids("</tool_response>")
    _newline_token_id = _tokenizer.encode("\n", add_special_tokens=False)[0]
    logger.info(
        f"Special tokens: <tool_response>={_tool_response_start_id}, "
        f"</tool_response>={_tool_response_end_id}, \\n={_newline_token_id}"
    )

    logger.info(f"Loading pruning head from {checkpoint} (hidden_size={hidden_size})")
    _pruning_head = PruningHead(checkpoint, hidden_size=hidden_size, device=device)

    logger.info(f"Waiting for SGLang at {_sglang_url}...")
    for attempt in range(60):
        try:
            r = _sglang_client.get(f"{_sglang_url}/health", timeout=5)
            if r.status_code == 200:
                logger.info(f"SGLang healthy after {attempt + 1} attempts")
                break
        except Exception:
            pass
        if attempt % 10 == 0:
            logger.info(f"Waiting for SGLang... (attempt {attempt + 1}/60)")
        time.sleep(5)
    else:
        logger.warning("SGLang not healthy after 5 minutes; starting anyway")

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    typer.run(main)
