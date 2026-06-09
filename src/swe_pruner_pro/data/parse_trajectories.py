"""Parse agent trajectories from HuggingFace datasets into a unified per-step JSONL.

Supports the five corpora used in the paper:

* ``few-sh/terminal-wrench``
* ``TIGER-Lab/SWE-Next-SFT-Trajectories``
* ``ByteDance-Seed/Multi-SWE-bench_trajs``
* ``zai-org/CC-Bench-trajectories``
* ``AweAI-Team/Scale-SWE-Distilled``

Each trajectory is parsed into a list of ``(history, tool_call, tool_response,
next_turn)`` quadruples. ``history`` is the sliding window of previous messages
keeping only complete ``(assistant, tool)`` turn pairs.

Output line schema::

    {
      "instance_id":   str,
      "step_idx":      int,
      "history":       list[dict],   # chat messages
      "tool_call":     {"name": str, "arguments": dict},
      "tool_response": str,
      "next_turn":     list[dict],   # up to 2 messages after this step
      "_source":       str,
      "response_lines": int,
      "code_line_ratio": float,
    }

Quality filter: 15-500 response lines, code-line ratio >= 0.3, not pure error
output, history has at least one prior turn.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import typer
from rich.console import Console

app = typer.Typer(add_completion=False, help="Parse HF trajectory datasets to unified per-step JSONL")
console = Console()

DEFAULT_DATASETS = [
    "few-sh/terminal-wrench",
    "TIGER-Lab/SWE-Next-SFT-Trajectories",
    "ByteDance-Seed/Multi-SWE-bench_trajs",
    "zai-org/CC-Bench-trajectories",
    "AweAI-Team/Scale-SWE-Distilled",
]

# Models we keep from Multi-SWE-bench's mixed-quality dump.
STRONG_MODELS = {"claude-3.5", "claude-3-5", "claude-sonnet", "gpt-4", "o1", "o3", "deepseek"}


# ---------------------------------------------------------------------------
# Heuristic helpers (originally train.scripts.trajectory_explore)
# ---------------------------------------------------------------------------

_CODE_KEYWORDS = re.compile(
    r"^\s*(?:def |class |import |from |return |if |elif |else:|for |while |"
    r"try:|except |raise |with |async |await |yield |lambda |"
    r"function |const |let |var |export |module\.|require\(|"
    r"public |private |protected |package |interface |"
    r"#include|#define|#ifdef|using namespace|"
    r"func |fn |impl |struct |enum |match |"
    r"@|}\s*$|{\s*$)"
)
_LINE_NUMBER_CODE = re.compile(r"^\s*\d+[\t|:]\s*.+")
_BRACKET_CHARS = set("{[(<")

_ERROR_PATTERNS = re.compile(
    r"(?i)^(?:Traceback \(most recent call last\)|"
    r"Error:|error:|ERROR:|FATAL:|fatal:|"
    r"Exception:|exception:|"
    r"command not found|No such file or directory|"
    r"Permission denied|"
    r"SyntaxError:|TypeError:|ValueError:|NameError:|"
    r"ImportError:|ModuleNotFoundError:|"
    r"FileNotFoundError:|OSError:|IOError:|"
    r"RuntimeError:|AttributeError:|KeyError:|IndexError:)"
)


def code_line_ratio(text: str) -> float:
    lines = text.strip().splitlines()
    if not lines:
        return 0.0
    code = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _CODE_KEYWORDS.match(line):
            code += 1
        elif _LINE_NUMBER_CODE.match(line):
            code += 1
        elif len(line) > len(stripped) and any(c in stripped for c in _BRACKET_CHARS):
            code += 1
    return code / len(lines)


def is_error_output(text: str) -> bool:
    lines = text.strip().splitlines()
    if not lines:
        return False
    err = sum(1 for l in lines if _ERROR_PATTERNS.match(l.strip()))
    if err == 0:
        return False
    return err / len(lines) > 0.3


# ---------------------------------------------------------------------------
# Trajectory parsing — generic OpenAI / Anthropic chat formats
# ---------------------------------------------------------------------------

# TIGER-Lab uses inline XML function calls inside assistant content.
_TIGER_FUNC_RE = re.compile(r"<function=(\w+)>(.*?)(?:</function>|$)", re.DOTALL)
_TIGER_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)(?:</parameter>|$)", re.DOTALL)
_TIGER_RESP_PREFIX = re.compile(r"^Execution output of \[\w+\]:\s*\n?")


def _parse_xml_tool_call(content: str) -> Optional[dict]:
    m = _TIGER_FUNC_RE.search(content or "")
    if not m:
        return None
    args = {}
    for pm in _TIGER_PARAM_RE.finditer(m.group(2)):
        args[pm.group(1)] = pm.group(2).strip()
    return {"name": m.group(1), "arguments": args}


def _strip_response_prefix(content: str) -> str:
    m = _TIGER_RESP_PREFIX.match(content or "")
    return content[m.end():] if m else (content or "")


def _normalize_tool_call(tc: dict) -> dict:
    """Accept OpenAI ({"function": {...}}) or Anthropic ({"name", "input"}) forms."""
    if "function" in tc and isinstance(tc["function"], dict):
        fn = tc["function"]
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        return {"name": fn.get("name", "unknown"), "arguments": args}
    if "name" in tc:
        return {"name": tc["name"], "arguments": tc.get("arguments", tc.get("input", {}))}
    return {"name": "unknown", "arguments": {}}


def _anthropic_to_chat(msgs: list[dict]) -> list[dict]:
    """Flatten Anthropic content-block messages to a simple chat list."""
    out: list[dict] = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            text_parts = []
            tool_calls = []
            tool_result = None
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        },
                    })
                elif btype == "tool_result":
                    raw = block.get("content", "")
                    if isinstance(raw, list):
                        raw = "".join(b.get("text", "") for b in raw if isinstance(b, dict))
                    tool_result = raw
            if tool_result is not None:
                out.append({"role": "tool", "content": tool_result})
            elif tool_calls:
                out.append({"role": "assistant", "content": "\n".join(text_parts), "tool_calls": tool_calls})
            else:
                out.append({"role": role, "content": "\n".join(text_parts)})
        else:
            out.append({"role": role, "content": content or ""})
    return out


def parse_chat_trajectory(record: dict) -> list[dict]:
    """Convert one trajectory record to a list of tool-step dicts.

    Tries (in order):
    1. OpenAI-style ``messages`` with ``tool_calls`` field.
    2. Anthropic-style content blocks (CC-Bench, claude-opus dumps).
    3. Inline XML function calls in assistant content (TIGER-Lab).
    """
    instance_id = record.get("instance_id") or record.get("id") or record.get("task_id") or ""
    msgs = (
        record.get("messages")
        or record.get("trajectory")
        or record.get("conversation")
        or []
    )
    if msgs and isinstance(msgs[0], dict) and isinstance(msgs[0].get("content"), list):
        msgs = _anthropic_to_chat(msgs)
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("message"):
        # CC-Bench wraps each message under a "message" key
        msgs = _anthropic_to_chat([m["message"] for m in msgs if isinstance(m, dict) and m.get("message")])

    steps: list[dict] = []
    step_idx = 0
    for i, msg in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        # Form 1/2: explicit tool_calls field.
        tool_calls = msg.get("tool_calls") or []
        tool_call = None
        if tool_calls:
            tool_call = _normalize_tool_call(tool_calls[0])
        else:
            # Form 3: XML-embedded.
            tool_call = _parse_xml_tool_call(msg.get("content", ""))
        if not tool_call:
            continue

        # Pair with following tool message.
        if i + 1 >= len(msgs) or msgs[i + 1].get("role") != "tool":
            continue
        tool_response = _strip_response_prefix(msgs[i + 1].get("content", "") or "")

        history = list(msgs[:i])
        next_turn = list(msgs[i + 2: i + 4])
        steps.append({
            "instance_id": instance_id,
            "step_idx": step_idx,
            "tool_call": tool_call,
            "tool_response": tool_response,
            "history": history,
            "next_turn": next_turn,
        })
        step_idx += 1
    return steps


# ---------------------------------------------------------------------------
# Loading & filtering
# ---------------------------------------------------------------------------

def download_dataset(repo_id: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download
    local_dir = cache_dir / repo_id.replace("/", "_")
    if local_dir.exists() and any(
        p.suffix in {".jsonl", ".json", ".parquet"} for p in local_dir.iterdir() if p.is_file()
    ):
        console.print(f"  [cached] {local_dir}")
        return local_dir
    console.print(f"  Downloading {repo_id} ...")
    snapshot_download(repo_id, local_dir=str(local_dir), repo_type="dataset")
    return local_dir


def load_records(dataset_dir: Path) -> list[dict]:
    records: list[dict] = []
    traj_files = sorted(dataset_dir.rglob("*.traj.json"))
    if traj_files:
        for f in traj_files:
            try:
                data = json.loads(f.read_text())
                data.setdefault("instance_id", f.parent.name)
                records.append(data)
            except Exception:
                pass
        if records:
            return records

    for f in sorted(dataset_dir.rglob("*.jsonl")) + sorted(dataset_dir.rglob("*.json")):
        if f.name.startswith(".") or "readme" in f.name.lower():
            continue
        try:
            if f.suffix == ".jsonl":
                for line in open(f):
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            else:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict):
                    records.append(data)
        except Exception as e:
            console.print(f"    [warn] {f.name}: {e}")

    if not records:
        parquets = sorted(dataset_dir.rglob("*.parquet"))
        if parquets:
            import pandas as pd  # noqa: WPS433 — optional dep
            for f in parquets:
                records.extend(pd.read_parquet(f).to_dict("records"))
    return records


def filter_strong_models(records: list[dict], repo_id: str) -> list[dict]:
    if "Multi-SWE-bench" not in repo_id:
        return records
    out: list[dict] = []
    counts: Counter = Counter()
    for r in records:
        model = str(r.get("model_name_or_path", r.get("model_name", r.get("data_source", "")))).lower()
        counts[model] += 1
        if any(m in model for m in STRONG_MODELS):
            out.append(r)
    console.print(f"    Multi-SWE-bench: kept {len(out)}/{len(records)} (strong models)")
    return out


def quality_filter(steps: Iterable[dict], min_lines: int = 15, max_lines: int = 500,
                    min_code_ratio: float = 0.3) -> tuple[list[dict], Counter]:
    out: list[dict] = []
    reasons: Counter = Counter()
    for s in steps:
        resp = s["tool_response"]
        n = len(resp.strip().splitlines())
        cr = code_line_ratio(resp)
        if n < min_lines:
            reasons["too_short"] += 1; continue
        if n > max_lines:
            reasons["too_long"] += 1; continue
        if cr < min_code_ratio:
            reasons["low_code_ratio"] += 1; continue
        if is_error_output(resp):
            reasons["error_output"] += 1; continue
        if len(s.get("history", [])) < 2:
            reasons["no_history"] += 1; continue
        s["response_lines"] = n
        s["code_line_ratio"] = cr
        out.append(s)
    return out, reasons


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    output: Path = typer.Option("filtered_steps.jsonl", "-o", help="Output JSONL"),
    cache_dir: Path = typer.Option(Path.cwd() / "hf_cache", "--cache-dir", help="HuggingFace download cache"),
    datasets: Optional[list[str]] = typer.Option(None, "--dataset", help="HF repo IDs (repeatable). Defaults to the paper's five sources."),
    min_lines: int = typer.Option(15, help="Minimum tool_response lines"),
    max_lines: int = typer.Option(500, help="Maximum tool_response lines"),
    min_code_ratio: float = typer.Option(0.3, help="Minimum code-line ratio"),
):
    """Download + parse + filter the trajectory corpora into a single JSONL."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    repos = list(datasets) if datasets else DEFAULT_DATASETS

    all_steps: list[dict] = []
    summary: list[dict] = []
    for repo_id in repos:
        console.print(f"\n[bold]== {repo_id} ==[/bold]")
        try:
            dpath = download_dataset(repo_id, cache_dir)
            records = load_records(dpath)
            console.print(f"  Loaded {len(records)} records")
            records = filter_strong_models(records, repo_id)
            short_name = repo_id.split("/")[-1][:15]
            steps: list[dict] = []
            for r in records:
                try:
                    parsed = parse_chat_trajectory(r)
                except Exception:
                    continue
                for s in parsed:
                    if not s["instance_id"].startswith(short_name[:10]):
                        s["instance_id"] = f"{short_name}_{s['instance_id']}"
                    s["_source"] = repo_id
                steps.extend(parsed)
            console.print(f"  Extracted {len(steps)} tool steps")
            kept, reasons = quality_filter(steps, min_lines, max_lines, min_code_ratio)
            console.print(f"  Filter passed: {len(kept)}  ({dict(reasons)})")
            all_steps.extend(kept)
            summary.append({"repo": repo_id, "records": len(records), "steps": len(steps), "kept": len(kept)})
        except Exception as e:
            console.print(f"  [red]ERROR[/red] {e}")
            summary.append({"repo": repo_id, "error": str(e)})

    # Dedup
    seen: set = set()
    deduped: list[dict] = []
    for s in all_steps:
        key = (s["instance_id"], s["step_idx"], s["tool_response"][:200])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    console.print(f"\n[green]Wrote {len(deduped)} steps to {output}[/green]")
    for row in summary:
        console.print(f"  {row}")


if __name__ == "__main__":
    app()
