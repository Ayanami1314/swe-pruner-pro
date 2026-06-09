"""Label per-line keep/prune decisions on agent trajectories using Claude.

Reads ``ANTHROPIC_API_KEY`` (and optionally ``ANTHROPIC_BASE_URL``) from the
environment. Input is a JSONL of steps with ``next_turn`` already injected (run
``parse_trajectories`` then ``submodular_sample`` first). Output appends one
labelled record per line — supports resume.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from tqdm import tqdm

from swe_pruner_pro.prompts.label import LABEL_TEMPLATE, SYSTEM_PROMPT

app = typer.Typer(add_completion=False, help="Label trajectory steps with Claude")
console = Console()

DEFAULT_MODEL = "claude-sonnet-4-6"

_INTERNAL_LINE_NUM = re.compile(r"^\s{0,6}\d{1,6}[\t ]{1,2}")


# ---------------------------------------------------------------------------
# Prompt construction (mirrors train.core.prompts.trajectory_label)
# ---------------------------------------------------------------------------

def _format_msgs(msgs: list[dict], cap: int = 1500) -> str:
    if not msgs:
        return "(none)"
    parts: list[str] = []
    for m in msgs:
        role = m.get("role", "?")
        content = str(m.get("content", "") or "")
        if len(content) > cap:
            content = content[:cap] + "..."
        parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


def format_history_window(history: list[dict], n_past: int = 4) -> str:
    recent = history[-n_past:] if len(history) > n_past else history
    return _format_msgs(recent) if recent else "(no prior conversation)"


def number_lines(code: str) -> str:
    """Add outer L<N>| prefixes; strip inner cat -n numbering to avoid confusion."""
    return "\n".join(
        f"L{i+1}| {_INTERNAL_LINE_NUM.sub('', line)}"
        for i, line in enumerate(code.splitlines())
    )


def build_user_prompt(step: dict) -> tuple[str, int]:
    tc = step.get("tool_call", {}) or {}
    tool_name = tc.get("name", "unknown")
    args = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
    if len(args) > 500:
        args = args[:500] + "..."
    code = step.get("tool_response", "") or ""
    n_lines = len(code.splitlines())
    prompt = LABEL_TEMPLATE.format(
        history=format_history_window(step.get("history", []), n_past=4),
        tool_name=tool_name,
        tool_args=args,
        n_lines=n_lines,
        numbered_code=number_lines(code),
        next_turn=_format_msgs(step.get("next_turn", []) or []) or "(end of trajectory)",
    )
    return prompt, n_lines


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            return None
    return None


def _expand_kept_lines(items: list) -> list[int]:
    out: list[int] = []
    for it in items:
        if isinstance(it, int):
            out.append(it)
        elif isinstance(it, str) and "-" in it:
            a, b = it.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                out.extend(range(int(a), int(b) + 1))
        elif isinstance(it, float):
            out.append(int(it))
    return sorted(set(out))


def parse_label(text: str, n_lines: int) -> tuple[list[int], Optional[str], Optional[str]]:
    parsed = _parse_json_response(text)
    if parsed and isinstance(parsed, dict):
        kept = _expand_kept_lines(parsed.get("kept_lines", []))
        return ([n for n in kept if 1 <= n <= n_lines],
                parsed.get("reasoning"), parsed.get("confidence"))
    return [], None, None


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

class ClaudeLabeler:
    def __init__(self, model: str, max_tokens: int = 1024, temperature: float = 0.3):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
        kwargs = {"api_key": api_key}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def label(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.content[0].text


def _seen_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    keys: set[str] = set()
    for line in open(output_path):
        try:
            d = json.loads(line)
            keys.add(f"{d.get('instance_id', '')}_{d.get('step_idx', '')}")
        except json.JSONDecodeError:
            pass
    return keys


@app.command()
def main(
    input_jsonl: Path = typer.Argument(..., help="Sampled JSONL with next_turn injected"),
    output: Path = typer.Option("trajectory_labeled.jsonl", "-o"),
    model: str = typer.Option(DEFAULT_MODEL, help="Anthropic model id"),
    workers: int = typer.Option(8, "-j", help="Parallel API workers"),
    max_code_chars: int = typer.Option(60000, help="Skip steps with longer tool_response"),
    temperature: float = typer.Option(0.3),
    max_tokens: int = typer.Option(1024),
    limit: int = typer.Option(0, help="Process at most N samples (0 = all)"),
):
    """Label tool-response lines with Claude. Resumes on the same output file."""
    seen = _seen_keys(output)
    console.print(f"Already labelled: {len(seen)}")

    todo: list[dict] = []
    skipped_long = 0
    for line in open(input_jsonl):
        line = line.strip()
        if not line:
            continue
        step = json.loads(line)
        key = f"{step.get('instance_id', '')}_{step.get('step_idx', '')}"
        if key in seen:
            continue
        if len(step.get("tool_response", "") or "") > max_code_chars:
            skipped_long += 1
            continue
        todo.append(step)
        if limit and len(todo) >= limit:
            break
    console.print(f"To process: {len(todo)} (skipped {skipped_long} for length)")
    if not todo:
        return

    labeler = ClaudeLabeler(model=model, max_tokens=max_tokens, temperature=temperature)

    def _one(step: dict) -> Optional[dict]:
        prompt, n_lines = build_user_prompt(step)
        try:
            text = labeler.label(prompt)
        except Exception as e:
            console.print(f"  [red]err[/red] {step.get('instance_id')} #{step.get('step_idx')}: {e}")
            return None
        kept, reasoning, confidence = parse_label(text, n_lines)
        return {
            "instance_id": step.get("instance_id", ""),
            "step_idx": step.get("step_idx", 0),
            "tool_call": step.get("tool_call", {}),
            "tool_response": step.get("tool_response", ""),
            "history": step.get("history", []),
            "next_turn": step.get("next_turn", []),
            "kept_frags": kept,
            "total_lines": n_lines,
            "reasoning": reasoning,
            "confidence": confidence,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "a") as fout, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, s): s for s in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Labelling"):
            rec = fut.result()
            if rec is None:
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    app()
