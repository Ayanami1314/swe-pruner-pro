"""HTTP client for the pruner server, plus `sanitize_history_for_prune`.

Public API:

* `call_pruner(url, history, tool_call, tool_response, *, threshold, tools,
  pruner_backend, context_focus_question, timeout) -> (pruned_code, stats)`
  POST `history / tool_call / tool_response / threshold / tools /
  pruner_backend / context_focus_question` to `$url/prune`, parse the JSON,
  and return the pruned string alongside a stats dict (original_chars,
  pruned_chars, original_lines, kept_line_count, latency_ms, error_msg,
  backend). On any exception, log once and return the raw `tool_response`
  with `{"error": str(exc)}` so the agent keeps going.

  ``pruner_backend`` defaults to ``"ours"``. Set it to one of the
  ablation backends (llmlingua2, longcodezip, selective_context, rerank,
  self_prune, swe_pruner) to route this request to a different pruner on
  the same server.

* `sanitize_history_for_prune(messages)` — filter the chat history down to
  the four roles the pruner understands and strip any extras the OpenAI
  SDK attaches (logprobs, usage, function_call, etc). The pruner doesn't
  need more than role/content/tool_calls/tool_call_id.
"""

from __future__ import annotations

import requests
from rich import print as rprint


def sanitize_history_for_prune(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        new: dict = {"role": role}
        if m.get("content") is not None:
            new["content"] = m["content"]
        if role == "assistant" and m.get("tool_calls"):
            new["tool_calls"] = m["tool_calls"]
        if role == "tool" and m.get("tool_call_id"):
            new["tool_call_id"] = m["tool_call_id"]
        out.append(new)
    return out


def call_pruner(
    url: str,
    history: list[dict],
    tool_call: dict,
    tool_response: str,
    *,
    threshold: float = 0.5,
    tools: list[dict] | None = None,
    pruner_backend: str = "",
    context_focus_question: str = "",
    timeout: int = 120,
) -> tuple[str, dict]:
    payload: dict = {
        "history": history,
        "tool_call": tool_call,
        "tool_response": tool_response,
        "threshold": threshold,
    }
    if tools is not None:
        payload["tools"] = tools
    if pruner_backend:
        payload["pruner_backend"] = pruner_backend
    if context_focus_question:
        payload["context_focus_question"] = context_focus_question
    try:
        r = requests.post(f"{url.rstrip('/')}/prune", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["pruned_code"], {
            "original_chars": data.get("original_chars", len(tool_response)),
            "pruned_chars": data.get("pruned_chars", len(data["pruned_code"])),
            "original_lines": data.get("original_lines", 0),
            "kept_line_count": data.get("kept_line_count", 0),
            "latency_ms": data.get("latency_ms", 0),
            "error_msg": data.get("error_msg"),
            "backend": data.get("backend", pruner_backend or "ours"),
        }
    except Exception as exc:
        rprint(f"  [red]prune failed: {exc}; falling back to raw[/red]")
        return tool_response, {"error": str(exc)}
