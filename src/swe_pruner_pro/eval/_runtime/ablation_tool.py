"""Ablation-mode tool / prompt helpers (shared by all multi_turn benchmarks).

When an `--ablation-backend X` flag is set on agent_eval.py, the bash tool
must accept an extra `context_focus_question` parameter. Baselines use this
field as the query for their pruning logic (RAG/LLMLingua/SelfPrune/...).

This module exposes:

* ``wrap_bash_tool_with_focus_question(tool)`` — return a deep-ish copy of
  the bash tool spec with `context_focus_question` added as a required
  parameter. Idempotent: if the field already exists the spec is returned
  unchanged.

* ``ABLATION_PROMPT_SUFFIX`` — append to the existing SYSTEM_PROMPT so the
  agent knows to populate `context_focus_question` on every bash call.

* ``derive_query(tool_call)`` — pull the focus question out of a tool_call
  dict (server-side helper for baselines). Returns "" if missing.

* ``ABLATION_BACKENDS`` — the list of supported backend names; useful for
  CLI flag validation in agent_eval.py.

Per CLAUDE.md guidance: NO fallback. If `context_focus_question` is missing
the server should passthrough; baselines must not run with synthetic queries.
"""

from __future__ import annotations

import copy
from typing import Any


ABLATION_BACKENDS = (
    "llmlingua2",
    "longcodezip",
    "selective_context",
    "self_prune",
    "rerank",
    "swe_pruner",
)


ABLATION_PROMPT_SUFFIX = """

## Ablation: Context Focus Question

Every call to the `bash` tool MUST include a `context_focus_question` argument: \
a short natural-language phrase describing what you are looking for in the \
output of this specific command. Examples:
  - "definition of class FooBar"
  - "where is X imported and used"
  - "callers of method baz()"

This focus question is used by the output-pruning module to keep only the \
lines relevant to the question. A vague or missing focus question disables \
pruning for that turn.
"""


def wrap_bash_tool_with_focus_question(tool: dict) -> dict:
    """Return a copy of the bash tool with `context_focus_question` added.

    The input is expected to follow the OpenAI tool-spec shape:
        {"type": "function", "function": {"name": "bash", "parameters": {...}}}

    Required parameter is added so the schema-following agents (Qwen3,
    mimo) reliably emit the field; missing-field handling lives server-side.
    """
    out = copy.deepcopy(tool)
    fn = out.get("function") or {}
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}

    if "context_focus_question" in props:
        return out

    props["context_focus_question"] = {
        "type": "string",
        "description": (
            "A short phrase describing what specific information you are "
            "trying to extract from this command's output (used by the "
            "output-pruning module as the relevance query). Required."
        ),
    }
    params["properties"] = props

    required = list(params.get("required") or [])
    if "context_focus_question" not in required:
        required.append("context_focus_question")
    params["required"] = required

    fn["parameters"] = params
    out["function"] = fn
    return out


def derive_query(tool_call: dict[str, Any]) -> str:
    """Extract `context_focus_question` from a tool_call dict.

    Accepts both the OpenAI shape {"function": {"arguments": {...}}} and the
    pruner-internal shape {"name": ..., "arguments": {...}}. Returns "" when
    the field is missing or non-string — server passthroughs on empty.
    """
    args = tool_call.get("arguments")
    if args is None:
        fn = tool_call.get("function")
        if isinstance(fn, dict):
            args = fn.get("arguments")
    if isinstance(args, dict):
        q = args.get("context_focus_question")
        if isinstance(q, str):
            return q.strip()
    return ""
