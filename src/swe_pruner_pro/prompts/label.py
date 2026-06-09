"""Trajectory line-labelling prompt.

Used by Claude Sonnet to annotate per-line keep/prune labels on agent
trajectories during dataset construction. Verbatim from paper appendix
``app:prompts`` ("Trajectory labelling prompt").
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert at understanding AI coding agents. You specialize in \
compressing tool output so that an agent can still navigate and complete its task. \
Your goal is to produce a filtered view that preserves structural context — \
the agent should be able to orient itself in the codebase after filtering.

You always respond with a single JSON object."""


LABEL_TEMPLATE = """\
An AI coding agent is working on a software engineering task. Below is a window \
from its trajectory: recent conversation context, the current tool call + response, \
and what the agent does next.

Your job: decide which lines to KEEP so the agent can still work effectively. \
The lines you remove will be replaced by "(filtered N lines)" markers in the output \
the agent sees. Think of yourself as producing a **skeleton view** — the agent \
should still be able to locate code, understand structure, and find what it needs.

## Previous conversation (context):
{history}

## Current tool call:
Tool: {tool_name}
Arguments: {tool_args}

## Tool response ({n_lines} lines):
Each line is prefixed with "L<number>| " — these are the OUTER line numbers you must use.

{numbered_code}

## What the agent does next:
{next_turn}

---

**CRITICAL**: The tool response may contain internal line numbers (e.g., from `cat -n` \
or editor output showing file line numbers like `    60\tdef foo():`). \
**IGNORE those internal numbers.** Only cite the OUTER line numbers (the "L1|", "L2|", \
"L3|"... prefixes added by this system). For example, if L3 shows `    60\tdef foo():`, \
cite [3], NOT [60].

Decide which lines to KEEP. The kept lines should form a **readable skeleton** of \
the original output. After filtering, the agent will see something like:

```
def authenticate(request):
    token = request.headers.get("Authorization")
(filtered 12 lines)
    if not user.is_active:
        raise PermissionError("Account disabled")
(filtered 8 lines)
def logout(request):
(filtered 5 lines)
```

### What to KEEP:

1. **Lines the agent directly uses next** — code it edits, references, or reasons about
2. **Structural boundaries** — function/class/method signatures, decorator lines, \
closing braces/brackets. When keeping any line inside a block, ALWAYS keep the block's \
opening signature (def/class/if/for/try/with). This is the most important rule — \
the agent needs these landmarks to navigate
3. **Key definitions** — imports, variable assignments, type declarations that the \
agent's focus depends on
4. **Error-relevant lines** — stack traces, error messages, assertion failures, \
test names with PASS/FAIL status
5. **Section headers** — file paths, separators, command output markers that help \
the agent orient in long output

### What to REMOVE:

- Blank lines, pure comment blocks, license headers
- Function bodies that are unrelated to the agent's current focus
- Repetitive output (e.g., long lists where a few examples suffice)
- Verbose boilerplate (import blocks where only 1-2 are relevant)

### Confidence:

- **confident** — You can clearly identify which lines matter from context + next action.
- **skeleton** — You're unsure which specific lines the agent will focus on (e.g., the \
next action targets a different file, the context is ambiguous, or multiple very \
different subsets seem equally valid). Keep only structural skeleton: \
function/class signatures, import lines, section boundaries. Strip body details.

Think step-by-step before producing the JSON:
1. What is the agent trying to accomplish?
2. What does it do next with this output?
3. Which lines does it directly need?
4. Which structural boundaries (function/class signatures) should remain as landmarks?
5. Imagine the filtered output with "(filtered N lines)" gaps — can the agent still work?
6. Am I confident about which specific content lines matter, or should I fall back to skeleton?

Respond with a single JSON object (no markdown fences):
{{
  "reasoning": "<1-2 sentences: what the agent is doing and why these lines matter>",
  "confidence": "confident" or "skeleton",
  "kept_lines": [1, 3, "5-7", "20-30"]
}}

kept_lines supports single numbers and "start-end" range strings."""


def format(
    *,
    history: str,
    tool_name: str,
    tool_args: str,
    n_lines: int,
    numbered_code: str,
    next_turn: str,
) -> str:
    """Render LABEL_TEMPLATE with the supplied fields."""
    return LABEL_TEMPLATE.format(
        history=history,
        tool_name=tool_name,
        tool_args=tool_args,
        n_lines=n_lines,
        numbered_code=numbered_code,
        next_turn=next_turn,
    )
