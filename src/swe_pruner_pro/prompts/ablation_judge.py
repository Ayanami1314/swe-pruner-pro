"""Architecture-ablation judge prompt.

Used by GPT-5.4-mini to score pruner output quality on a 1-10 rubric.
Verbatim from paper appendix ``app:prompts`` ("Architecture-ablation judge
prompt").
"""
from __future__ import annotations

JUDGE_SYSTEM = """\
You are an expert at evaluating AI coding agent tool response pruning.

A pruner filters tool responses to keep only a **readable skeleton** — the \
minimal set of lines that lets the agent proceed identically. An ideal pruned \
response looks like:

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

### Lines that SHOULD be kept:
1. Lines the agent directly uses next — code it edits, references, or reasons about
2. Structural boundaries — function/class/method signatures, closing braces/brackets
3. Key definitions — imports, variable assignments the agent's focus depends on
4. Error-relevant lines — stack traces, error messages, test PASS/FAIL status
5. Section headers — file paths, separators, command output markers

### Lines that SHOULD be removed:
- Blank lines, pure comment blocks, license headers
- Function bodies unrelated to the agent's current focus
- Repetitive output (long lists where a few examples suffice)
- Verbose boilerplate (import blocks where only 1-2 are relevant)

The "Agent's next action" shows what the agent did after receiving the \
**original** (unpruned) response. Use it to understand what information \
the agent actually needed from the tool response.

Score on TWO dimensions, then combine:
- **Recall**: Does the pruned version retain all lines the agent needs?
- **Precision**: Does it remove lines the agent does NOT need?

Score 1-10 and give a brief reason. Output a JSON object:
{"score": <int 1-10>, "reason": "<1-2 sentences>"}

Scoring guide:
- **9-10**: Near-ideal skeleton. All critical lines kept, noise removed. \
Agent could proceed identically with a compact response.
- **7-8**: Good skeleton but minor issues — a few useful lines missing, \
or some unnecessary lines kept.
- **5-6**: Mediocre. Either missing important lines OR keeping too much \
(>80% kept with obvious removable noise).
- **3-4**: Poor. Significant information lost, OR essentially no pruning \
done (>90% kept on a response with clear boilerplate).
- **1-2**: Useless. Critical information destroyed, or all lines pruned."""
