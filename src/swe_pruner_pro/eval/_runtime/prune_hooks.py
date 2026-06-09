"""Pluggable pre-processing hooks for the pruner.

NOTE: structurally mirrors
  downstream_eval/multi_turn/swebench/mini-swe-agent--with-pruning/src/minisweagent/utils/prune_hooks.py
but deliberately diverges in ``_FULLY_PRUNED_HINT`` / ``hook_fully_pruned_hint``:
the swebench fork's agents use ``<output_threshold>`` XML tags embedded in
THOUGHT (no structured tool calls); the benchmarks wired through ``eval_lib``
all use OpenAI-style tool calls where ``output_threshold`` is a tool argument.
The reminder text reflects that. Pre-hooks stay in lockstep — mirror edits to
those across both files.

Problem this solves: even strong models like Opus set `output_threshold` < 30%
of the time, so the agent often re-reads a pruned region hoping to see the
filtered content — which then gets pruned again. This module lets runtimes
short-circuit the pruner when the new tool call clearly targets a region the
agent already saw, giving the raw response back so the re-read actually helps.

Usage (runtime-side):

    from prune_hooks import PruneContext, run_pre_hooks

    ctx = PruneContext(
        messages=messages,                               # chat history BEFORE the new tool response
        tool_call={"name": tc_name, "arguments": args},
        tool_response=output,
        iteration=iteration,
    )
    decision = run_pre_hooks(ctx)
    if decision and decision.skip:
        stats.setdefault("prune_skipped", []).append({
            "iteration": iteration,
            "reason": decision.reason,
            "metadata": decision.metadata,
        })
    else:
        # proceed with existing _call_pruner(...) path
        ...

Adding a new hook: write a `PreHook` (see signature) and prepend/append it to
the list passed to `run_pre_hooks` (or mutate `DEFAULT_PRE_HOOKS`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator


@dataclass
class PruneContext:
    messages: list[dict]
    tool_call: dict
    tool_response: str
    iteration: int | None = None


@dataclass
class PruneDecision:
    skip: bool
    reason: str
    metadata: dict = field(default_factory=dict)


PreHook = Callable[[PruneContext], "PruneDecision | None"]


_FILTERED_MARKER_RE = re.compile(r"\(filtered\s+\d+\s+lines?[^)]*\)", re.IGNORECASE)


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _iter_prior_tool_pairs(messages: list[dict]) -> Iterator[tuple[dict, str]]:
    """Yield (tool_call dict, tool_response str) pairs from prior assistant→tool turns.

    Tool call dict has keys {"id", "name", "arguments"}. Pairing prefers
    `tool_call_id` when present; otherwise falls back to positional order, so
    transcripts without ids (e.g. mini-swe-agent converted chat format) still
    pair correctly.
    """
    pending: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            pending = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                pending.append({
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": _parse_args(fn.get("arguments")),
                })
        elif role == "tool":
            tcid = m.get("tool_call_id")
            match = None
            if tcid is not None:
                for i, p in enumerate(pending):
                    if p["id"] == tcid:
                        match = pending.pop(i)
                        break
            if match is None and pending:
                match = pending.pop(0)
            if match is not None:
                yield match, (m.get("content") or "")


def _nonempty_lines(text: str) -> set[str]:
    """Stripped non-empty lines, excluding `(filtered N lines ...)` markers."""
    out: set[str] = set()
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if _FILTERED_MARKER_RE.search(s):
            continue
        out.add(s)
    return out


def _args_equal(a: dict, b: dict) -> bool:
    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except TypeError:
        return a == b


def hook_repeat_read(
    ctx: PruneContext,
    *,
    min_overlap_lines: int = 3,
    min_overlap_ratio: float = 0.1,
) -> PruneDecision | None:
    """Skip pruning if the new tool call re-reads material we already returned.

    Triggers when either:
      (a) prior tool_call has the same name + same arguments, OR
      (b) current tool_response shares at least `min_overlap_lines` stripped
          non-empty lines with a prior response, AND that overlap is at least
          `min_overlap_ratio` of the smaller line set.
    """
    cur_args = ctx.tool_call.get("arguments") or {}
    cur_name = ctx.tool_call.get("name")
    cur_lines = _nonempty_lines(ctx.tool_response)

    for prior_tc, prior_resp in _iter_prior_tool_pairs(ctx.messages):
        if prior_tc["name"] == cur_name and _args_equal(prior_tc["arguments"], cur_args):
            return PruneDecision(
                skip=True,
                reason="repeat_read_same_args",
                metadata={"prior_args": prior_tc["arguments"]},
            )
        if cur_lines and prior_resp:
            prior_lines = _nonempty_lines(prior_resp)
            if not prior_lines:
                continue
            overlap = cur_lines & prior_lines
            smaller = min(len(cur_lines), len(prior_lines))
            if (len(overlap) >= min_overlap_lines
                    and smaller > 0
                    and len(overlap) / smaller >= min_overlap_ratio):
                return PruneDecision(
                    skip=True,
                    reason="repeat_read_content_overlap",
                    metadata={
                        "overlap_lines": len(overlap),
                        "prior_args": prior_tc["arguments"],
                    },
                )
    return None


# Bash-like tool names whose `arguments.command` we inspect for whitelist match.
_BASH_TOOL_NAMES: frozenset[str] = frozenset({"bash", "shell", "execute_bash", "run_command"})

# Single-word commands whose typical output is large enough that pruning helps.
# Deliberately excludes `ls`, `pwd`, `echo`, `cd`, `mkdir`, `rm`, `test` (bash
# conditional builtin) — those produce small output or shouldn't be pruned.
_LARGE_OUTPUT_SINGLE_CMDS: tuple[str, ...] = (
    # Read / pager
    "cat", "zcat", "bat", "less", "more", "head", "tail",
    # Search
    "grep", "egrep", "fgrep", "rg", "ag", "ack",
    # File walk (can be large)
    "find", "fd", "tree",
    # Text processing (stream-heavy)
    "sed", "awk", "sort", "uniq",
    # Diff
    "diff",
    # Test runners
    "pytest", "unittest", "nosetests", "tox",
    "jest", "mocha", "rspec", "phpunit",
)

# Multi-word invocations we want to catch as "large output".
# Note: `git` (log/diff/show/blame/grep) is intentionally excluded — git output
# is always preserved raw because the agent relies on exact diff / commit text.
_LARGE_OUTPUT_PHRASES: tuple[str, ...] = (
    r"python[0-9]?\s+-m\s+(?:pytest|unittest)",
    r"make\s+(?:test|check)",
    r"npm\s+(?:run\s+)?test",
    r"pnpm\s+test",
    r"yarn\s+test",
    r"go\s+test",
    r"cargo\s+test",
)

_LARGE_OUTPUT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(c) for c in _LARGE_OUTPUT_SINGLE_CMDS) + r")\b"
    + "|" + "|".join(_LARGE_OUTPUT_PHRASES)
)

# Commands whose output must be preserved verbatim — never prune, even if the
# command incidentally matches the whitelist above (e.g. `git diff` → `diff`).
# The agent relies on exact git text (diff hunks, commit SHAs, blame lines).
# Matches `git` when it's the invoked program: at start of command, or right
# after a shell separator (`;`, `&&`, `||`, `|`, `(`, `` ` ``), tolerating
# whitespace between separator and `git`.
_ALWAYS_PRESERVE_RE = re.compile(r"(?:^|[;&|(`]|&&|\|\|)\s*git(?:\s|$|[;&|`])")


def hook_command_whitelist(ctx: PruneContext) -> PruneDecision | None:
    """Skip pruning unless the tool is bash AND the command produces large output.

    Non-bash tools → skipped (pass raw output through).
    `git …` commands → skipped (output preserved verbatim).
    Bash with `ls`, `pwd`, `cd`, short builtins → skipped.
    Bash with `cat`, `grep`, `pytest`, `find`, … → pruner runs.
    """
    name = (ctx.tool_call.get("name") or "").lower()
    if name not in _BASH_TOOL_NAMES:
        return PruneDecision(
            skip=True,
            reason="non_bash_tool",
            metadata={"tool_name": name},
        )
    args = ctx.tool_call.get("arguments") or {}
    command = args.get("command") or args.get("cmd") or ""
    if not isinstance(command, str):
        command = str(command)
    if _ALWAYS_PRESERVE_RE.search(command):
        return PruneDecision(
            skip=True,
            reason="command_always_preserved",
            metadata={"command_head": command[:80]},
        )
    if _LARGE_OUTPUT_RE.search(command):
        return None
    return PruneDecision(
        skip=True,
        reason="command_not_in_whitelist",
        metadata={"command_head": command[:80]},
    )


def hook_early_history(
    ctx: PruneContext,
    *,
    min_rounds: int = 5,
) -> PruneDecision | None:
    """Skip pruning during the first `min_rounds` tool rounds.

    Uses ``ctx.iteration`` when provided (0-indexed); otherwise counts the
    completed ``(tool_call, tool_response)`` pairs in ``ctx.messages``. Early
    rounds are often orientation steps (`pwd`, `ls`, first `cat`) where raw
    context helps the agent build a repo model before heavier exploration.
    """
    if ctx.iteration is not None:
        if ctx.iteration < min_rounds:
            return PruneDecision(
                skip=True,
                reason="early_history_iteration",
                metadata={"iteration": ctx.iteration, "min_rounds": min_rounds},
            )
        return None
    prior_count = sum(1 for _ in _iter_prior_tool_pairs(ctx.messages))
    if prior_count < min_rounds:
        return PruneDecision(
            skip=True,
            reason="early_history_rounds",
            metadata={"prior_rounds": prior_count, "min_rounds": min_rounds},
        )
    return None


DEFAULT_PRE_HOOKS: list[PreHook] = [
    # hook_early_history,
    # hook_command_whitelist,
    # hook_repeat_read,
]


# Registry + spec parser used by every benchmark's CLI. Kept here so the
# spec → callable mapping lives next to the hook definitions themselves.
_PRE_HOOK_REGISTRY: dict[str, PreHook] = {
    "early_history": hook_early_history,
    "command_whitelist": hook_command_whitelist,
    "repeat_read": hook_repeat_read,
}


def resolve_pre_hooks_spec(spec: str) -> "list[PreHook] | None":
    """Turn a CLI spec string into a hook list (or None for defaults).

    ``default``        → None   (fall through to DEFAULT_PRE_HOOKS)
    ``none``           → []     (explicitly no hooks)
    ``all``            → all three hooks in canonical order
    ``a,b,c``          → those hooks, in the order given

    Raises ``ValueError`` for unknown hook names so the CLI layer can wrap the
    message into its own typer.BadParameter / argparse error.
    """
    s = (spec or "").strip().lower()
    if s in ("", "default"):
        return None
    if s == "none":
        return []
    if s == "all":
        return [hook_early_history, hook_command_whitelist, hook_repeat_read]
    out: list[PreHook] = []
    for name in (n.strip() for n in s.split(",")):
        if not name:
            continue
        if name not in _PRE_HOOK_REGISTRY:
            raise ValueError(
                f"unknown hook '{name}' (known: "
                f"{', '.join(sorted(_PRE_HOOK_REGISTRY))}, plus none/all/default)"
            )
        out.append(_PRE_HOOK_REGISTRY[name])
    return out


def run_pre_hooks(
    ctx: PruneContext,
    hooks: list[PreHook] | None = None,
) -> PruneDecision | None:
    for hook in (hooks if hooks is not None else DEFAULT_PRE_HOOKS):
        decision = hook(ctx)
        if decision is not None and decision.skip:
            return decision
    return None


# ── Post-hooks: run on pruner output to annotate / rewrite ─────────────
# A post-hook returns a reminder string to append to the pruned output
# (or None to do nothing). Multiple reminders are joined with a blank line.

@dataclass
class PrunePostContext:
    ctx: PruneContext
    pruned_code: str
    original_chars: int
    pruned_chars: int
    original_lines: int = 0
    kept_line_count: int = 0


PostHook = Callable[[PrunePostContext], "str | None"]


_FULLY_PRUNED_HINT = (
    "The pruner removed every line of this tool output. "
    "If you need the raw content, re-run the same tool call with an "
    "`output_threshold` argument set lower (e.g. 0.2 to keep more lines, "
    "or 0.0 to disable pruning for that call)."
)


def hook_fully_pruned_hint(post: PrunePostContext) -> str | None:
    """Append a system reminder when the pruner kept zero lines.

    Triggers only when the original had real content (``original_lines > 0``)
    and the pruner returned ``kept_line_count == 0`` — i.e. everything was
    replaced by ``(filtered N lines)`` markers. The reminder nudges the agent
    to retry with a lower ``output_threshold`` tool argument (this copy of
    ``prune_hooks.py`` assumes tool-call-based agents; the ``swebench/``
    mini-swe-agent fork has its own copy that speaks ``<output_threshold>``
    XML tags instead).
    """
    if post.original_lines <= 0:
        return None
    if post.kept_line_count > 0:
        return None
    return f"<system_reminder>{_FULLY_PRUNED_HINT}</system_reminder>"


DEFAULT_POST_HOOKS: list[PostHook] = [hook_fully_pruned_hint]


def run_post_hooks(
    post_ctx: PrunePostContext,
    hooks: list[PostHook] | None = None,
) -> str:
    """Apply post-hooks in order, appending their reminders to ``pruned_code``.

    Returns the (possibly augmented) final string. If no hook produces a
    reminder, returns ``post_ctx.pruned_code`` unchanged.
    """
    reminders: list[str] = []
    for hook in (hooks if hooks is not None else DEFAULT_POST_HOOKS):
        out = hook(post_ctx)
        if out:
            reminders.append(out)
    if not reminders:
        return post_ctx.pruned_code
    suffix = "\n".join(reminders)
    if not post_ctx.pruned_code:
        return suffix
    return f"{post_ctx.pruned_code}\n\n{suffix}"


# ── Inline tests ─────────────────────────────────────────────────────────

def _mk_pair(idx: int, command: str, response: str) -> list[dict]:
    """Build one (assistant with bash tool_call, tool) pair for test histories."""
    return [
        {"role": "assistant", "tool_calls": [{
            "id": f"t{idx}",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }]},
        {"role": "tool", "tool_call_id": f"t{idx}", "content": response},
    ]


def _history_with_pairs(n: int, command: str = "cat x", response: str = "x") -> list[dict]:
    """Return ``n`` completed bash tool pairs to push past the early-history hook."""
    history = [{"role": "user", "content": "q"}]
    for i in range(n):
        history.extend(_mk_pair(i, command, response))
    return history


def _run_tests() -> None:
    # Baseline history with 5 completed rounds — bypasses hook_early_history.
    # Use an unusual command/response so it won't collide with later assertions.
    warm = _history_with_pairs(5, command="cat warm_x", response="warm_output")

    # 1. hook_early_history blocks on cold start (no prior rounds).
    cold_ctx = PruneContext(
        messages=[{"role": "user", "content": "q"}],
        tool_call={"name": "bash", "arguments": {"command": "cat foo.py"}},
        tool_response="alpha\nbeta",
    )
    d = run_pre_hooks(cold_ctx)
    assert d is not None and d.skip and d.reason == "early_history_rounds", d

    # 1b. Explicit iteration < min_rounds also trips the hook.
    early_iter = PruneContext(
        messages=warm,
        tool_call={"name": "bash", "arguments": {"command": "cat foo.py"}},
        tool_response="alpha",
        iteration=2,
    )
    d1b = run_pre_hooks(early_iter)
    assert d1b is not None and d1b.skip and d1b.reason == "early_history_iteration", d1b

    # 2. hook_command_whitelist skips non-bash tools.
    non_bash = PruneContext(
        messages=warm,
        tool_call={"name": "read_file", "arguments": {"path": "/x"}},
        tool_response="stuff",
    )
    d2 = run_pre_hooks(non_bash)
    assert d2 is not None and d2.skip and d2.reason == "non_bash_tool", d2

    # 3. hook_command_whitelist skips `ls` / `pwd` style commands.
    for bad_cmd in ("ls src/", "pwd", "echo hi", "cd /tmp", "mkdir foo"):
        ctx = PruneContext(
            messages=warm,
            tool_call={"name": "bash", "arguments": {"command": bad_cmd}},
            tool_response="x\ny\nz",
        )
        dc = run_pre_hooks(ctx)
        assert dc is not None and dc.skip and dc.reason == "command_not_in_whitelist", (bad_cmd, dc)

    # 3b. Whitelisted commands fall through the whitelist + early hooks.
    for good_cmd in (
        "cat foo.py", "grep -r foo .", "find . -name '*.py'",
        "pytest tests/test_x.py", "python -m pytest -k foo",
        "make test", "tail -n 100 log.txt",
    ):
        ctx = PruneContext(
            messages=warm,
            tool_call={"name": "bash", "arguments": {"command": good_cmd}},
            tool_response=f"unique body for <{good_cmd}>",
        )
        assert run_pre_hooks(ctx) is None, f"expected pass-through: {good_cmd}"

    # 3c. All `git …` commands are always preserved verbatim (never pruned),
    # including compound forms and commands where a whitelisted word appears.
    for git_cmd in (
        "git log --oneline",
        "git diff HEAD",
        "git status",
        "git show abc123",
        "cd /testbed && git diff",
        "git log | head -n 50",
    ):
        ctx = PruneContext(
            messages=warm,
            tool_call={"name": "bash", "arguments": {"command": git_cmd}},
            tool_response="lots of diff text\n" * 20,
        )
        dc = run_pre_hooks(ctx)
        assert dc is not None and dc.skip and dc.reason == "command_always_preserved", (git_cmd, dc)

    # 3d. `git` appearing as a search argument (not an invocation) does NOT trigger preserve.
    grep_git = PruneContext(
        messages=warm,
        tool_call={"name": "bash", "arguments": {"command": "grep -rn git src/"}},
        tool_response=f"hit\n" * 20,
    )
    assert run_pre_hooks(grep_git) is None, "grep git should be pruned normally"

    # 4. Same-args repeat (needs warm + whitelisted cmd to reach hook_repeat_read).
    history = warm + _mk_pair(100, "cat foo.py", "line a\nline b\nline c\nline d")
    ctx4 = PruneContext(
        messages=history,
        tool_call={"name": "bash", "arguments": {"command": "cat foo.py"}},
        tool_response="line a\nline b",
    )
    d4 = run_pre_hooks(ctx4)
    assert d4 is not None and d4.skip and d4.reason == "repeat_read_same_args", d4

    # 5. Different args but overlapping content.
    ctx5 = PruneContext(
        messages=history,
        tool_call={"name": "bash", "arguments": {"command": "sed -n '1,3p' foo.py"}},
        tool_response="line a\nline b\nline c\nline e",
    )
    d5 = run_pre_hooks(ctx5)
    assert d5 is not None and d5.skip and d5.reason == "repeat_read_content_overlap", d5

    # 6. Arguments passed as JSON string (OpenAI format).
    history_str = warm + [
        {"role": "assistant", "tool_calls": [{
            "id": "tX",
            "function": {"name": "bash", "arguments": '{"command": "grep foo bar"}'},
        }]},
        {"role": "tool", "tool_call_id": "tX", "content": "x\ny\nz"},
    ]
    ctx6 = PruneContext(
        messages=history_str,
        tool_call={"name": "bash", "arguments": {"command": "grep foo bar"}},
        tool_response="new output",
    )
    d6 = run_pre_hooks(ctx6)
    assert d6 is not None and d6.skip and d6.reason == "repeat_read_same_args", d6

    # 7. Missing tool_call_id falls back to positional pairing.
    history_noid = warm + [
        {"role": "assistant", "tool_calls": [{
            "function": {"name": "bash", "arguments": {"command": "cat a"}},
        }]},
        {"role": "tool", "content": "alpha\nbeta\ngamma\ndelta\nepsilon"},
    ]
    ctx7 = PruneContext(
        messages=history_noid,
        tool_call={"name": "bash", "arguments": {"command": "cat a"}},
        tool_response="x",
    )
    d7 = run_pre_hooks(ctx7)
    assert d7 is not None and d7.skip and d7.reason == "repeat_read_same_args", d7

    # 8. `(filtered N lines)` markers don't count toward overlap.
    history_filt = warm + [
        {"role": "assistant", "tool_calls": [{
            "id": "tF",
            "function": {"name": "bash", "arguments": {"command": "cat big.py"}},
        }]},
        {"role": "tool", "tool_call_id": "tF",
         "content": "head line\n(filtered 200 lines: 5-204)\ntail line"},
    ]
    ctx8 = PruneContext(
        messages=history_filt,
        tool_call={"name": "bash", "arguments": {"command": "cat other.py"}},
        tool_response="(filtered 200 lines: 5-204)\nunique text",
    )
    assert run_pre_hooks(ctx8) is None

    # 9. Post-hook: fully-pruned output gets a <system_reminder> appended.
    stub_ctx = PruneContext(
        messages=[], tool_call={"name": "bash", "arguments": {"command": "cat x"}},
        tool_response="a\nb\nc",
    )
    fully = PrunePostContext(
        ctx=stub_ctx,
        pruned_code="(filtered 3 lines: 1-3)",
        original_chars=10, pruned_chars=25,
        original_lines=3, kept_line_count=0,
    )
    out = run_post_hooks(fully)
    assert out.startswith("(filtered 3 lines: 1-3)"), out
    assert "<system_reminder>" in out and "</system_reminder>" in out, out
    assert "output_threshold" in out, out

    # 9b. Partial prune (some lines kept) → no reminder.
    partial = PrunePostContext(
        ctx=stub_ctx,
        pruned_code="a\n(filtered 1 lines: 2-2)\nc",
        original_chars=10, pruned_chars=20,
        original_lines=3, kept_line_count=2,
    )
    out_partial = run_post_hooks(partial)
    assert "<system_reminder>" not in out_partial, out_partial
    assert out_partial == partial.pruned_code, out_partial

    # 9c. Empty original (original_lines=0) → no reminder.
    empty = PrunePostContext(
        ctx=stub_ctx, pruned_code="", original_chars=0, pruned_chars=0,
        original_lines=0, kept_line_count=0,
    )
    assert run_post_hooks(empty) == ""

    print("prune_hooks: all tests passed")


if __name__ == "__main__":
    _run_tests()
