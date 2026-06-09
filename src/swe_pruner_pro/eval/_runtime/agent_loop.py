"""Shared agent loop used by every multi-turn benchmark.

`run_agent_loop` wraps the chat-completion → tool-execute → maybe-prune →
append-tool-message cycle that previously lived in every
`agent_eval.py`. Per-benchmark variation is expressed through three
optional callbacks (no subclassing, no sentinels):

* ``is_terminal_tool(name, args) -> (bool, answer)``
      For tools that end the run with a specific answer (longcodeqa's
      ``finish`` tool submitting A/B/C/D). When the tuple's first element
      is True, the loop records ``answer`` as the final answer, appends a
      synthetic ``"Submitted."`` tool message, and breaks out of the outer
      iteration. Default: no tool is terminal.

* ``resolve_tool_output(name, args) -> str | None``
      For tools handled without the sandbox (e.g. future non-bash tools).
      Returning a string short-circuits the sandbox and feeds that string
      into the prune/append path. Returning ``None`` falls through to the
      sandbox. Default: always None.

* ``on_empty_answer(messages, client, model, stats) -> str``
      Runs once after the iteration loop finishes if ``answer_text == ""``.
      Trail uses it to force a final-JSON chat call; sweqa uses it to
      synthesize a summary answer from exploration history. Default: fall
      back to the last assistant content (matching the previous behavior).

The loop also takes a ``tool_call_fallback_parser`` that gets threaded into
``chat_completion_with_retry`` — pass ``maybe_mimo_parser(model)`` from each
benchmark to pick up mimo-v2-flash transparently.
"""

from __future__ import annotations

import json
from typing import Callable

import openai

from .ablation_tool import derive_query
from .chat import chat_completion_with_retry
from .models import sampling_params, system_prefix
from .prune_hooks import (
    DEFAULT_POST_HOOKS,
    DEFAULT_PRE_HOOKS,
    PruneContext,
    PrunePostContext,
    run_post_hooks,
    run_pre_hooks,
)
from .pruner_client import call_pruner, sanitize_history_for_prune
from .sandbox import Sandbox
from .stats import RunStats


IsTerminalTool = Callable[[str, dict], tuple[bool, str]]
ResolveToolOutput = Callable[[str, dict], "str | None"]
OnEmptyAnswer = Callable[[list[dict], openai.OpenAI, str, RunStats], str]


def _should_prune(output: str, strategy: str, min_chars: int) -> bool:
    if strategy == "none":
        return False
    if strategy == "always":
        return True
    return len(output) > min_chars  # "threshold"


def _coerce_threshold(raw, default: float) -> float | None:
    if raw is None:
        return default
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return default
    if val <= 0.0:
        return None  # disabled
    return max(0.01, min(1.0, val))


def run_agent_loop(
    *,
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
    sandbox: Sandbox,
    workspace: str,
    max_iterations: int = 40,
    temperature: float | None = None,
    # Pruning config
    experiment: str = "baseline",
    pruner_url: str = "",
    prune_strategy: str = "threshold",
    prune_min_chars: int = 2000,
    prune_threshold: float = 0.5,
    pruner_backend: str = "",  # "" or "ours" → main pruner; non-empty → ablation
    pre_hooks=None,
    post_hooks=None,
    # Model quirks
    tool_call_fallback_parser=None,
    # Output-format extras
    include_per_turn_usage: bool = False,
    # Per-benchmark escape hatches
    is_terminal_tool: IsTerminalTool | None = None,
    resolve_tool_output: ResolveToolOutput | None = None,
    on_empty_answer: OnEmptyAnswer | None = None,
) -> tuple[str, dict, list[dict]]:
    """Drive a single agent trajectory to completion.

    Returns (answer_text, stats_dict, messages).
    """
    # Prepend the model-specific system prologue (e.g. mimo's identity +
    # current-date stanza). Empty string for models that don't need one.
    prefix = system_prefix(model)
    full_system = f"{prefix}\n\n{system_prompt}" if prefix else system_prompt
    messages: list[dict] = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_prompt},
    ]
    stats = RunStats()
    answer_text = ""
    pre_hooks_val = DEFAULT_PRE_HOOKS if pre_hooks is None else pre_hooks
    post_hooks_val = DEFAULT_POST_HOOKS if post_hooks is None else post_hooks
    should_run_pruner = bool(pruner_url) and experiment == "pruner"
    # Model-family sampling config (mimo T=0.3/top_p=0.95 + enable_thinking,
    # qwen3 T=0.7/top_p=0.8/top_k=20/rep_pen=1.05). Caller's explicit
    # `temperature` arg overrides the family default.
    family_kwargs = sampling_params(model)
    if temperature is not None:
        family_kwargs["temperature"] = temperature

    terminated = False
    for iteration in range(max_iterations):
        stats.total_iterations = iteration + 1

        resp, fallback_used = chat_completion_with_retry(
            client,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            tool_call_fallback_parser=tool_call_fallback_parser,
            **family_kwargs,
        )
        stats.mimo_parse_fallbacks += fallback_used
        usage = resp.usage
        stats.add_usage(usage)

        choice = resp.choices[0]
        msg = choice.message
        msg_dict: dict = {"role": "assistant", "content": msg.content or ""}
        # Thinking-mode models (mimo, Qwen3 with enable_thinking) return their
        # reasoning in `reasoning_content`. The model expects that field to be
        # echoed back in subsequent requests so it can continue the thought
        # thread across turns — drop it and multi-turn tool-use degrades.
        # chat_completion_with_retry has already promoted reasoning_content
        # into msg.content (with tool-call XML stripped) for downstream text
        # consumers; keep the raw reasoning in the dict for the next request.
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            msg_dict["reasoning_content"] = reasoning
        if include_per_turn_usage and usage:
            msg_dict["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if choice.finish_reason == "stop" and not msg.tool_calls:
            answer_text = msg.content or ""
            break

        if not msg.tool_calls:
            continue

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            name = tc.function.name

            # (a) Terminal tool — e.g. longcodeqa's `finish`.
            if is_terminal_tool is not None:
                is_term, term_answer = is_terminal_tool(name, args)
                if is_term:
                    answer_text = term_answer
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id, "content": "Submitted.",
                    })
                    terminated = True
                    break

            # (b) Tool output resolved outside sandbox.
            output: str | None = None
            if resolve_tool_output is not None:
                override = resolve_tool_output(name, args)
                if override is not None:
                    output = override

            # (c) Default: execute in sandbox. Command arg key is "command".
            if output is None:
                command = args.get("command", "echo 'no command'")
                output = sandbox.exec(command, workspace)
            else:
                command = args.get("command", "")

            # (d) Threshold override from tool args, or KEEP_ALL marker opt-out.
            if "KEEP_ALL_IN_THIS_COMMAND" in command:
                effective_threshold = None
            else:
                effective_threshold = _coerce_threshold(args.get("output_threshold"),
                                                       prune_threshold)

            # (e) Prune path.
            if (should_run_pruner
                    and effective_threshold is not None
                    and _should_prune(output, prune_strategy, prune_min_chars)):
                tool_call_dict = {"name": name, "arguments": args}
                original_output = output
                ctx = PruneContext(
                    messages=messages,
                    tool_call=tool_call_dict,
                    tool_response=original_output,
                    iteration=iteration,
                )
                decision = run_pre_hooks(ctx, pre_hooks_val)
                if decision and decision.skip:
                    stats.prune_skipped.append({
                        "iteration": iteration,
                        "reason": decision.reason,
                        "metadata": decision.metadata,
                        "command": command[:200],
                    })
                else:
                    pruned, pstats = call_pruner(
                        pruner_url,
                        sanitize_history_for_prune(messages[:-1]),
                        tool_call_dict,
                        original_output,
                        threshold=effective_threshold,
                        tools=tools,
                        pruner_backend=pruner_backend,
                        context_focus_question=derive_query(tool_call_dict),
                    )
                    pstats["iteration"] = iteration
                    pstats["command"] = command[:200]
                    if args.get("output_threshold") is not None:
                        pstats["output_threshold"] = args.get("output_threshold")
                    stats.prune_events.append(pstats)
                    stats.prune_count += 1
                    # post-hooks decorate the pruned output
                    post_ctx = PrunePostContext(
                        ctx=ctx,
                        pruned_code=pruned,
                        original_chars=pstats.get("original_chars", len(original_output)),
                        pruned_chars=pstats.get("pruned_chars", len(pruned)),
                        original_lines=pstats.get("original_lines", 0),
                        kept_line_count=pstats.get("kept_line_count", 0),
                    )
                    output = run_post_hooks(post_ctx, post_hooks_val)

            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": output,
            })
        if terminated:
            break

    if not answer_text:
        if on_empty_answer is not None:
            answer_text = on_empty_answer(messages, client, model, stats) or ""
        if not answer_text:
            for m in reversed(messages):
                if m.get("role") == "assistant" and m.get("content"):
                    answer_text = m["content"]
                    break

    return answer_text, stats.to_dict(), messages
