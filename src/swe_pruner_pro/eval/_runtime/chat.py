"""Thin retry wrapper around `chat.completions.create`.

Three jobs:

1. Retry on `openai.RateLimitError` (up to `max_retries`, linear backoff).
2. Synthesize `msg.tool_calls` from text when the server returned tool calls
   as a non-standard XML blob in `reasoning_content` (the mimo-v2-flash
   failure mode). Only kicks in when:
      - `tool_call_fallback_parser` was provided by the caller
      - `msg.tool_calls` is empty / None
      - the parser returns a non-empty list when given `msg.reasoning_content`
        (falling back to `msg.content`).
3. Promote `reasoning_content` to `msg.content` when the server returned
   `msg.content is None` but has non-empty `reasoning_content` (mimo puts
   *everything*, including the final natural-language answer, into
   `reasoning_content`). We strip the `<tool_call>` XML blocks before the
   promotion so the tool-call markup doesn't leak into the agent's final
   text answer.

Returns the OpenAI response object with the synthesized `tool_calls`
attached to `choice.message` and (for mimo-style responses) a non-empty
`msg.content`. Normal OpenAI-tool-calls paths are untouched.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

import openai

# Strip `<tool_call>…</tool_call>` blocks (mimo-style XML) before promoting
# reasoning_content to content. Same pattern as eval_lib/mimo.py uses for
# parsing; we reuse it here just to delete.
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>.*?</tool_call>\s*", re.DOTALL
)


def _coerce_tool_calls(raw: list[dict]) -> list[Any]:
    """Turn dict-shaped tool calls into SimpleNamespace so `tc.id / tc.function.name /
    tc.function.arguments` access works in caller code (which was written against
    the OpenAI SDK return type)."""
    from types import SimpleNamespace
    coerced = []
    for tc in raw:
        fn = tc.get("function") or {}
        coerced.append(SimpleNamespace(
            id=tc.get("id"),
            type=tc.get("type", "function"),
            function=SimpleNamespace(
                name=fn.get("name"),
                arguments=fn.get("arguments"),
            ),
        ))
    return coerced


def chat_completion_with_retry(
    client: openai.OpenAI,
    *,
    max_retries: int = 3,
    tool_call_fallback_parser: Callable[[str | None], list[dict]] | None = None,
    **kwargs,
) -> tuple[Any, int]:
    """Call `client.chat.completions.create(**kwargs)` with retry + fallback parsing.

    Returns `(response, parsed_fallback_count)`. The counter is 0 when the
    normal OpenAI tool_calls path was used; 1 when we synthesized tool_calls
    from fallback text.

    Retry policy:
      - RateLimitError → back off (3/6/9 s), retry
      - APIConnectionError / APITimeoutError → brief retry (2/4/6 s)
      - 5xx APIStatusError (502/503/504 gateway errors) → fail FAST with one
        quick retry (2 s). Gateway timeouts mean the upstream pod itself
        has stalled; the 504 response already arrived after the gateway's
        own wait window, so client-side retry is mostly useless. Better to
        bubble up and let the caller skip this row (resume picks it up on
        the next pass when the server is healthy again).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except openai.RateLimitError as exc:
            last_exc = exc
            time.sleep(3 * (attempt + 1))
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
        except openai.APIStatusError as exc:
            status = getattr(exc, "status_code", 0) or 0
            if 500 <= status < 600:
                last_exc = exc
                if attempt >= 1:  # at most 2 attempts total for 5xx
                    raise
                time.sleep(2)
            else:
                raise
    else:
        raise last_exc if last_exc else RuntimeError("chat.completions.create retries exhausted")

    parsed_fallback = 0
    if tool_call_fallback_parser is not None:
        choice = resp.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", None)
        if not msg.tool_calls:
            # mimo sometimes emits chain-of-thought in reasoning_content and
            # the actual <tool_call> XML in content (the modern case), and
            # sometimes the other way around (the original docstring case).
            # Try `content` first because it is the model's official output
            # channel — `reasoning_content` is an auxiliary surface that is
            # more likely to contain XML-like text as part of the narrative.
            for text in (msg.content, reasoning):
                if not text:
                    continue
                parsed = tool_call_fallback_parser(text)
                if parsed:
                    msg.tool_calls = _coerce_tool_calls(parsed)
                    # If the model emitted a tool call via fallback, treat this as
                    # "tool_calls" finish even if the server said "stop"; callers
                    # rely on finish_reason to decide whether to break.
                    if choice.finish_reason == "stop":
                        choice.finish_reason = "tool_calls"
                    parsed_fallback = 1
                    break
        # Promote reasoning_content → content when the server put the natural-
        # language answer there (mimo behavior). Stripped of tool-call XML so
        # downstream "final assistant content" consumers don't see markup.
        if not msg.content and reasoning:
            stripped = _TOOL_CALL_BLOCK_RE.sub("", reasoning).strip()
            if stripped:
                msg.content = stripped
    return resp, parsed_fallback
