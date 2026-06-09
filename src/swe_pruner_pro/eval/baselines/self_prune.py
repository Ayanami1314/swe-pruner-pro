"""Self-prune baseline.

Asks the same backbone served by the local SGLang to select which lines of
the tool_response are relevant to the focus question. We use ``json``
guided output for robustness — the model returns a list of line numbers to
keep, and we reconstruct the same "(filtered N lines: X-Y)" output format
the main pruner produces so downstream agents see a consistent shape.

This baseline reuses the LLM that's already running for inference; no
extra GPU is needed beyond what SGLang occupies. It DOES bottleneck on
SGLang's request queue, so high-concurrency runs will see latency stack.
"""

from __future__ import annotations

import json
import logging
import re
import time

import requests

from .types import BaselineResult

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a precision filter. Given a code/tool output and a focus "
    "question, return ONLY the line numbers (1-indexed) that are relevant "
    "to answering the question. Output strict JSON: "
    '{"keep_lines": [<int>, ...]} with no extra text. '
    "Be aggressive — drop noise, comments, blank lines, and unrelated "
    "imports/declarations."
)


def _build_user_prompt(query: str, tool_response: str) -> str:
    numbered = "\n".join(
        f"{i:>5} | {line}" for i, line in enumerate(tool_response.splitlines(), 1)
    )
    return (
        f"Focus question: {query}\n\n"
        f"Tool output (line-numbered):\n```\n{numbered}\n```\n\n"
        f"Return JSON: {{\"keep_lines\": [<line numbers to keep>]}}"
    )


def _parse_keep_lines(text: str) -> list[int]:
    """Parse the model's JSON response. Tolerant of code-fence wrapping."""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.I).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    raw = obj.get("keep_lines") or []
    out: list[int] = []
    for v in raw:
        try:
            out.append(int(v))
        except (ValueError, TypeError):
            continue
    return sorted(set(ln for ln in out if ln > 0))


class SelfPrunePruner:
    name = "self_prune"

    def __init__(
        self,
        sglang_url: str,
        model: str = "",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ):
        self.sglang_url = sglang_url.rstrip("/")
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        logger.info(f"[self_prune] target SGLang={self.sglang_url}, model={model or '<auto>'}")

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        try:
            r = self.session.get(f"{self.sglang_url}/v1/models", timeout=10)
            r.raise_for_status()
            data = r.json().get("data") or []
            if data and data[0].get("id"):
                self.model = data[0]["id"]
                return self.model
        except Exception as exc:
            logger.warning(f"[self_prune] could not auto-resolve model: {exc}")
        return ""

    def prune(
        self,
        *,
        history: list[dict],
        tool_call: dict,
        tool_response: str,
        threshold: float,
        query: str,
    ) -> BaselineResult:
        t0 = time.time()
        try:
            model = self._resolve_model()
            if not model:
                return BaselineResult.passthrough(
                    tool_response,
                    latency_ms=(time.time() - t0) * 1000,
                    error_msg="self_prune: SGLang model name not resolvable",
                )

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(query, tool_response)},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
                "response_format": {"type": "json_object"},
            }
            r = self.session.post(
                f"{self.sglang_url}/v1/chat/completions",
                json=payload, timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
            keep_set = set(_parse_keep_lines(text))
            if not keep_set:
                return BaselineResult.passthrough(
                    tool_response,
                    latency_ms=(time.time() - t0) * 1000,
                    error_msg="self_prune: model returned empty keep_lines",
                )

            src_lines = tool_response.splitlines()
            out_parts: list[str] = []
            prev_ln = 0
            for ln in range(1, len(src_lines) + 1):
                if ln in keep_set:
                    gap = ln - prev_ln - 1
                    if gap > 0:
                        if prev_ln == 0:
                            out_parts.append(f"(filtered {gap} lines: 1-{ln - 1})")
                        else:
                            out_parts.append(f"(filtered {gap} lines: {prev_ln + 1}-{ln - 1})")
                    out_parts.append(src_lines[ln - 1])
                    prev_ln = ln
            trailing = len(src_lines) - prev_ln
            if trailing > 0 and prev_ln > 0:
                out_parts.append(f"(filtered {trailing} lines: {prev_ln + 1}-{len(src_lines)})")
            pruned = "\n".join(out_parts)

            return BaselineResult(
                pruned_code=pruned,
                kept_lines=sorted(keep_set),
                original_lines=len(src_lines),
                kept_line_count=len(keep_set),
                original_chars=len(tool_response),
                pruned_chars=len(pruned),
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:
            logger.exception("[self_prune] prune failed")
            return BaselineResult.passthrough(
                tool_response,
                latency_ms=(time.time() - t0) * 1000,
                error_msg=f"self_prune: {type(exc).__name__}: {exc}",
            )
