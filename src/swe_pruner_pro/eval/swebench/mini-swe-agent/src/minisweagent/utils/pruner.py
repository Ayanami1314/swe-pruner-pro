"""Phase 2 pruner client for mini-swe-agent.

Communicates with the SWE-Pruner Phase 2 server which uses
{history, tool_call, tool_response} format instead of {query, code}.

Ablation routing: if ``PrunerConfig.backend`` is non-empty (e.g.
``llmlingua2``, ``longcodezip``, ...) the request is forwarded to that
ablation backend on the same server. ``context_focus_question`` is taken
from each tool_call's arguments (so the agent must populate it on every
bash command — see the ablation system prompt).
"""
from __future__ import annotations

from typing import Any

import requests
from pydantic import BaseModel, Field

from minisweagent.utils.log import logger


class PrunerConfig(BaseModel):
    url: str
    threshold: float = 0.5
    timeout: float = 120.0
    retries: int = 3
    min_chars: int = 500
    headers: dict[str, str] = Field(default_factory=dict)
    backend: str = ""  # "" → "ours"; non-empty selects an ablation backend

    class Config:
        extra = "allow"


class PruneResponse(BaseModel):
    """Response from Phase 2 pruner server."""
    pruned_code: str
    kept_lines: list[int] = Field(default_factory=list)
    original_lines: int = 0
    kept_line_count: int = 0
    original_chars: int = 0
    pruned_chars: int = 0
    latency_ms: float = 0.0
    error_msg: str | None = None
    backend: str = "ours"


class PrunerClient:
    def __init__(self, config: PrunerConfig):
        self.config = config
        base_headers = {"Content-Type": "application/json"} | config.headers
        self.session = requests.Session()
        self.session.headers.update(base_headers)

    def prune(
        self,
        history: list[dict[str, Any]],
        tool_call: dict[str, Any],
        tool_response: str,
        threshold: float | None = None,
        context_focus_question: str = "",
    ) -> PruneResponse:
        """Call Phase 2 pruner: {history, tool_call, tool_response, threshold,
        pruner_backend, context_focus_question}."""
        if len(tool_response) < self.config.min_chars:
            return PruneResponse(
                pruned_code=tool_response,
                original_lines=len(tool_response.splitlines()),
                kept_line_count=len(tool_response.splitlines()),
                original_chars=len(tool_response),
                pruned_chars=len(tool_response),
            )

        payload: dict[str, Any] = {
            "history": history,
            "tool_call": tool_call,
            "tool_response": tool_response,
            "threshold": threshold if threshold is not None else self.config.threshold,
            "output_format": "filtered_markers",
        }
        if self.config.backend:
            payload["pruner_backend"] = self.config.backend
        if context_focus_question:
            payload["context_focus_question"] = context_focus_question

        response = self.session.post(
            self.config.url,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return PruneResponse(**response.json())
