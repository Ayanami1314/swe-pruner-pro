"""Ablation baselines for the SWE-Pruner server.

Each baseline implements the ``BaselinePruner`` protocol below: given a
tool_response (string) and a focus query (string), return a pruned string
plus stats matching the ``PruneResponse`` schema in ``pruner_server.py``.

Server-side routing lives in ``pruner_server.py``; this package only
defines the backends and a registry that the server populates at startup
based on the ``EXTRA_PRUNERS`` env var (or ``--extra-pruners`` flag).

Per CLAUDE.md guidance:
  * NO fallback to "ours" or another baseline on failure — return the raw
    tool_response with an ``error_msg`` so the caller can see what happened.
  * NO degradation of swe_pruner to reranker-only — if the external service
    is down, baseline returns passthrough with error_msg.
  * Empty/missing ``context_focus_question`` is handled in the server before
    a backend is dispatched (server returns passthrough). Backends here may
    assume ``query`` is non-empty.
"""

from __future__ import annotations

from typing import Protocol

from .types import BaselineResult


class BaselinePruner(Protocol):
    """Stateless wrapper around one ablation pruning backend."""

    name: str

    def prune(
        self,
        *,
        history: list[dict],
        tool_call: dict,
        tool_response: str,
        threshold: float,
        query: str,
    ) -> BaselineResult:
        """Return pruned text plus stats. Must not raise on bad input —
        catch internally and return passthrough + error_msg.
        """
        ...


__all__ = ["BaselinePruner", "BaselineResult"]
