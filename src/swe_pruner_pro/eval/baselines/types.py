"""Common type for baseline pruners — shape matches pruner_server.PruneResponse."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaselineResult:
    pruned_code: str
    kept_lines: list[int] = field(default_factory=list)
    original_lines: int = 0
    kept_line_count: int = 0
    original_chars: int = 0
    pruned_chars: int = 0
    latency_ms: float = 0.0
    error_msg: str | None = None

    @classmethod
    def passthrough(cls, tool_response: str, latency_ms: float = 0.0,
                    error_msg: str | None = None) -> "BaselineResult":
        n_lines = len(tool_response.splitlines())
        return cls(
            pruned_code=tool_response,
            kept_lines=list(range(1, n_lines + 1)),
            original_lines=n_lines,
            kept_line_count=n_lines,
            original_chars=len(tool_response),
            pruned_chars=len(tool_response),
            latency_ms=latency_ms,
            error_msg=error_msg,
        )
