"""Per-run stats accumulator used by `run_agent_loop`.

All five benchmarks currently build the same dict shape inline:
`{total_prompt_tokens, total_completion_tokens, total_iterations, prune_count,
prune_events, prune_skipped?}`. `RunStats` centralizes that with a matching
`.to_dict()` so benchmark writers can keep dumping the same JSON schema into
`full_output.jsonl`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_iterations: int = 0
    prune_count: int = 0
    prune_events: list[dict] = field(default_factory=list)
    prune_skipped: list[dict] = field(default_factory=list)
    mimo_parse_fallbacks: int = 0

    def add_usage(self, usage) -> None:
        if not usage:
            return
        self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def to_dict(self) -> dict:
        d = {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_iterations": self.total_iterations,
            "prune_count": self.prune_count,
            "prune_events": self.prune_events,
        }
        if self.prune_skipped:
            d["prune_skipped"] = self.prune_skipped
        if self.mimo_parse_fallbacks:
            d["mimo_parse_fallbacks"] = self.mimo_parse_fallbacks
        return d
