"""Reranker-only "RAG" baseline using bge-reranker-v2-m3.

Per paper spec: rerank-only (no embedder), 40-line sliding window, top_k=3.
Joins kept chunks back in original order with ``(filtered N lines: X-Y)``
markers between non-adjacent kept chunks, mirroring the main pruner's
output style.

Depends on a vendored ``reranker`` module exposing ``BGEV2M3Reranker``
(the bge-reranker-v2-m3 wrapper). Set ``SWE_PRUNER_BASELINE_LIB_PATH`` to
its directory, or place ``reranker.py`` on ``sys.path``.

Focus question is the rerank query; this baseline REQUIRES it. The agent
runner / pruner-server passthroughs when the query is empty.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

from .types import BaselineResult

logger = logging.getLogger(__name__)


def _ensure_baseline_lib_path():
    p = os.environ.get("SWE_PRUNER_BASELINE_LIB_PATH")
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _sliding_window_lines(code: str, window: int, overlap: int) -> list[tuple[int, int, str]]:
    """Yield (start_line, end_line_exclusive, chunk_text). 1-indexed lines."""
    lines = code.splitlines()
    if not lines:
        return []
    stride = max(1, window - overlap)
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(lines):
        end = min(start + window, len(lines))
        chunk_text = "\n".join(lines[start:end])
        chunks.append((start + 1, end + 1, chunk_text))
        if end == len(lines):
            break
        start += stride
    return chunks


class RerankPruner:
    name = "rerank"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda:0",
        window_size: int = 40,
        overlap: int = 10,
        top_k: int = 3,
    ):
        _ensure_baseline_lib_path()
        from reranker import BGEV2M3Reranker  # type: ignore

        self.window_size = window_size
        self.overlap = overlap
        self.top_k = top_k
        self.reranker = BGEV2M3Reranker(model_name=model_name, device=device)
        # HF fast tokenizers (Rust ``tokenizers``) panic on concurrent use.
        # BGEV2M3Reranker.rerank tokenizes + scores in a single call; serialize.
        self._tok_lock = threading.Lock()
        logger.info(
            f"[rerank] loaded {model_name} on {device}, "
            f"window={window_size}, overlap={overlap}, top_k={top_k}"
        )

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
            chunks = _sliding_window_lines(tool_response, self.window_size, self.overlap)
            if not chunks:
                return BaselineResult.passthrough(
                    tool_response, latency_ms=(time.time() - t0) * 1000
                )

            chunk_texts = [c[2] for c in chunks]
            with self._tok_lock:
                scored = self.reranker.rerank(query, chunk_texts, batch_size=16)
            score_map: dict[str, float] = {}
            for score, doc in scored:
                if doc not in score_map or score > score_map[doc]:
                    score_map[doc] = score

            indexed = [(score_map.get(text, float("-inf")), idx, start, end, text)
                       for idx, (start, end, text) in enumerate(chunks)]
            indexed.sort(key=lambda r: r[0], reverse=True)
            keep_idx = sorted(r[1] for r in indexed[: self.top_k])
            if not keep_idx:
                return BaselineResult.passthrough(
                    tool_response, latency_ms=(time.time() - t0) * 1000
                )

            kept_lines: set[int] = set()
            for idx in keep_idx:
                start, end, _ = chunks[idx]
                for ln in range(start, end):
                    kept_lines.add(ln)

            src_lines = tool_response.splitlines()
            out_parts: list[str] = []
            prev_ln = 0
            for ln in range(1, len(src_lines) + 1):
                if ln in kept_lines:
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
                kept_lines=sorted(kept_lines),
                original_lines=len(src_lines),
                kept_line_count=len(kept_lines),
                original_chars=len(tool_response),
                pruned_chars=len(pruned),
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:
            logger.exception("[rerank] prune failed")
            return BaselineResult.passthrough(
                tool_response,
                latency_ms=(time.time() - t0) * 1000,
                error_msg=f"rerank: {type(exc).__name__}: {exc}",
            )
