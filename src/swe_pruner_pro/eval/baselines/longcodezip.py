"""LongCodeZip baseline (coarse-only / rank_only=True).

Wraps a ``CodeCompressor`` (perplexity-based function-level ranker) and runs
only the function-level coarse stage: split into function chunks, rank by
query relevance, keep the top-N within budget. ``rank_only=True`` skips
fine-grained per-token compression.

This baseline depends on a vendored ``code_compressor`` module exposing a
``CodeCompressor`` class — set the env var ``SWE_PRUNER_BASELINE_LIB_PATH``
to a directory containing it (the LongCodeZip reference implementation),
or place a ``code_compressor.py`` somewhere on ``sys.path``.

Default ranking model: ``Qwen/Qwen2.5-Coder-1.5B-Instruct``. Override via
``LONGCODEZIP_MODEL_PATH`` env var.

GPU memory note: CodeCompressor caches CUDA tensors keyed by raw text,
which never hit across pruner requests (every tool_response is unique).
We clear all caches + ``empty_cache()`` after every prune() call to keep
memory bounded.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

import torch

from .types import BaselineResult

logger = logging.getLogger(__name__)


def _ensure_baseline_lib_path():
    p = os.environ.get("SWE_PRUNER_BASELINE_LIB_PATH")
    if p and p not in sys.path:
        sys.path.insert(0, p)


class LongCodeZipPruner:
    name = "longcodezip"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda:0",
        rate: float = 0.5,
        language: str = "python",
    ):
        _ensure_baseline_lib_path()
        from code_compressor import CodeCompressor  # type: ignore

        if model_name is None:
            model_name = os.environ.get(
                "LONGCODEZIP_MODEL_PATH",
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            )
        self.rate = rate
        self.language = language
        self.device = device
        self.compressor = CodeCompressor(model_name=model_name, device_map=device)
        # HF fast tokenizers (Rust ``tokenizers``) panic on concurrent use.
        # CodeCompressor.compress_code_file uses a Qwen2.5-Coder fast tokenizer
        # interleaved with the ranking LLM forward, so we serialize the call.
        self._tok_lock = threading.Lock()
        # The built-in caches hold CUDA tensors and never hit across requests.
        # Set ceiling small so internal eviction runs; we also wipe + empty_cache()
        # after every prune() call below.
        self.compressor.max_cache_size = 1
        logger.info(f"[longcodezip] loaded {model_name} on {device}, "
                    f"rate={rate} (coarse / rank_only=True, caches disabled)")

    def _drain_caches(self) -> None:
        """Drop all cached tensors and release CUDA memory."""
        for k in ("token_length", "encodings", "perplexity",
                  "conditional_ppl", "context_rankings"):
            d = self.compressor.cache.get(k)
            if d is not None:
                d.clear()
        if self.device.startswith("cuda"):
            try:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            except Exception:
                logger.debug("[longcodezip] empty_cache failed", exc_info=True)

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
            if not tool_response.strip():
                return BaselineResult.passthrough(
                    tool_response, latency_ms=(time.time() - t0) * 1000
                )

            with self._tok_lock:
                result = self.compressor.compress_code_file(
                    code=tool_response,
                    query=query,
                    instruction="",
                    rate=self.rate,
                    language=self.language,
                    rank_only=True,
                )
            pruned = result.get("compressed_code", tool_response) or tool_response

            n_lines = len(tool_response.splitlines())
            return BaselineResult(
                pruned_code=pruned,
                kept_lines=[],
                original_lines=n_lines,
                kept_line_count=len(pruned.splitlines()),
                original_chars=len(tool_response),
                pruned_chars=len(pruned),
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:
            logger.exception("[longcodezip] prune failed")
            return BaselineResult.passthrough(
                tool_response,
                latency_ms=(time.time() - t0) * 1000,
                error_msg=f"longcodezip: {type(exc).__name__}: {exc}",
            )
        finally:
            self._drain_caches()
