"""SelectiveContext baseline.

The ``selective-context`` package wraps a small LM (default gpt2) plus
spaCy ``en_core_web_sm`` and uses self-information scores to drop
low-information spans. We keep the chunking pattern from the single-turn
eval: split into ~1024-tok pieces, compress each, concat.

Like LLMLingua-2 this is a query-AGNOSTIC compressor — the focus question
is accepted for protocol uniformity but ignored.

Offline assets required (pre-baked into the ablation Docker image):
  * gpt2 weights + tokenizer
  * spacy ``en_core_web_sm`` model

GPU placement: pass ``device`` to the constructor; selective-context picks
up cuda automatically when the underlying transformers model is on GPU.

Concurrency: ``self._tok_lock`` serializes the tokenizer + model body.
``compressor.tokenizer`` is a HF fast tokenizer (Rust, not thread-safe →
"Already borrowed" panic) and ``compressor.__call__`` runs gpt2 forward
that internally tokenizes again. The two are interleaved deeply enough
that we lock the whole prune body. The server-level lock was removed in
2026-05-13 in favor of these per-baseline locks.
"""

from __future__ import annotations

import logging
import threading
import time

from .types import BaselineResult

logger = logging.getLogger(__name__)


class SelectiveContextPruner:
    name = "selective_context"

    def __init__(
        self,
        model_type: str = "gpt2",
        lang: str = "en",
        reduce_ratio: float = 0.5,
        device: str = "cuda:0",
    ):
        import torch
        from selective_context import SelectiveContext

        self.reduce_ratio = reduce_ratio
        self.compressor = SelectiveContext(model_type=model_type, lang=lang)
        try:
            self.compressor.model = self.compressor.model.to(device)
            self.compressor.device = torch.device(device)
        except Exception:
            logger.warning(f"[selective_context] could not move to {device}, "
                           f"using package default")
        self._sc_max_len = getattr(self.compressor.tokenizer, "model_max_length", 1024) or 1024
        self._chunk_max = max(64, self._sc_max_len - 24)
        self._tok_lock = threading.Lock()
        logger.info(f"[selective_context] loaded {model_type} on {device}, "
                    f"reduce_ratio={reduce_ratio}, chunk_max={self._chunk_max}")

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
                return BaselineResult.passthrough(tool_response,
                                                   latency_ms=(time.time() - t0) * 1000)

            with self._tok_lock:
                tokenizer = self.compressor.tokenizer
                input_ids = tokenizer.encode(tool_response)
                if len(input_ids) <= self._chunk_max:
                    pruned, _ = self.compressor(tool_response,
                                                reduce_ratio=self.reduce_ratio)
                else:
                    chunks_out: list[str] = []
                    for i in range(0, len(input_ids), self._chunk_max):
                        chunk_ids = input_ids[i : i + self._chunk_max]
                        chunk_text = tokenizer.decode(chunk_ids,
                                                      skip_special_tokens=True)
                        if not chunk_text.strip():
                            continue
                        try:
                            compressed_chunk, _ = self.compressor(
                                chunk_text, reduce_ratio=self.reduce_ratio
                            )
                            if compressed_chunk.strip():
                                chunks_out.append(compressed_chunk)
                        except Exception as ce:
                            logger.warning(f"[selective_context] chunk error: {ce}; "
                                           f"keeping chunk as-is")
                            chunks_out.append(chunk_text)
                    pruned = "\n\n".join(chunks_out) if chunks_out else tool_response

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
            logger.exception("[selective_context] prune failed")
            return BaselineResult.passthrough(
                tool_response,
                latency_ms=(time.time() - t0) * 1000,
                error_msg=f"selective_context: {type(exc).__name__}: {exc}",
            )
