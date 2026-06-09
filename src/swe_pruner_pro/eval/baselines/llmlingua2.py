"""LLMLingua-2 token-level prompt compressor.

Wraps ``llmlingua.PromptCompressor(use_llmlingua2=True)`` with the same
chunking pattern used in ``downstream_eval/single_turn/.../eval.py``:
  - chunk the tool_response into ~512-tok pieces with the cl100k_base
    encoder so we stay under the model's input limit
  - compress each chunk with the configured rate
  - stitch the chunks back together

LLMLingua-2 is a sequence-tagging compressor (not perplexity-based), so it
does NOT consume the focus query — it produces a query-agnostic compression.
We accept the query in the signature for protocol uniformity but ignore it.

Tokenizer requirement: ``cl100k_base`` (tiktoken). Pre-cache it via
``TIKTOKEN_CACHE_DIR`` in the Dockerfile to avoid runtime download on CML.
"""

from __future__ import annotations

import logging
import threading
import time

from .types import BaselineResult

logger = logging.getLogger(__name__)


class LLMLingua2Pruner:
    name = "llmlingua2"

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device: str = "cuda:0",
        rate: float = 0.33,
        chunk_max_tokens: int = 512,
        force_tokens: list[str] | None = None,
    ):
        from llmlingua import PromptCompressor
        import tiktoken

        self.rate = rate
        self.chunk_max_tokens = chunk_max_tokens
        self.force_tokens = force_tokens or []
        self._tiktoken = tiktoken.get_encoding("cl100k_base")
        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device,
        )
        # See pruner_server.py: HF fast tokenizers (Rust `tokenizers`) are not
        # thread-safe. PromptCompressor.compress_prompt internally uses an
        # XLM-RoBERTa fast tokenizer interleaved with model forward, so we
        # serialize the whole call. Tiktoken (used for the chunking step
        # above) IS thread-safe, but locking unconditionally is simpler.
        self._tok_lock = threading.Lock()
        logger.info(f"[llmlingua2] loaded {model_name} on {device}, rate={rate}")

    def _chunk_by_tokens(self, text: str) -> list[str]:
        if not text.strip():
            return []
        tokens = self._tiktoken.encode(text)
        if len(tokens) <= self.chunk_max_tokens:
            return [text]
        out: list[str] = []
        for i in range(0, len(tokens), self.chunk_max_tokens):
            chunk_tokens = tokens[i : i + self.chunk_max_tokens]
            out.append(self._tiktoken.decode(chunk_tokens))
        return out

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
            chunks = self._chunk_by_tokens(tool_response)
            if not chunks:
                return BaselineResult.passthrough(tool_response,
                                                   latency_ms=(time.time() - t0) * 1000)

            compressed: list[str] = []
            for chunk in chunks:
                kwargs: dict = {"context": [chunk], "rate": self.rate}
                if self.force_tokens:
                    kwargs["force_tokens"] = self.force_tokens
                with self._tok_lock:
                    result = self.compressor.compress_prompt(**kwargs)
                if isinstance(result, dict):
                    if "compressed_prompt" in result:
                        compressed.append(result["compressed_prompt"])
                    elif "compressed_prompt_list" in result:
                        compressed.append("\n\n".join(result["compressed_prompt_list"]))

            pruned = "\n\n".join(compressed) if compressed else tool_response
            n_lines = len(tool_response.splitlines())
            n_pruned = len(pruned.splitlines())
            return BaselineResult(
                pruned_code=pruned,
                kept_lines=[],
                original_lines=n_lines,
                kept_line_count=n_pruned,
                original_chars=len(tool_response),
                pruned_chars=len(pruned),
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:
            logger.exception("[llmlingua2] prune failed")
            return BaselineResult.passthrough(
                tool_response,
                latency_ms=(time.time() - t0) * 1000,
                error_msg=f"llmlingua2: {type(exc).__name__}: {exc}",
            )
