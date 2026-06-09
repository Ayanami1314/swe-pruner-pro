"""Pruning model + standalone inference head, paper-faithful trimmed version.

Per the paper (Appendix B, Training Details):
  - Frozen agent backbone provides per-token hidden states.
  - A small FFN classifier (LayerNorm → Linear(d,d)-GELU-Dropout →
    Linear(d,d)-GELU-Dropout → Linear(d,1)) produces a per-token keep/prune
    logit. Hidden width is the backbone hidden size ``d`` so the head is
    re-sized per backbone; dropout is 0.4.
  - A learned length-aware embedding (8 log-spaced n_lines buckets) is added
    to hidden_states immediately before the head (`pre_head` position),
    zero-initialised so it starts as a no-op and only learns the per-bucket
    correction term.
  - Line decision is the majority vote of per-token sigmoid > threshold.

This file deliberately keeps only that one path. Other heads explored in the
research codebase (CRF, line transformer, bilinear/MoE prototype, windowed
attn, ScalarMix, FiLM, multi-layer fusion, LoRA/finetune) are dropped.
"""
from __future__ import annotations

import importlib.util
import json
import math
import threading
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# Canonical bucket edges from the paper (8 log-spaced buckets over n_lines).
_SIZE_BUCKET_EDGES: Tuple[int, ...] = (2, 5, 10, 20, 50, 100, 200)


def _make_size_bucket_edges(n_buckets: int) -> Tuple[int, ...]:
    """Return n_buckets-1 log-spaced edges. n_buckets=8 reproduces the paper."""
    if n_buckets == 8:
        return _SIZE_BUCKET_EDGES
    if n_buckets <= 1:
        return ()
    lo = math.log(1.5)
    hi = math.log(400.0)
    step = (hi - lo) / max(n_buckets - 1, 1)
    edges = []
    last = 0
    for i in range(n_buckets - 1):
        v = max(int(round(math.exp(lo + (i + 0.5) * step))), last + 1)
        edges.append(v)
        last = v
    return tuple(edges)


def _bucket_n_lines(n_lines: torch.Tensor, n_buckets: int = 8) -> torch.Tensor:
    """Map per-sample n_lines (long tensor) to bucket index in [0, n_buckets-1]."""
    edges_t = torch.tensor(
        _make_size_bucket_edges(n_buckets),
        device=n_lines.device, dtype=n_lines.dtype,
    )
    if edges_t.numel() == 0:
        return torch.zeros_like(n_lines)
    idx = (n_lines.unsqueeze(-1) > edges_t).sum(dim=-1)
    return idx.clamp(max=n_buckets - 1)


def _build_size_embedding(module: nn.Module, n_buckets: int, fused_dim: int) -> None:
    """Attach a zero-initialised additive size embedding at pre_head position.

    Zero-init means the head reduces to plain FFN at step 0; the per-bucket
    correction term is learned end-to-end. Embedding is fp32 to match head.
    """
    module.size_embedding = nn.Embedding(n_buckets, fused_dim, dtype=torch.float32)
    nn.init.zeros_(module.size_embedding.weight)


def _build_ffn_head(input_dim: int, dropout: float) -> nn.Sequential:
    """Per-token FFN classifier as specified in the paper (Appendix B).

    Two hidden Linear-GELU-Dropout blocks with hidden width matching the
    backbone hidden size ``d`` (so the head is re-sized per backbone), then a
    final Linear projection to a single keep logit:
    ``LayerNorm(d) -> Linear(d,d) -> GELU -> Dropout
       -> Linear(d,d) -> GELU -> Dropout -> Linear(d,1)``.
    """
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, input_dim, dtype=torch.float32),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(input_dim, input_dim, dtype=torch.float32),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(input_dim, 1, dtype=torch.float32),
    )


class PruningModel(nn.Module):
    """Frozen-backbone pruning model for from-features training.

    When ``hidden_states`` are passed to ``forward``, the backbone is skipped
    entirely (training reads cached HS from disk, see
    ``swe_pruner_pro.train.feature_dataset``). When ``input_ids`` are passed
    instead, the backbone runs in no_grad mode.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        hidden_size: Optional[int] = None,
        dropout: float = 0.4,
        use_size_emb: bool = True,
        n_size_buckets: int = 8,
        skip_backbone: bool = False,
    ):
        super().__init__()
        if skip_backbone:
            if hidden_size is None:
                raise ValueError("skip_backbone=True requires explicit hidden_size")
            self.backbone = None
            self.hidden_size = hidden_size
        else:
            if model_name is None:
                raise ValueError("model_name required unless skip_backbone=True")
            attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
            self.backbone = AutoModel.from_pretrained(
                model_name, dtype=torch.bfloat16, attn_implementation=attn_impl,
            )
            self.hidden_size = self.backbone.config.hidden_size
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        fused_dim = self.hidden_size
        self.compression_head_type = "ffn"  # paper uses only FFN
        self.use_size_emb = use_size_emb
        self.n_size_buckets = n_size_buckets
        self.compression_head = _build_ffn_head(fused_dim, dropout)
        if use_size_emb:
            _build_size_embedding(self, n_size_buckets, fused_dim)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        token_line_ids: Optional[torch.Tensor] = None,
        **_unused,
    ) -> dict:
        target_device = next(self.compression_head.parameters()).device
        if hidden_states is not None:
            h = hidden_states.to(device=target_device).float()
        else:
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=False, return_dict=True,
                )
                h = outputs.last_hidden_state.to(device=target_device).float()

        if self.use_size_emb and token_line_ids is not None:
            n_lines = token_line_ids.to(h.device).max(dim=1).values.long()
            bucket = _bucket_n_lines(n_lines, self.n_size_buckets)
            sig = self.size_embedding(bucket)  # [B, D]
            h = h + sig.unsqueeze(1)

        token_logits = self.compression_head(h).squeeze(-1)
        return {"token_logits": token_logits}


class PruningHead(nn.Module):
    """Inference-only head loaded from ``model_config.json + best_model.pt``.

    Loads ~18M params (FFN + optional 8-bucket size embedding); no backbone.
    Hidden states are externally supplied (e.g. from SGLang's
    ``return_hidden_states``). Thread-safe via an internal lock.
    """

    def __init__(self, checkpoint_dir: str, hidden_size: int = 2048, device: str = "cuda:0"):
        super().__init__()
        self._lock = threading.Lock()
        config = json.loads(Path(checkpoint_dir).joinpath("model_config.json").read_text())
        self._device = torch.device(device)

        fused_dim = hidden_size
        dropout = config.get("dropout", 0.4)

        self.compression_head = _build_ffn_head(fused_dim, dropout)

        self.use_size_emb = config.get("use_size_emb", config.get("use_size_embedding", False))
        self.n_size_buckets = config.get("n_size_buckets", 8)
        if self.use_size_emb:
            _build_size_embedding(self, self.n_size_buckets, fused_dim)

        weights_path = Path(checkpoint_dir) / "best_model.pt"
        sd = torch.load(weights_path, map_location="cpu")
        head_prefixes = ("compression_head", "size_embedding")
        head_sd = {k: v for k, v in sd.items() if k.startswith(head_prefixes)}
        self.load_state_dict(head_sd, strict=False)
        self.to(self._device)
        self.eval()

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, line_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run head on externally supplied HS. Returns CPU logits [seq_len]."""
        with self._lock:
            h = hidden_states.to(self._device).float()
            if h.dim() == 2:
                h = h.unsqueeze(0)
            if self.use_size_emb and line_ids is not None:
                lid = line_ids.to(self._device)
                n_lines = lid.max().long().unsqueeze(0)
                bucket = _bucket_n_lines(n_lines, self.n_size_buckets)
                sig = self.size_embedding(bucket)
                h = h + sig.unsqueeze(1)
            logits = self.compression_head(h).squeeze(-1)
            result = logits.squeeze(0).cpu()
            del h, logits
            return result

    def predict_kept_lines(
        self,
        hidden_states: torch.Tensor,
        tool_response: str,
        tokenizer,
        threshold: float = 0.5,
    ) -> tuple:
        """High-level helper: HS + raw tool_response → (pruned_text, kept_lines, stats).

        Line decision uses majority vote of per-token sigmoid > threshold,
        per the paper. Single-line gaps between two kept lines are filled.
        """
        from ..train.metrics import build_token_line_ids

        line_ids = build_token_line_ids(tool_response, tokenizer)
        hs_len = hidden_states.shape[-2] if hidden_states.dim() == 3 else hidden_states.shape[0]
        if len(line_ids) > hs_len:
            line_ids = line_ids[:hs_len]
        elif len(line_ids) < hs_len:
            line_ids = torch.cat(
                [line_ids, torch.zeros(hs_len - len(line_ids), dtype=torch.long)]
            )

        logits = self.forward(hidden_states, line_ids=line_ids)
        line_ids = line_ids[: logits.size(0)]
        pred = torch.sigmoid(logits)

        line_scores: dict[int, list[float]] = defaultdict(list)
        for p, lid in zip(pred.tolist(), line_ids.tolist()):
            if lid > 0:
                line_scores[lid].append(p)
        kept = [
            lid for lid in sorted(line_scores)
            if sum(1 for s in line_scores[lid] if s >= threshold) > len(line_scores[lid]) / 2
        ]
        # Fill single-line gaps to reduce orphan fragments.
        extra = [kept[i] - 1 for i in range(1, len(kept)) if kept[i] - kept[i - 1] == 2]
        kept = sorted(set(kept + extra))

        lines = tool_response.splitlines()
        result, kept_set = [], set(kept)
        n_width = len(str(len(lines)))
        for ln in range(1, len(lines) + 1):
            if ln in kept_set:
                result.append(f"{ln:>{n_width}} | {lines[ln - 1]}")
        pruned = "\n".join(result)
        stats = {
            "original_lines": len(lines),
            "kept_lines": len(kept),
            "original_chars": len(tool_response),
            "pruned_chars": len(pruned),
        }
        return pruned, kept, stats
