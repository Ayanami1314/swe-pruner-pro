"""Per-sample binary classification losses used in the paper's ablation.

Paper main results use ``PerSampleBalancedFocal`` (γ=2). Other losses are
provided for the loss-ablation table: BCE, Focal, Dice, Tversky.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Standard focal loss. With auto-alpha=1-pos_ratio computed by the trainer."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * torch.pow(1 - p_t, self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Soft-Dice region-overlap loss. Naturally handles class imbalance."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class TverskyLoss(nn.Module):
    """Tversky index loss — asymmetric Dice. α>β emphasises recall."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fn = ((1.0 - probs) * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        t = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - t


class PerSampleBalancedFocal(nn.Module):
    """Paper's main loss. Within each sample: focal-BCE averaged per class
    (n_pos and n_neg averaged separately) then mixed 50/50.

    Solves per-sample length imbalance that pool-level alpha cannot — a
    sample with 5 keeps and 50 prunes weights both halves equally regardless
    of count. (1 − p_t)^γ saturates the gradient on already-correct tokens
    so the head can't collapse to all-keep / all-prune.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        y = targets.float()
        p_t = probs * y + (1.0 - probs) * (1.0 - y)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        focal = focal_weight * bce

        sum_keep = y.sum()
        sum_prune = (1.0 - y).sum()
        has_keep = sum_keep.item() > 0
        has_prune = sum_prune.item() > 0
        keep_loss = (focal * y).sum() / sum_keep.clamp(min=1.0)
        prune_loss = (focal * (1.0 - y)).sum() / sum_prune.clamp(min=1.0)
        if has_keep and has_prune:
            return 0.5 * keep_loss + 0.5 * prune_loss
        if has_keep:
            return keep_loss
        if has_prune:
            return prune_loss
        return logits.new_zeros(())
