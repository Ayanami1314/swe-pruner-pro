"""Validation loop with line-level micro-F1 and a 0.1..0.9 threshold sweep."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from rich.console import Console
from tqdm import tqdm

from ..model.losses import DiceLoss, FocalLoss, PerSampleBalancedFocal, TverskyLoss
from .metrics import _aggregate_tokens_to_lines, _binary_metrics

console = Console()


def _compute_eval_loss(loss_type, logits, targets, focal_alpha, focal_gamma):
    """Mirror the training loss for validation reporting."""
    if loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)(logits, targets).item()
    if loss_type == "dice":
        return DiceLoss()(logits, targets).item()
    if loss_type == "tversky":
        return TverskyLoss()(logits, targets).item()
    if loss_type == "psbf":
        return PerSampleBalancedFocal(gamma=focal_gamma)(logits, targets).item()
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean").item()


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    threshold: float = 0.5,
    loss_type: str = "psbf",
    focal_alpha: float = 0.5,
    focal_gamma: float = 2.0,
    sweep: bool = True,
) -> dict:
    """Compute val loss + token & line micro-F1; optionally sweep thresholds.

    Returns dict with keys: loss, token_f1, line_f1, line_precision,
    line_recall, sweep (list of (th, f1, p, r)).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    tok_tp = tok_fp = tok_fn = 0
    line_tp = line_fp = line_fn = 0

    sweep_probs: list[float] = []
    sweep_labels: list[int] = []
    sweep_line_records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for batch in tqdm(dataloader, desc="eval", leave=False):
        if not batch:
            continue
        hidden_states = batch["hidden_states"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_labels = batch["token_labels"].to(device)
        token_line_ids = batch.get("token_line_ids")
        if token_line_ids is not None:
            token_line_ids = token_line_ids.to(device)
        doc_mask = batch["doc_mask"].to(device).bool()

        outputs = model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            token_line_ids=token_line_ids,
        )
        logits = outputs["token_logits"].float()
        valid = doc_mask & attention_mask.bool() & (token_labels != -100)
        if valid.sum() == 0:
            continue
        lv = logits[valid]
        tv = token_labels[valid].float()
        total_loss += _compute_eval_loss(loss_type, lv, tv, focal_alpha, focal_gamma)
        n_batches += 1

        probs = torch.sigmoid(lv)
        pred = (probs >= threshold).long()
        tlab = tv.long()
        tok_tp += ((pred == 1) & (tlab == 1)).sum().item()
        tok_fp += ((pred == 1) & (tlab == 0)).sum().item()
        tok_fn += ((pred == 0) & (tlab == 1)).sum().item()
        sweep_probs.extend(probs.cpu().tolist())
        sweep_labels.extend(tlab.cpu().tolist())

        if token_line_ids is not None:
            for i in range(logits.size(0)):
                si = valid[i]
                if si.sum() == 0:
                    continue
                p_i = torch.sigmoid(logits[i][si])
                l_i = token_labels[i][si].float()
                lid_i = token_line_ids[i][si]
                lp, lt, n_lines = _aggregate_tokens_to_lines(p_i, l_i, lid_i, threshold)
                if n_lines == 0:
                    continue
                line_tp += ((lp == 1) & (lt == 1)).sum().item()
                line_fp += ((lp == 1) & (lt == 0)).sum().item()
                line_fn += ((lp == 0) & (lt == 1)).sum().item()
                if sweep:
                    sweep_line_records.append((p_i.cpu(), l_i.cpu(), lid_i.cpu()))

    _, _, tok_f1 = _binary_metrics(tok_tp, tok_fp, tok_fn)
    line_p, line_r, line_f1 = _binary_metrics(line_tp, line_fp, line_fn)

    sweep_results = []
    if sweep and sweep_line_records:
        for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            stp = sfp = sfn = 0
            for p_i, l_i, lid_i in sweep_line_records:
                lp, lt, n_lines = _aggregate_tokens_to_lines(p_i, l_i, lid_i, th)
                if n_lines == 0:
                    continue
                stp += ((lp == 1) & (lt == 1)).sum().item()
                sfp += ((lp == 1) & (lt == 0)).sum().item()
                sfn += ((lp == 0) & (lt == 1)).sum().item()
            p_, r_, f_ = _binary_metrics(stp, sfp, sfn)
            sweep_results.append((th, f_, p_, r_))

    return {
        "loss": total_loss / max(n_batches, 1),
        "token_f1": tok_f1,
        "line_f1": line_f1,
        "line_precision": line_p,
        "line_recall": line_r,
        "sweep": sweep_results,
    }
