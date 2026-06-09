"""Token-level label construction and evaluation metrics for line-level pruning."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from pydantic import BaseModel


def _encode_with_offsets(tokenizer, text: str):
    """Tokenize and return (ids, offsets). Falls back to a per-token decode
    walk when the tokenizer doesn't support return_offsets_mapping=True."""
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]
    except (NotImplementedError, ValueError, TypeError):
        ids = tokenizer.encode(text, add_special_tokens=False)
        offsets: List[Tuple[int, int]] = []
        pos = 0
        n = len(text)
        for tid in ids:
            piece = tokenizer.decode([tid])
            if not piece:
                offsets.append((pos, pos))
                continue
            found = text.find(piece, pos)
            if found < 0:
                offsets.append((pos, min(pos + 1, n)))
                pos = min(pos + 1, n)
            else:
                offsets.append((found, found + len(piece)))
                pos = found + len(piece)
        return ids, offsets


def kept_frags_to_label(kept_frags: List[int], code: str, tokenizer) -> torch.Tensor:
    """Convert 1-based kept line numbers to per-token binary labels."""
    lines = code.splitlines(keepends=True)
    kept_set = set(kept_frags)
    keep_char_spans = []
    char_cnt = 0
    for idx, line in enumerate(lines, 1):
        if idx in kept_set:
            keep_char_spans.append((char_cnt, char_cnt + len(line)))
        char_cnt += len(line)
    enc_ids, offsets = _encode_with_offsets(tokenizer, code)
    mask = torch.zeros(len(offsets), dtype=torch.float32)
    for i, (tok_s, tok_e) in enumerate(offsets):
        for ks, ke in keep_char_spans:
            if tok_s < ke and tok_e > ks:
                mask[i] = 1.0
                break
    return mask


def build_token_line_ids(code: str, tokenizer) -> torch.Tensor:
    """Per-token 1-based line index. 0 = unmapped / padding."""
    lines = code.splitlines(keepends=True)
    char_to_line: List[int] = []
    for idx, line in enumerate(lines, 1):
        char_to_line.extend([idx] * len(line))
    enc_ids, offsets = _encode_with_offsets(tokenizer, code)
    line_ids = torch.zeros(len(offsets), dtype=torch.long)
    for i, (tok_s, _) in enumerate(offsets):
        if tok_s < len(char_to_line):
            line_ids[i] = char_to_line[tok_s]
    return line_ids


class LineMetrics(BaseModel):
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


def _aggregate_tokens_to_lines(
    preds: torch.Tensor,
    labels: torch.Tensor,
    line_ids: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Majority-vote token preds and labels into line-level decisions."""
    unique_lines = line_ids.unique()
    unique_lines = unique_lines[unique_lines > 0]
    n_lines = len(unique_lines)
    line_true = torch.zeros(n_lines, dtype=torch.long)
    line_pred = torch.zeros(n_lines, dtype=torch.long)
    for i, lid in enumerate(unique_lines):
        mask = line_ids == lid
        n_tok = mask.sum().item()
        line_true[i] = 1 if labels[mask].sum().item() > n_tok / 2 else 0
        line_pred[i] = 1 if (preds[mask] >= threshold).sum().item() > n_tok / 2 else 0
    return line_pred, line_true, n_lines


def _binary_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return prec, rec, f1


def line_f1_from_tokens(
    probs: torch.Tensor,
    labels: torch.Tensor,
    line_ids: torch.Tensor,
    threshold: float = 0.5,
) -> Optional[LineMetrics]:
    """Per-sample line-level F1 from per-token probs/labels (paper's metric)."""
    if line_ids is None or len(line_ids) != len(labels) or len(labels) == 0:
        return None
    lp, lt, n_lines = _aggregate_tokens_to_lines(probs, labels, line_ids, threshold)
    if n_lines == 0:
        return None
    tp = ((lp == 1) & (lt == 1)).sum().item()
    fp = ((lp == 1) & (lt == 0)).sum().item()
    fn = ((lp == 0) & (lt == 1)).sum().item()
    p, r, f1 = _binary_metrics(tp, fp, fn)
    return LineMetrics(f1=f1, precision=p, recall=r)
