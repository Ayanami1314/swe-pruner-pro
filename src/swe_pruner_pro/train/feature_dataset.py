"""Packed memmap feature dataset for from-features training.

Single-layer last-layer hidden states only (paper does not use multi-layer
fusion or prefix-HS cross-attention). Augmentations from the research code
(skeleton-fix, neg-ratio, noise injection) are dropped — paper trains on
plain features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class PackedFeatureDataset:
    """Zero-copy memmap dataset for cached hidden states.

    Directory layout (produced by your feature-extraction script):
        hidden_states.bin   float32  [total_tokens, hidden_dim]
        token_labels.bin    int64    [total_tokens]
        token_line_ids.bin  int64    [total_tokens]   (optional)
        index.json          {"hidden_dim": int, "samples": [{"offset","length",...}]}
    """

    def __init__(self, packed_dir: str | Path):
        packed_dir = Path(packed_dir)
        with open(packed_dir / "index.json") as f:
            index = json.load(f)
        self.hidden_dim: int = index["hidden_dim"]
        self.samples: list[dict] = index["samples"]
        total_tokens = self.samples[-1]["offset"] + self.samples[-1]["length"]

        self.hidden = np.memmap(
            packed_dir / "hidden_states.bin",
            dtype=np.float32, mode="r",
            shape=(total_tokens, self.hidden_dim),
        )
        self.labels = np.memmap(
            packed_dir / "token_labels.bin",
            dtype=np.int64, mode="r", shape=(total_tokens,),
        )
        line_ids_path = packed_dir / "token_line_ids.bin"
        self.has_line_ids = line_ids_path.exists()
        if self.has_line_ids:
            self.line_ids = np.memmap(
                line_ids_path, dtype=np.int64, mode="r", shape=(total_tokens,),
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        start, length = s["offset"], s["length"]
        result = {
            "hidden_states": torch.from_numpy(self.hidden[start: start + length].copy()),
            "token_labels": torch.from_numpy(self.labels[start: start + length].copy()),
        }
        if self.has_line_ids:
            result["token_line_ids"] = torch.from_numpy(
                self.line_ids[start: start + length].copy()
            )
        return result


def feature_collate_fn(batch: list[dict]) -> dict:
    """Plain dynamic padding for from-features batches."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    max_len = max(b["hidden_states"].shape[0] for b in batch)
    has_line_ids = "token_line_ids" in batch[0]

    hs_list, label_list, mask_list, line_id_list = [], [], [], []
    for b in batch:
        seq_len = b["hidden_states"].shape[0]
        pad_len = max_len - seq_len
        hs_list.append(F.pad(b["hidden_states"], (0, 0, 0, pad_len)))
        label_list.append(F.pad(b["token_labels"], (0, pad_len), value=-100))
        m = torch.ones(max_len, dtype=torch.long)
        m[seq_len:] = 0
        mask_list.append(m)
        if has_line_ids:
            line_id_list.append(F.pad(b["token_line_ids"], (0, pad_len), value=0))

    result = {
        "hidden_states": torch.stack(hs_list),
        "attention_mask": torch.stack(mask_list),
        "doc_mask": torch.stack(mask_list).bool(),
        "token_labels": torch.stack(label_list),
    }
    if has_line_ids:
        result["token_line_ids"] = torch.stack(line_id_list)
    return result
