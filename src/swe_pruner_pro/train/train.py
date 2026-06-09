"""From-features training CLI for the pruning head.

Paper recipe (default flags below): single-layer last-layer HS cached on disk
→ FFN head with optional 8-bucket length-aware additive embedding → per-sample
balanced focal loss (γ=2) → AdamW (lr 3e-5, weight_decay 0) with linear
warmup (5%) and cosine decay to 1.5e-5 → 10 epochs, bs=4 per GPU,
grad-clip 1.0. DDP via torchrun for multi-GPU; plain training for world=1.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import typer
from rich.console import Console
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ..model.head import PruningModel
from ..model.losses import DiceLoss, FocalLoss, PerSampleBalancedFocal, TverskyLoss
from .evaluation import evaluate
from .feature_dataset import PackedFeatureDataset, feature_collate_fn
from .metrics import _aggregate_tokens_to_lines, _binary_metrics

console = Console()


def _setup_ddp() -> tuple[int, int, int]:
    """Initialize torch.distributed if launched via torchrun, else return (0,1,0)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def _is_main(rank: int) -> bool:
    return rank == 0


def _compute_loss(loss_name, logits, targets, focal_alpha, focal_gamma):
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    if loss_name == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)(logits, targets)
    if loss_name == "dice":
        return DiceLoss()(logits, targets)
    if loss_name == "tversky":
        return TverskyLoss()(logits, targets)
    if loss_name == "psbf":
        return PerSampleBalancedFocal(gamma=focal_gamma)(logits, targets)
    raise ValueError(f"unknown loss {loss_name!r}")


def _compute_pos_ratio(dataset, n_probe: int = 5000) -> float:
    """Auto focal-alpha probe: read raw labels memmap to avoid pulling huge HS slices."""
    if hasattr(dataset, "samples") and hasattr(dataset, "labels"):
        n = min(len(dataset), n_probe)
        total_pos = total_valid = 0
        for i in range(n):
            s = dataset.samples[i]
            arr = dataset.labels[s["offset"]: s["offset"] + s["length"]]
            valid = arr != -100
            total_pos += int((arr[valid] == 1).sum())
            total_valid += int(valid.sum())
        return total_pos / max(total_valid, 1)
    total_pos = total_valid = 0
    for i in range(min(len(dataset), n_probe)):
        s = dataset[i]
        if s is None:
            continue
        labels = s["token_labels"]
        valid = labels != -100
        total_pos += (labels[valid] == 1).sum().item()
        total_valid += valid.sum().item()
    return total_pos / max(total_valid, 1)


def main(
    features_dir: Path = typer.Option(..., "--features-dir", help="Packed feature directory"),
    eval_data: str = typer.Option("", "--eval-data", help="Optional source jsonl for downstream judge eval (unused for training)"),
    log_dir: Path = typer.Option(..., "--log-dir"),
    epochs: int = typer.Option(10, "--epochs"),
    lr: float = typer.Option(3e-5, "--lr"),
    min_lr: float = typer.Option(1.5e-5, "--min-lr"),
    warmup_ratio: float = typer.Option(0.05, "--warmup-ratio"),
    batch_size: int = typer.Option(4, "--batch-size"),
    dropout: float = typer.Option(0.4, "--dropout"),
    use_size_emb: bool = typer.Option(True, "--use-size-emb/--no-use-size-emb"),
    n_size_buckets: int = typer.Option(8, "--n-size-buckets"),
    loss: str = typer.Option("psbf", "--loss", help="bce|focal|dice|tversky|psbf"),
    focal_alpha: float = typer.Option(0.5, "--focal-alpha"),
    auto_focal_alpha: bool = typer.Option(True, "--auto-focal-alpha/--no-auto-focal-alpha"),
    focal_gamma: float = typer.Option(2.0, "--focal-gamma"),
    weight_decay: float = typer.Option(0.0, "--weight-decay"),
    grad_clip: float = typer.Option(1.0, "--grad-clip"),
    threshold: float = typer.Option(0.5, "--threshold"),
    train_split: float = typer.Option(0.9, "--train-split"),
    seed: int = typer.Option(42, "--seed"),
    num_workers: int = typer.Option(4, "--num-workers"),
):
    rank, world, local_rank = _setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    import numpy as np
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset = PackedFeatureDataset(features_dir)
    if _is_main(rank):
        console.print(f"From-features: {len(dataset)} samples, dim={dataset.hidden_dim}")

    effective_alpha = focal_alpha
    if auto_focal_alpha and loss in ("focal",):
        pos_ratio = _compute_pos_ratio(dataset)
        effective_alpha = max(1.0 - pos_ratio, 1e-3)
        if _is_main(rank):
            console.print(f"Auto focal_alpha: {effective_alpha:.4f} (pos_ratio={pos_ratio:.4f})")

    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    if _is_main(rank):
        console.print(f"Train: {n_train}, Val: {n_val}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True) if world > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=feature_collate_fn,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch_size), shuffle=False,
        sampler=val_sampler, collate_fn=feature_collate_fn,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )

    model = PruningModel(
        hidden_size=dataset.hidden_dim,
        dropout=dropout,
        use_size_emb=use_size_emb,
        n_size_buckets=n_size_buckets,
        skip_backbone=True,
    ).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model_unwrapped = model.module if isinstance(model, DDP) else model

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if _is_main(rank):
        console.print(f"Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    min_ratio = min_lr / lr

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1 = 0.0
    global_step = 0
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        if _is_main(rank):
            console.print(f"\n=== Epoch {epoch}/{epochs} ===")
        pbar = tqdm(train_loader, desc=f"ep{epoch}") if _is_main(rank) else train_loader
        for batch in pbar:
            if not batch:
                continue
            hidden_states = batch["hidden_states"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_labels = batch["token_labels"].to(device)
            token_line_ids = batch.get("token_line_ids")
            if token_line_ids is not None:
                token_line_ids = token_line_ids.to(device)
            doc_mask = batch["doc_mask"].to(device).bool()

            out = model(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                token_line_ids=token_line_ids,
            )
            logits = out["token_logits"].float()
            valid = doc_mask & attention_mask.bool() & (token_labels != -100)
            if valid.sum() == 0:
                continue
            # Per-sample loss aggregation: gives long samples and short
            # samples equal weight on the gradient (matches the paper's
            # description of per-sample balancing).
            sample_losses = []
            for i in range(logits.size(0)):
                si = valid[i]
                if si.sum() == 0:
                    continue
                lv = logits[i][si]
                tv = token_labels[i][si].float()
                sample_losses.append(_compute_loss(loss, lv, tv, effective_alpha, focal_gamma))
            if not sample_losses:
                continue
            batch_loss = torch.stack(sample_losses).mean()
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            batch_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if _is_main(rank) and isinstance(pbar, tqdm) and global_step % 20 == 0:
                pbar.set_postfix(
                    loss=f"{batch_loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        if _is_main(rank):
            console.print("Evaluating...")
        # Evaluate only on rank 0 (head is replicated and small).
        if _is_main(rank):
            metrics = evaluate(
                model_unwrapped, val_loader, device=device,
                threshold=threshold, loss_type=loss,
                focal_alpha=effective_alpha, focal_gamma=focal_gamma,
            )
            console.print(
                f"  loss={metrics['loss']:.4f}  token_F1={metrics['token_f1']:.4f}  "
                f"line_F1={metrics['line_f1']:.4f}  P={metrics['line_precision']:.4f}  "
                f"R={metrics['line_recall']:.4f}"
            )
            if metrics["sweep"]:
                console.print("  [threshold sweep]")
                for th, f1, p, r in metrics["sweep"]:
                    mark = " <-- selected" if abs(th - threshold) < 1e-6 else ""
                    console.print(f"    th={th:.1f}: F1={f1:.4f} P={p:.4f} R={r:.4f}{mark}")

            if metrics["line_f1"] > best_f1:
                best_f1 = metrics["line_f1"]
                torch.save(model_unwrapped.state_dict(), log_dir / "best_model.pt")
                config = {
                    "format": "swe-pruner-pro",
                    "compression_head_type": "ffn",
                    "dropout": dropout,
                    "hidden_dim": dataset.hidden_dim,
                    "use_size_emb": use_size_emb,
                    "n_size_buckets": n_size_buckets,
                    "loss": loss,
                    "focal_alpha": effective_alpha,
                    "focal_gamma": focal_gamma,
                    "epochs": epochs,
                    "lr": lr,
                    "min_lr": min_lr,
                    "best_line_f1": best_f1,
                }
                (log_dir / "model_config.json").write_text(json.dumps(config, indent=2))
                console.print(f"  Saved best model (line_F1={best_f1:.4f}) to {log_dir}")

        if world > 1 and dist.is_initialized():
            dist.barrier()

    if _is_main(rank):
        console.print("Training complete.")
    if world > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    typer.run(main)
