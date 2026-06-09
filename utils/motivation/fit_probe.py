"""Fit the motivation-figure linear probe and cache its outputs so plot
iteration doesn't re-train.

Trains a 1-vector logistic regression on mean-line hidden states from a packed
features directory, evaluates on a held-out split, and dumps everything the
plotter needs to `<out-dir>/probe_cache.npz`:

  logit_test        [N_test] float32   — probe logit per held-out line
  label_test        [N_test] int8      — 0/1 ground-truth label per line
  best_logit_th     scalar             — argmax-F1 decision boundary (on logit scale)
  easy_prune_hi     scalar             — 90th percentile of prune logits
  easy_keep_lo      scalar             — 10th percentile of keep logits
  auc, f1           scalars
  easy_prune_mass, ambiguous_mass, easy_keep_mass   scalars (fractions)
  n_train_lines, n_test_lines, pos_ratio_train, pos_ratio_test   scalars
  train_samples, test_samples                                    scalars

Re-run this only when features / split / probe config changes. The plot script
reads the cache and is cheap to iterate on.
"""

import json
from pathlib import Path

import numpy as np
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


def main(
    features_dir: Path = typer.Option(Path("features/0424_coder_next_patched_noquax")),
    train_ratio: float = typer.Option(0.9, help="Fraction of trajectories used for probe training"),
    seed: int = typer.Option(0),
    out_dir: Path = typer.Option(Path("figures/motivation")),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    with open(features_dir / "index.json") as f:
        idx = json.load(f)
    hidden_dim: int = idx["hidden_dim"]
    samples = idx["samples"]
    end_token = max(int(s["offset"]) + int(s["length"]) for s in samples)
    fp_size = (features_dir / "hidden_states.bin").stat().st_size
    hs_dtype = np.float16 if fp_size == end_token * hidden_dim * 2 else np.float32
    print(f"[features] {len(samples)} samples, end_token={end_token:,}, dim={hidden_dim}, dtype={hs_dtype.__name__}")

    hs = np.memmap(features_dir / "hidden_states.bin", dtype=hs_dtype, mode="r", shape=(end_token, hidden_dim))
    labels = np.memmap(features_dir / "token_labels.bin", dtype=np.int64, mode="r", shape=(end_token,))
    line_ids = np.memmap(features_dir / "token_line_ids.bin", dtype=np.int64, mode="r", shape=(end_token,))

    sample_ids = rng.permutation(len(samples))
    n_train = int(round(train_ratio * len(samples)))
    train_sids, test_sids = sample_ids[:n_train], sample_ids[n_train:]
    print(f"[split] train_samples={len(train_sids):,}  test_samples={len(test_sids):,}  "
          f"(ratio {train_ratio:.2f}:{1 - train_ratio:.2f})")

    def collect(sids: np.ndarray, tag: str) -> tuple[np.ndarray, np.ndarray]:
        Hp, Yp = [], []
        n_dropped = 0
        for i, si in enumerate(sids):
            if i % 2000 == 0 and i > 0:
                print(f"  [{tag}] {i:,}/{len(sids):,} samples  ({len(Hp):,} lines so far)")
            s = samples[int(si)]
            offset, length = int(s["offset"]), int(s["length"])
            seg_lab = np.asarray(labels[offset:offset + length])
            seg_lid = np.asarray(line_ids[offset:offset + length])
            valid = ((seg_lab == 0) | (seg_lab == 1)) & (seg_lid > 0)
            if not valid.any():
                n_dropped += 1
                continue
            ulids = np.unique(seg_lid[valid])
            ulids = ulids[ulids > 0]
            seg_hs = np.asarray(hs[offset:offset + length], dtype=np.float32)
            for lid in ulids:
                m = (seg_lid == int(lid)) & valid
                if not m.any():
                    continue
                Hp.append(seg_hs[m].mean(axis=0))
                Yp.append(int(np.round(seg_lab[m].mean())))
        print(f"  [{tag}] done: {len(sids):,} samples, {len(Hp):,} lines, "
              f"{n_dropped} samples dropped (no valid lines)")
        return np.stack(Hp).astype(np.float32), np.array(Yp, dtype=np.int64)

    H_tr, Y_tr = collect(train_sids, "train")
    H_te, Y_te = collect(test_sids, "test")
    print(f"[lines] train={len(Y_tr):,} (pos={Y_tr.mean():.3f})  test={len(Y_te):,} (pos={Y_te.mean():.3f})")

    clf = LogisticRegression(max_iter=400, C=1.0)
    clf.fit(H_tr, Y_tr)
    logit_te = H_te @ clf.coef_.flatten() + clf.intercept_[0]
    prob_te = 1.0 / (1.0 + np.exp(-logit_te))
    auc = roc_auc_score(Y_te, logit_te)

    th_grid = np.linspace(prob_te.min() + 1e-3, prob_te.max() - 1e-3, 121)
    f1s = [f1_score(Y_te, (prob_te > t).astype(np.int64)) for t in th_grid]
    best_idx = int(np.argmax(f1s))
    best_f1 = float(f1s[best_idx])
    best_th = float(th_grid[best_idx])
    best_logit_th = float(np.log(best_th / (1 - best_th)))
    print(f"[probe] raw mean-HS + plain LR:  AUC={auc:.4f}  F1={best_f1:.4f}@th={best_th:.3f}  "
          f"||w||={np.linalg.norm(clf.coef_):.3f}")

    s_keep = logit_te[Y_te == 1]
    s_prune = logit_te[Y_te == 0]
    p90_prune = float(np.percentile(s_prune, 90))
    p10_keep = float(np.percentile(s_keep, 10))
    if p90_prune <= p10_keep:
        easy_prune_hi = p90_prune
        easy_keep_lo = p10_keep
    else:
        easy_prune_hi = best_logit_th - 1.0
        easy_keep_lo = best_logit_th + 1.0
    easy_prune_mass = float((s_prune < easy_prune_hi).mean())
    easy_keep_mass = float((s_keep > easy_keep_lo).mean())
    ambiguous_mass = float(((logit_te >= easy_prune_hi) & (logit_te <= easy_keep_lo)).mean())
    print(f"[regions] easy_prune={easy_prune_mass:.0%} of prune  |  "
          f"ambiguous={ambiguous_mass:.0%} of all  |  easy_keep={easy_keep_mass:.0%} of keep")

    cache_path = out_dir / "probe_cache.npz"
    np.savez(
        cache_path,
        logit_test=logit_te.astype(np.float32),
        label_test=Y_te.astype(np.int8),
        best_logit_th=np.float32(best_logit_th),
        easy_prune_hi=np.float32(easy_prune_hi),
        easy_keep_lo=np.float32(easy_keep_lo),
        auc=np.float32(auc),
        f1=np.float32(best_f1),
        easy_prune_mass=np.float32(easy_prune_mass),
        ambiguous_mass=np.float32(ambiguous_mass),
        easy_keep_mass=np.float32(easy_keep_mass),
        n_train_lines=np.int64(len(Y_tr)),
        n_test_lines=np.int64(len(Y_te)),
        pos_ratio_train=np.float32(Y_tr.mean()),
        pos_ratio_test=np.float32(Y_te.mean()),
        train_samples=np.int64(len(train_sids)),
        test_samples=np.int64(len(test_sids)),
        features_dir=str(features_dir),
        train_ratio=np.float32(train_ratio),
    )
    print(f"[saved] {cache_path}  ({cache_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    typer.run(main)
