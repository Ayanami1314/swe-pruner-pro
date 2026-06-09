"""Motivation figure: how do common dimensionality-reduction methods spread keep / prune lines?

Replaces the LogReg probe in `motivation_pca.py` with a 2-D embedding from each of:
PCA, LDA, KernelPCA (RBF), t-SNE, MDS, Isomap. Produces one PDF/PNG with a grid
of 2-D scatter plots — no probe, no decision boundary, just raw geometry.

Subsampled to keep t-SNE / MDS / Isomap (which are O(n²)) tractable.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.preprocessing import StandardScaler


def collect_lines(
    samples: list[dict],
    sids: np.ndarray,
    hs: np.memmap,
    labels: np.memmap,
    line_ids: np.memmap,
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
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
    print(f"  [{tag}] done: {len(sids):,} samples, {len(Hp):,} lines, {n_dropped} samples dropped")
    return np.stack(Hp).astype(np.float32), np.array(Yp, dtype=np.int64)


def run_method(name: str, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    t0 = time.time()
    if name == "PCA":
        Z = PCA(n_components=2, random_state=0).fit_transform(X)
    elif name == "LDA":
        Z1 = LinearDiscriminantAnalysis(n_components=1).fit_transform(X, y).flatten()
        pc = PCA(n_components=2, random_state=0).fit_transform(X)
        residual = pc[:, 1]
        residual = (residual - residual.mean()) / (residual.std() + 1e-8) * (Z1.std() + 1e-8)
        Z = np.stack([Z1, residual], axis=1)
    elif name == "KernelPCA(RBF)":
        gamma = 1.0 / (X.shape[1] * X.var())
        Z = KernelPCA(n_components=2, kernel="rbf", gamma=gamma, random_state=0).fit_transform(X)
    elif name == "t-SNE":
        Z = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(X)
    elif name == "MDS":
        Z = MDS(n_components=2, n_init=1, max_iter=200, random_state=0, normalized_stress="auto").fit_transform(X)
    elif name == "Isomap":
        Z = Isomap(n_components=2, n_neighbors=15).fit_transform(X)
    else:
        raise ValueError(name)
    print(f"  [{name}] fit done in {time.time() - t0:.1f}s, shape={Z.shape}")
    return Z


def main(
    features_dir: Path = typer.Option(Path("features/0424_coder_next_patched_noquax")),
    n_lines: int = typer.Option(6000, help="Total lines to subsample (balanced across classes)"),
    seed: int = typer.Option(0),
    out_dir: Path = typer.Option(Path("figures/motivation")),
    standardize: bool = typer.Option(True, help="Z-score features before reduction"),
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
    H, Y = collect_lines(samples, sample_ids, hs, labels, line_ids, "all")
    print(f"[lines] total={len(Y):,}  pos_ratio={Y.mean():.3f}")

    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == 0)[0]
    per_class = n_lines // 2
    pos_pick = rng.choice(pos_idx, size=min(per_class, len(pos_idx)), replace=False)
    neg_pick = rng.choice(neg_idx, size=min(per_class, len(neg_idx)), replace=False)
    pick = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(pick)
    Xs, Ys = H[pick], Y[pick]
    print(f"[subsample] kept {len(Ys):,} lines  (keep={int((Ys==1).sum()):,}, prune={int((Ys==0).sum()):,})")

    X_in = StandardScaler().fit_transform(Xs) if standardize else Xs
    methods = ["PCA", "LDA", "KernelPCA(RBF)", "Isomap", "t-SNE", "MDS"]
    embeddings = {m: run_method(m, X_in, Ys) for m in methods}

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.6))
    axes = axes.flatten()
    color_prune = "#cf3a3a"
    color_keep = "#1f7a1f"

    for ax, name in zip(axes, methods):
        Z = embeddings[name]
        m_keep = Ys == 1
        m_prune = Ys == 0
        ax.scatter(Z[m_prune, 0], Z[m_prune, 1], s=4, c=color_prune, alpha=0.45, linewidth=0, label=f"prune (n={m_prune.sum():,})")
        ax.scatter(Z[m_keep, 0], Z[m_keep, 1], s=4, c=color_keep, alpha=0.45, linewidth=0, label=f"keep (n={m_keep.sum():,})")
        ax.set_title(name, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.4)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85, frameon=True, markerscale=2.0)

    fig.suptitle(
        f"Last-layer HS — 2-D embeddings via 6 dim-reduction methods  ({len(Ys):,} lines, dim={hidden_dim}→2)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.93, bottom=0.03, wspace=0.06, hspace=0.14)

    out_pdf = out_dir / "motivation_dimred.pdf"
    out_png = out_dir / "motivation_dimred.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    print(f"[saved] {out_pdf}\n[saved] {out_png}")


if __name__ == "__main__":
    typer.run(main)
