"""Motivation figure (LDA): keep / prune separability under Linear Discriminant Analysis.

Two-stage: heavy data prep + LDA fit are cached to disk so iteration on the
plot is instant. Single-column figure with the main scatter (LDA-1 vs PCA
residual) and a top KDE marginal showing keep / prune separability and the
fraction of lines in the easy-prune / ambiguous / easy-keep regions.

Usage:
  uv run python -m train.scripts.motivation_lda                # uses cache if present
  uv run python -m train.scripts.motivation_lda --refit        # force re-fit
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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


def fit_and_cache(
    features_dir: Path,
    cache_path: Path,
    n_lines: int,
    seed: int,
    standardize: bool,
) -> dict:
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

    t0 = time.time()
    lda = LinearDiscriminantAnalysis(n_components=1).fit(X_in, Ys)
    z_lda = lda.transform(X_in).flatten().astype(np.float32)
    train_acc = float(lda.score(X_in, Ys))
    print(f"[LDA] fit {time.time() - t0:.1f}s  ||w||={np.linalg.norm(lda.coef_):.3f}  acc={train_acc:.4f}")

    w = lda.coef_.flatten()
    w_unit = w / (np.linalg.norm(w) + 1e-12)
    X_perp = X_in - np.outer(X_in @ w_unit, w_unit)
    t0 = time.time()
    z_res = PCA(n_components=1, random_state=0).fit_transform(X_perp).flatten().astype(np.float32)
    z_res = (z_res - z_res.mean()) / (z_res.std() + 1e-8) * (z_lda.std() + 1e-8)
    print(f"[PCA-residual] fit {time.time() - t0:.1f}s")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        z_lda=z_lda, z_res=z_res, Ys=Ys.astype(np.int64),
        hidden_dim=np.int64(hidden_dim),
        train_acc=np.float32(train_acc),
    )
    print(f"[cache] saved {cache_path}")
    return {"z_lda": z_lda, "z_res": z_res, "Ys": Ys, "hidden_dim": hidden_dim, "train_acc": train_acc}


def load_cache(cache_path: Path) -> dict:
    d = np.load(cache_path)
    print(f"[cache] loaded {cache_path}  ({d['Ys'].shape[0]:,} lines)")
    return {
        "z_lda": d["z_lda"], "z_res": d["z_res"], "Ys": d["Ys"],
        "hidden_dim": int(d["hidden_dim"]),
        "train_acc": float(d["train_acc"]),
    }


def render(data: dict, out_dir: Path) -> None:
    z_lda = data["z_lda"]
    z_res = data["z_res"]
    Ys = data["Ys"]

    s_keep = z_lda[Ys == 1]
    s_prune = z_lda[Ys == 0]
    mu_k, mu_p = float(s_keep.mean()), float(s_prune.mean())
    var_k, var_p = float(s_keep.var() + 1e-12), float(s_prune.var() + 1e-12)
    decision = (mu_k * var_p + mu_p * var_k) / (var_k + var_p)

    p90_prune = float(np.percentile(s_prune, 90))
    p10_keep = float(np.percentile(s_keep, 10))
    if p90_prune <= p10_keep:
        easy_prune_hi, easy_keep_lo = p90_prune, p10_keep
    else:
        easy_prune_hi = decision - 1.0
        easy_keep_lo = decision + 1.0
    print(f"[LDA-1] mean(keep)={mu_k:+.3f}  mean(prune)={mu_p:+.3f}  decision={decision:+.3f}")

    color_prune = "#cf3a3a"
    color_keep = "#1f7a1f"

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
    })
    fig, ax_main = plt.subplots(figsize=(6.4, 4.8))

    x_lo = float(z_lda.min()) - 0.4
    x_hi = float(z_lda.max()) + 0.4
    y_lo = float(z_res.min()) - 0.4
    y_hi = float(z_res.max()) + 0.4

    m_keep = Ys == 1
    m_prune = Ys == 0
    ax_main.axvspan(x_lo, easy_prune_hi, color=color_prune, alpha=0.06, zorder=0)
    ax_main.axvspan(easy_prune_hi, easy_keep_lo, color="#888888", alpha=0.05, zorder=0)
    ax_main.axvspan(easy_keep_lo, x_hi, color=color_keep, alpha=0.06, zorder=0)
    ax_main.scatter(z_lda[m_prune], z_res[m_prune], s=6, c=color_prune, alpha=0.32, linewidth=0,
                    label="Pruned")
    ax_main.scatter(z_lda[m_keep], z_res[m_keep], s=6, c=color_keep, alpha=0.32, linewidth=0,
                    label="Kept")
    ax_main.axvline(decision, color="black", linestyle="--", linewidth=1.6, zorder=2)
    ax_main.set_xlim(x_lo, x_hi)
    ax_main.set_ylim(y_lo, y_hi)
    ax_main.set_xlabel(r"LDA Discriminant Axis  $\mathbf{w \cdot h_{\mathrm{line}}}$")
    ax_main.set_ylabel("Orthogonal Axis (PCA Residual)")
    leg = ax_main.legend(loc="upper left", framealpha=0.93, frameon=True,
                         markerscale=3.5, prop={"weight": "bold"})
    leg.get_frame().set_edgecolor("#888")
    ax_main.grid(True, alpha=0.18)
    for label in ax_main.get_xticklabels() + ax_main.get_yticklabels():
        label.set_fontweight("bold")

    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.14)

    out_pdf = out_dir / "motivation_lda.pdf"
    out_png = out_dir / "motivation_lda.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"[saved] {out_pdf}\n[saved] {out_png}")


def main(
    features_dir: Path = typer.Option(Path("features/0424_coder_next_patched_noquax")),
    n_lines: int = typer.Option(10000, help="Total lines to subsample (balanced across classes)"),
    seed: int = typer.Option(0),
    out_dir: Path = typer.Option(Path("figures/motivation")),
    cache_path: Path = typer.Option(Path("figures/motivation/motivation_lda_cache.npz")),
    standardize: bool = typer.Option(True, help="Z-score features before LDA / PCA"),
    refit: bool = typer.Option(False, help="Force re-fit (ignore existing cache)"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refit:
        data = load_cache(cache_path)
    else:
        data = fit_and_cache(features_dir, cache_path, n_lines, seed, standardize)
    render(data, out_dir)


if __name__ == "__main__":
    typer.run(main)
