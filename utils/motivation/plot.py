"""Render the motivation figure from a cached probe (see
`train/scripts/fit_motivation_probe.py`).

Fast to iterate on — no LR retraining. Loads
`<cache-dir>/probe_cache.npz` and draws a single-panel figure: class-conditional
histograms of the linear-probe score, KDE overlays, decision boundary, and
"easy prune / ambiguous / easy keep" region percentages.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from scipy.stats import gaussian_kde


def main(
    cache_dir: Path = typer.Option(Path("figures/motivation")),
    n_bins: int = typer.Option(60, help="Histogram bins over the score range"),
    kde_bw: float = typer.Option(0.25, help="gaussian_kde bw_method"),
    out_dir: Path = typer.Option(Path("figures/motivation")),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(cache_dir / "probe_cache.npz", allow_pickle=False)
    logit_te = cache["logit_test"].astype(np.float32)
    Y_te = cache["label_test"].astype(np.int64)
    best_logit_th = float(cache["best_logit_th"])
    easy_prune_hi = float(cache["easy_prune_hi"])
    easy_keep_lo = float(cache["easy_keep_lo"])
    auc = float(cache["auc"])
    f1 = float(cache["f1"])
    easy_prune_mass = float(cache["easy_prune_mass"])
    easy_keep_mass = float(cache["easy_keep_mass"])
    ambiguous_mass = float(cache["ambiguous_mass"])
    print(f"[cache] loaded probe_cache.npz   AUC={auc:.4f}  F1={f1:.4f}   "
          f"{int(cache['n_test_lines']):,} test lines  pos_ratio={float(cache['pos_ratio_test']):.3f}")

    s_keep = logit_te[Y_te == 1]
    s_prune = logit_te[Y_te == 0]

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelweight": "bold"})
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    x_lo = float(min(s_keep.min(), s_prune.min())) - 0.3
    x_hi = float(max(s_keep.max(), s_prune.max())) + 0.3
    bins = np.linspace(x_lo, x_hi, n_bins)
    grid_x = np.linspace(x_lo, x_hi, 400)
    kde_p = gaussian_kde(s_prune, bw_method=kde_bw)(grid_x)
    kde_k = gaussian_kde(s_keep, bw_method=kde_bw)(grid_x)

    ax.hist(s_prune, bins=bins, color="#cf3a3a", alpha=0.38, density=True,
            edgecolor="white", linewidth=0.3, zorder=2, label="Pruned")
    ax.hist(s_keep, bins=bins, color="#1f7a1f", alpha=0.38, density=True,
            edgecolor="white", linewidth=0.3, zorder=2, label="Kept")
    ax.plot(grid_x, kde_p, color="#9a1a1a", linewidth=2.4, zorder=3)
    ax.plot(grid_x, kde_k, color="#0a4f0a", linewidth=2.4, zorder=3)
    ax.axvline(best_logit_th, color="black", linestyle="--", linewidth=1.5, zorder=4)

    hist_p_max = float(np.histogram(s_prune, bins=bins, density=True)[0].max())
    hist_k_max = float(np.histogram(s_keep, bins=bins, density=True)[0].max())
    y_top = max(kde_p.max(), kde_k.max(), hist_p_max, hist_k_max) * 1.08
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, y_top)

    ax.set_xlabel(r"Linear probe score   $\mathbf{w \cdot h_{\mathrm{line}} + b}$", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.legend(loc="upper left", fontsize=12, framealpha=0.92, frameon=False,
              prop={"weight": "bold"})
    ax.grid(True, alpha=0.18)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.14)
    out_pdf = out_dir / "motivation_pca.pdf"
    out_png = out_dir / "motivation_pca.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    print(f"[saved] {out_pdf}\n[saved] {out_png}")


if __name__ == "__main__":
    typer.run(main)
