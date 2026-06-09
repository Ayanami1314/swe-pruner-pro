"""Generate the latency figure used in the paper.

Reads the latency bench JSON (per-trajectory wall times for /generate and
/prune at fixed concurrency) and emits a paired-bar figure: each trajectory
contributes one bar pair, x-axis sorted by trajectory length (# turns).

Output: paper/figures/latency_C_paired_bars.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer


cli = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@cli.command()
def main(
    bench_json: Path = typer.Option(
        Path("results/overhead_bench_sweqa_mimo_embedded.json"),
        "--bench-json", "-i",
        help="Output of bench/overhead_bench_sweqa.py.",
    ),
    out_dir: Path = typer.Option(
        Path("paper/figures"), "--out-dir", "-o",
        help="Where to drop the .pdf/.png pair.",
    ),
    stem: str = typer.Option(
        "latency_C_paired_bars", "--stem",
        help="Output filename stem; '.pdf' and '.png' are appended.",
    ),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = json.loads(bench_json.read_text())
    per = sorted(d["runs"][0]["per_traj"], key=lambda r: r["K"])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 14,
        "font.weight": "bold",
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#222",
        "xtick.color": "#222",
        "ytick.color": "#222",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    GEN_COLOR = "#3a6ea5"
    PRUNE_COLOR = "#d1495b"

    fig, ax = plt.subplots(figsize=(8, 6))  # 4:3
    xs = np.arange(len(per))
    w = 0.42
    tb = [r["T_base_ms"] / 1000 for r in per]
    tp = [r["T_prune_ms"] / 1000 for r in per]

    ax.bar(xs - w / 2, tb, w, label="Generate",
           color=GEN_COLOR, edgecolor="white", linewidth=0.8)
    ax.bar(xs + w / 2, tp, w, label="Extra Prune",
           color=PRUNE_COLOR, edgecolor="white", linewidth=0.8)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['K']}" for r in per])
    ax.set_ylabel("Wall Time (s)")
    ax.set_xlabel("Trajectory (sorted by # turns)")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, color="#bbb", alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(frameon=False, loc="upper left", prop={"weight": "bold", "size": 14})

    plt.tight_layout()
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    typer.echo(f"saved {pdf} {png}")


if __name__ == "__main__":
    cli()
