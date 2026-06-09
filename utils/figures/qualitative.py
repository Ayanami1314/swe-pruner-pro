#!/usr/bin/env python3
"""Render qualitative-case figures (PDF) for the paper's Appendix C
(Figures 6/7/8/9 — read / search / listing / test).

Loads pre-extracted records from ``data/cases/qualitative.jsonl`` keyed by
``case_id``. The accompanying README documents the (case_id, instance_id,
step_idx) mapping for the paper's checkpoint.
"""
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import typer
from matplotlib import rcParams

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "monospace"

ELLIPSIS = -1

# (case_id, lines_to_show, max_text_chars)
CASES = [
    (
        "read",
        # cat -n /testbed/snapshot_dbg_cli/__init__.py (44 lines)
        [1, ELLIPSIS,
         14, 15, 17, 18, 19, ELLIPSIS,
         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
        110,
    ),
    (
        "search",
        # cat -n templating.py | grep -A 25 'def render_template_string'
        [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        110,
    ),
    (
        "listing",
        # ls -la /home/legacy_admin/configs/{cron,db,secrets}
        [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
         19, 20, 21, 22, 23, 24, 25],
        105,
    ),
    (
        "test",
        # python test_fix.py — TypeError traceback
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
         14, 15, 16, 17, 18, 19, 20, 21],
        115,
    ),
]


def gutter_color(score: float) -> tuple[float, float, float]:
    s = max(0.0, min(1.0, score))
    s_norm = max(0.0, min(1.0, (s - 0.30) / 0.55))
    s_norm = s_norm ** 0.85
    r = 1.0 - s_norm * 0.95
    g = 1.0 - s_norm * 0.45
    b = 1.0 - s_norm * 0.95
    return (r, g, b)


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "\u2026"


def render_case(case_id: str, lines_spec: list[int],
                max_chars: int, record: dict, out_dir: Path) -> Path:
    tr_lines = record["tool_response"].split("\n")
    scores = record.get("pred_line_scores", {}) or {}
    gt = set(record.get("kept_frags", []) or [])
    pred = set(record.get("pred_kept_lines", []) or [])

    char_w = 0.072
    row_h = 0.165
    gutter_w = 0.08
    ln_w = 0.30
    g_w = 0.16
    p_w = 0.16
    score_w = 0.34
    pad_left = 0.04
    pad_text = 0.07

    text_w = max_chars * char_w
    fig_w = pad_left + gutter_w + ln_w + g_w + p_w + score_w + pad_text + text_w + 0.05
    n_rows = len(lines_spec)
    fig_h = n_rows * row_h + 0.08

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_facecolor("white")

    x_gutter = pad_left
    x_ln = x_gutter + gutter_w
    x_g = x_ln + ln_w
    x_p = x_g + g_w
    x_score = x_p + p_w
    x_text = x_score + score_w + pad_text

    for i, ln in enumerate(lines_spec):
        y = i * row_h
        if ln == ELLIPSIS:
            ax.text(
                x_text + 0.1, y + row_h / 2,
                ". . .",
                color="#aaa", fontsize=7,
                family="monospace", va="center", ha="left",
            )
            continue
        if ln < 1 or ln > len(tr_lines):
            continue
        text = tr_lines[ln - 1].rstrip()
        raw = scores.get(str(ln))
        score = float(raw) if raw is not None else 0.0
        is_gt = ln in gt
        is_pred = ln in pred

        if is_pred:
            ax.add_patch(mpatches.Rectangle(
                (x_ln - 0.02, y), fig_w - x_ln + 0.02 - 0.02, row_h,
                facecolor="#fff5d6", edgecolor="none",
            ))
        ax.add_patch(mpatches.Rectangle(
            (x_gutter, y + 0.005), gutter_w, row_h - 0.01,
            facecolor=gutter_color(score), edgecolor="none",
        ))

        ax.text(
            x_ln + 0.03, y + row_h / 2, f"{ln:>3d}",
            color="#888", fontsize=6.0,
            family="monospace", va="center", ha="left",
        )

        g_box_x = x_g + 0.025
        g_box_y = y + row_h * 0.18
        g_box_w = g_w - 0.05
        g_box_h = row_h * 0.64
        if is_gt:
            ax.add_patch(mpatches.Rectangle(
                (g_box_x, g_box_y), g_box_w, g_box_h,
                facecolor="#2e7d32", edgecolor="#1b5e20", linewidth=0.3,
            ))
        else:
            ax.add_patch(mpatches.Rectangle(
                (g_box_x, g_box_y), g_box_w, g_box_h,
                facecolor="none", edgecolor="#dadada", linewidth=0.4,
            ))

        p_box_x = x_p + 0.025
        p_box_y = y + row_h * 0.18
        p_box_w = p_w - 0.05
        p_box_h = row_h * 0.64
        if is_pred:
            ax.add_patch(mpatches.Rectangle(
                (p_box_x, p_box_y), p_box_w, p_box_h,
                facecolor="#1565c0", edgecolor="#0d47a1", linewidth=0.3,
            ))
        else:
            ax.add_patch(mpatches.Rectangle(
                (p_box_x, p_box_y), p_box_w, p_box_h,
                facecolor="none", edgecolor="#dadada", linewidth=0.4,
            ))

        score_color = "#222" if is_pred else "#888"
        ax.text(
            x_score + score_w - 0.04, y + row_h / 2, f"{score:.2f}",
            color=score_color, fontsize=6.0,
            family="monospace", va="center", ha="right",
        )

        text_color = "#111" if is_pred else "#9a9a9a"
        ax.text(
            x_text, y + row_h / 2, truncate(text, max_chars),
            color=text_color, fontsize=6.5,
            family="monospace", va="center", ha="left",
        )

    ax.add_patch(mpatches.Rectangle(
        (0, 0), fig_w, fig_h,
        fill=False, edgecolor="#cfcfcf", linewidth=0.5,
    ))

    out_path = out_dir / f"{case_id}.pdf"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_dir / f"{case_id}.png", bbox_inches="tight",
                pad_inches=0.02, dpi=160)
    plt.close(fig)
    return out_path


def main(
    data: Path = typer.Option(Path("data/cases/qualitative.jsonl"),
        help="Pre-extracted case records (one per case_id)."),
    out_dir: Path = typer.Option(Path("figures/qualitative"),
        help="Output directory for PDFs / PNGs."),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    for line in open(data):
        r = json.loads(line)
        records[r["case_id"]] = r
    for case_id, lines_spec, max_chars in CASES:
        d = records[case_id]
        out = render_case(case_id, lines_spec, max_chars, d, out_dir)
        print(f"  {case_id:8s} iid={d.get('instance_id')} step={d.get('step_idx')} "
              f"n_lines={len(d['tool_response'].splitlines())} "
              f"gt={len(d.get('kept_frags', []))} pred={len(d.get('pred_kept_lines', []))} -> {out}")


if __name__ == "__main__":
    typer.run(main)
