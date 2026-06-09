#!/usr/bin/env python3
"""Render the paper's F1-vs-Judge case figures (Figures 10/11).

Each case shows the same prompt scored by two heads side-by-side
(original | bad case | good case). Records are loaded from
``data/cases/f1_vs_judge.jsonl`` keyed by ``(case_id, model_variant)``,
with model_variant in {psbf, bce, focal}.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import typer
from matplotlib import rcParams

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "monospace"

GOLD_BG = "#fff3b0"
GOOD_BG = "#c8e6c9"
BAD_BG = "#ffcdd2"

# Each case: (case_id, good_variant, bad_variant, good_label, bad_label,
#             original_label, max_chars, prefix_to_strip)
CASES = [
    (
        "comparison",
        "psbf", "bce",
        "Per-sample balanced focal\nJudge $=$ 8/10  (F1 $=$ 0.49)",
        "Binary cross-entropy\nJudge $=$ 2/10  (F1 $=$ 0.53)",
        "Original Tool Response With GT",
        46, None,
    ),
    (
        "pandas_quantile",
        "focal", "bce",
        "Corpus-level focal loss\nJudge $=$ 8/10  (F1 $=$ 0.71)",
        "Binary cross-entropy\nJudge $=$ 3/10  (F1 $=$ 0.80)",
        "Original Tool Response With GT",
        48, None,
    ),
]


def truncate(text, n, prefix=None):
    text = text.replace("\t", "    ")
    if prefix and text.startswith(prefix):
        text = text[len(prefix):]
    # Use ASCII "..." (3 chars) so the visual width is predictable in
    # monospace fonts. Reserve the trailing 3 char-cells for the marker.
    if len(text) <= n:
        return text
    return text[: max(0, n - 3)] + "..."


def render_case(case_id, good, bad,
                good_label, bad_label, original_label, max_chars, prefix,
                out_dir: Path):
    tr_lines = good["tool_response"].split("\n")
    gold = set(good.get("kept_frags", []) or [])
    good_pred = set(good.get("pred_kept_lines", []) or [])
    bad_pred = set(bad.get("pred_kept_lines", []) or [])

    char_w = 0.072
    rect_char_w = 0.078  # buffer for highlight rectangle width
    row_h = 0.20
    pad_inner = 0.10
    pad_left = 0.12
    pad_col = 0.30
    ln_col = 0.30
    header_h = row_h * 3.0  # two-line header (loss name / scores) with breathing room
    col_text_w = max_chars * rect_char_w
    col_w = ln_col + 0.05 + col_text_w + pad_inner
    fig_w = pad_left + 3 * col_w + 2 * pad_col + pad_left
    n_rows = len(tr_lines)
    fig_h = header_h + n_rows * row_h + 0.30

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_facecolor("white")

    col_x = [
        pad_left,
        pad_left + col_w + pad_col,
        pad_left + 2 * (col_w + pad_col),
    ]
    # Order: Original | Bad | Good (user request: bad in middle, good on right)
    col_titles = [
        original_label,
        bad_label,
        good_label,
    ]

    def col_kept_fn(idx):
        if idx == 0:
            return lambda ln: (ln in gold, GOLD_BG)
        if idx == 1:
            return lambda ln: (ln in bad_pred, BAD_BG)
        return lambda ln: (ln in good_pred, GOOD_BG)

    header_lines_styles = [
        (0.25, 11.0, "#111", "bold"),
        (0.65, 9.5, "#444", "normal"),
    ]
    for cx, title in zip(col_x, col_titles):
        parts = title.split("\n")
        for i_line, (frac, fs, color, weight) in enumerate(header_lines_styles):
            if i_line >= len(parts):
                continue
            ax.text(cx, header_h * frac, parts[i_line],
                    color=color, fontsize=fs, family="monospace",
                    weight=weight, va="center", ha="left")

    body_top = header_h + 0.05
    text_end_offset = ln_col + 0.05 + max_chars * rect_char_w + 0.04
    for i_row, ln in enumerate(range(1, len(tr_lines) + 1), start=0):
        y = body_top + i_row * row_h
        if ln < 1 or ln > len(tr_lines):
            continue
        text = truncate(tr_lines[ln - 1].rstrip(), max_chars, prefix)
        for c_i, cx in enumerate(col_x):
            kept_fn = col_kept_fn(c_i)
            is_kept, bg = kept_fn(ln)
            if is_kept:
                ax.add_patch(mpatches.Rectangle(
                    (cx - 0.02, y), text_end_offset + 0.02, row_h,
                    facecolor=bg, edgecolor="none",
                ))
            ln_color = "#555" if is_kept else "#cfcfcf"
            ax.text(cx + ln_col - 0.04, y + row_h / 2, f"{ln:>3d}",
                    color=ln_color, fontsize=6.5, family="monospace",
                    va="center", ha="right")
            txt_color = "#1b1b1b" if is_kept else "#bdbdbd"
            ax.text(cx + ln_col + 0.05, y + row_h / 2, text,
                    color=txt_color, fontsize=6.8, family="monospace",
                    va="center", ha="left")

    body_bot = body_top + n_rows * row_h
    sep_top = header_h * 0.10
    for c_i in range(1, 3):
        sx = col_x[c_i] - pad_col / 2
        ax.plot([sx, sx], [sep_top, body_bot + 0.02],
                color="#999999", linewidth=0.6)

    rule_y = header_h - 0.02
    ax.plot([pad_left * 0.5, fig_w - pad_left * 0.5], [rule_y, rule_y],
            color="#999999", linewidth=0.6)

    out_pdf = out_dir / f"{case_id}.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.savefig(out_dir / f"{case_id}.png", bbox_inches="tight",
                pad_inches=0.04, dpi=200)
    plt.close(fig)
    print(f"  {case_id:18s} n={len(tr_lines):>3d} gold={len(gold):>2}"
          f"  good_kept={len(good_pred):>2}  bad_kept={len(bad_pred):>2}"
          f"  -> {out_pdf}")


def main(
    data: Path = typer.Option(Path("data/cases/f1_vs_judge.jsonl"),
        help="Pre-extracted case records ((case_id, model_variant) keyed)."),
    out_dir: Path = typer.Option(Path("figures/f1_vs_judge"),
        help="Output directory."),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, str], dict] = {}
    for line in open(data):
        r = json.loads(line)
        by_key[(r["case_id"], r["model_variant"])] = r
    for case_id, gv, bv, gl, bl, ol, mc, prefix in CASES:
        good = by_key[(case_id, gv)]
        bad = by_key[(case_id, bv)]
        render_case(case_id, good, bad, gl, bl, ol, mc, prefix, out_dir)


if __name__ == "__main__":
    typer.run(main)
