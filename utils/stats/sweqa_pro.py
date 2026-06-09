#!/usr/bin/env python3
"""Plot SWE-QA / SWE-QA-Pro / Oolong efficiency: baseline vs SWE-Pruner Pro."""
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import typer
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

ROOT_SWEQA = Path(os.environ.get("SWEQA_RESULTS_ROOT", "results/sweqa"))
ROOT_OOLONG = Path(os.environ.get("OOLONG_RESULTS_ROOT", "results/oolong"))

BENCHES = {
    "sweqa": dict(
        root=ROOT_SWEQA, suffix="", glob_pat="*/*_answers.jsonl",
        cases=[
            ("Qwen3-Coder-Next",
             "results_matrix_0520_20260520-231930_coder-next_baseline_0522final",
             "results_matrix_vote_20260522-163235_coder-next_pruner_capfix"),
            ("MiMo-V2-Flash",
             "results_matrix_0520_20260520-231930_mimo_baseline_0522final",
             "results_matrix_vote_20260522-163235_mimo_pruner_capfix"),
        ],
    ),
    "sweqa-pro": dict(
        root=ROOT_SWEQA, suffix="_pro", glob_pat="*/*_answers.jsonl",
        cases=[
            ("Qwen3-Coder-Next",
             "results_matrix_0520_20260520-231930_coder-next_baseline_pro_0522final",
             "results_matrix_vote_20260522-163235_coder-next_pruner_pro_capfix"),
            ("MiMo-V2-Flash",
             "results_matrix_0520_20260520-231930_mimo_baseline_pro_0522final",
             "results_matrix_vote_20260522-163235_mimo_pruner_pro_capfix"),
        ],
    ),
    "oolong": dict(
        root=ROOT_OOLONG, suffix="", glob_pat="*/*/full_output.jsonl",
        cases=[
            ("Qwen3-Coder-Next",
             "results_matrix_0520_20260520-231930_coder-next_baseline_0522final",
             "results_matrix_vote_20260522-163235_coder-next_pruner_capfix"),
            ("MiMo-V2-Flash",
             "results_matrix_0520_20260520-231930_mimo_baseline_0522final",
             "results_matrix_vote_20260522-163235_mimo_pruner_capfix"),
        ],
    ),
}

METRICS = [
    ("total_prompt_tokens", "Prompt Tokens"),
    ("total_completion_tokens", "Completion Tokens"),
    ("total_tokens", "Total Tokens"),
]


def load_records(d: Path, glob_pat: str):
    out = []
    for f in sorted(d.glob(glob_pat)):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            st = r.get("agent_stats") or {}
            p = int(st.get("total_prompt_tokens", 0) or 0)
            c = int(st.get("total_completion_tokens", 0) or 0)
            out.append({
                "total_prompt_tokens": p,
                "total_completion_tokens": c,
                "total_tokens": p + c,
            })
    return out


def add_pair(fig, row, col, data1, data2, color1, color2, c1_line, c2_line, show_legend, label1, label2, axis_idx):
    combined = data1 + data2
    if not combined:
        return
    sd = sorted(combined)
    p95 = sd[int(len(sd) * 0.95)] if len(sd) > 1 else max(combined)
    x_max = max(p95 * 1.1, 20)
    bin_size = x_max / 12
    xbins = dict(start=0, end=x_max, size=bin_size)

    fig.add_trace(go.Histogram(x=data1, name=f"<b>{label1}</b>", marker_color=color1,
        opacity=0.6, xbins=xbins, autobinx=False, histnorm="probability density",
        showlegend=show_legend, legendgroup="g1", marker=dict(line=dict(width=0))),
        row=row, col=col)
    fig.add_trace(go.Histogram(x=data2, name=f"<b>{label2}</b>", marker_color=color2,
        opacity=0.6, xbins=xbins, autobinx=False, histnorm="probability density",
        showlegend=show_legend, legendgroup="g2", marker=dict(line=dict(width=0))),
        row=row, col=col)

    def kde_peak(data, line_color):
        if len(data) <= 1 or np.std(data) == 0:
            return None
        try:
            k = gaussian_kde(data, bw_method=0.45)
            xr = np.linspace(0, x_max, 800)
            yv = k(xr)
            pi = int(np.argmax(yv))
            fig.add_trace(go.Scatter(x=xr, y=yv, mode="lines",
                line=dict(color=line_color, width=5), showlegend=False,
                hoverinfo="skip"), row=row, col=col)
            return float(xr[pi]), float(yv[pi])
        except Exception:
            return None

    p1 = kde_peak(data1, c1_line)
    p2 = kde_peak(data2, c2_line)

    # Arrow uses MEAN (robust to bimodal); peaks only used to size y-range
    m1 = float(np.mean(data1)) if data1 else 0.0
    m2 = float(np.mean(data2)) if data2 else 0.0
    y_range = None
    if m1 > 0 and m2 > 0 and m1 > m2:
        saving = (m1 - m2) / m1 * 100
        max_y = max(p1[1] if p1 else 0, p2[1] if p2 else 0)
        if max_y == 0:
            max_y = 1.0
        arrow_y = max_y * 1.2
        y_range = [0, max_y * 1.6]
        xref = f"x{axis_idx}" if axis_idx > 1 else "x"
        yref = f"y{axis_idx}" if axis_idx > 1 else "y"
        fig.add_shape(type="line", x0=m1, y0=0, x1=m1, y1=arrow_y,
            xref=xref, yref=yref, opacity=0.85,
            line=dict(color=c1_line, width=4, dash="dash"))
        fig.add_shape(type="line", x0=m2, y0=0, x1=m2, y1=arrow_y,
            xref=xref, yref=yref, opacity=0.85,
            line=dict(color=c2_line, width=4, dash="dash"))
        fig.add_annotation(x=m2, y=arrow_y, ax=m1, ay=arrow_y,
            xref=xref, yref=yref, axref=xref, ayref=yref,
            text="", showarrow=True, arrowhead=2, arrowsize=1.3,
            arrowwidth=3, arrowcolor="#d62728")
        fig.add_annotation(x=(m1 + m2) / 2, y=arrow_y, xref=xref, yref=yref,
            text=f"<b>↓ {saving:.1f}%</b>", showarrow=False, yshift=22,
            font=dict(size=22, color="#d62728", family="Arial"))

    return x_max, y_range


def main(bench: str = typer.Argument("sweqa-pro")):
    cfg = BENCHES[bench]
    color1_fill = "rgba(120,120,120,0.45)"
    color1_line = "rgb(80,80,80)"
    color2_fill = "rgba(255,140,30,0.5)"
    color2_line = "rgb(230,110,10)"

    fig = make_subplots(
        rows=2, cols=3,
        vertical_spacing=0.18, horizontal_spacing=0.05,
        subplot_titles=[m[1] for m in METRICS] * 2,
    )

    for r, (bb_name, base_sub, vote_sub) in enumerate(cfg["cases"], start=1):
        recs_b = load_records(cfg["root"] / base_sub, cfg["glob_pat"])
        recs_v = load_records(cfg["root"] / vote_sub, cfg["glob_pat"])
        print(f"[{bench}] {bb_name}: baseline={len(recs_b)}  vote={len(recs_v)}")
        for c, (key, _label) in enumerate(METRICS, start=1):
            d1 = [s[key] for s in recs_b]
            d2 = [s[key] for s in recs_v]
            axis_idx = (r - 1) * 3 + c
            add_pair(fig, r, c, d1, d2,
                color1_fill, color2_fill, color1_line, color2_line,
                show_legend=(r == 1 and c == 1),
                label1="w/o SWE-Pruner Pro", label2="w/ SWE-Pruner Pro",
                axis_idx=axis_idx)

    for r, (bb_name, _, _) in enumerate(cfg["cases"], start=1):
        y = 0.78 if r == 1 else 0.22
        fig.add_annotation(x=-0.05, y=y, xref="paper", yref="paper",
            text=f"<b>({chr(96+r)}) {bb_name}</b>", showarrow=False,
            font=dict(size=24, family="Arial", color="black"),
            textangle=-90, xanchor="center", yanchor="middle")

    # Style axes
    for axis in fig.layout:
        if axis.startswith("xaxis"):
            fig.layout[axis].update(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                zeroline=True, zerolinecolor="black", zerolinewidth=1.5,
                showline=True, linecolor="black", linewidth=1.5,
                ticks="outside", tickfont=dict(size=18, family="Arial"))
        if axis.startswith("yaxis"):
            fig.layout[axis].update(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                showline=True, linecolor="black", linewidth=1.5,
                ticks="outside", showticklabels=False, rangemode="tozero")

    # Frequency label only on left column
    for r in range(1, 3):
        fig.update_yaxes(title_text="<b>Frequency</b>",
            title_font=dict(size=22, family="Arial"), row=r, col=1)

    # Bold subplot titles
    for ann in fig.layout.annotations:
        if any(m[1] in (ann.text or "") for m in METRICS):
            ann.update(font=dict(size=22, family="Arial", color="black"))

    fig.update_layout(
        font=dict(family="Arial", size=20, color="black"),
        template="simple_white",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.07,
            xanchor="center", x=0.5, font=dict(size=22, family="Arial"),
            bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=110, r=30, t=80, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
        bargap=0.05,
        width=1600, height=900,
    )

    out_pdf = Path(os.environ.get("STATS_OUT_DIR", ".")) / f"{bench}-token-save.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_html = out_pdf.with_suffix(".html")
    fig.write_html(out_html)
    print(f"HTML: {out_html}")
    try:
        fig.write_image(out_pdf, format="pdf", scale=2)
        print(f"PDF: {out_pdf}")
    except Exception as e:
        print(f"PDF skipped: {e}")
    try:
        fig.write_image(out_png, format="png", scale=2)
        print(f"PNG: {out_png}")
    except Exception as e:
        print(f"PNG skipped: {e}")


if __name__ == "__main__":
    typer.run(main)
