# `utils/`

Auxiliary scripts that support the analyses, figures, and result tables in the
SWE-Pruner-Pro paper. None are on the inference / training hot path; they all
operate on artifacts produced by the main pipeline.

## `motivation/` — Section 3 (linear separability of hidden states)

Backs the paper's motivation analysis (Figure 2 and Table 2). The probe finds a
1-vector linear projection over per-line mean hidden states that already
separates "keep" from "prune" lines at AUC ≈ 0.81 on held-out trajectories.

| File | Purpose |
|---|---|
| `fit_probe.py` | Train the held-out logistic probe on packed features and dump `probe_cache.npz`. |
| `lda.py` | Linear discriminant analysis variant of the probe (sanity check). |
| `dimred.py` | PCA / UMAP projection of mean-line HS for the appendix scatter plot. |
| `plot.py` | Render the histogram + KDE figure from the cached probe. |
| `eval_lr.py` | Re-evaluate a saved LR probe on a new holdout split. |

All read packed features written by `swe_pruner_pro.data.pack_features`.

## `figures/` — Appendix C qualitative figures

| File | Paper figure | Source data |
|---|---|---|
| `qualitative.py` | Figures 6/7/8/9 (read / search / listing / test cases) | `data/cases/qualitative.jsonl` |
| `f1_vs_judge.py` | Figures 10/11 (per-sample balanced focal vs corpus-level focal vs BCE) | `data/cases/f1_vs_judge.jsonl` |

Both load pre-extracted records keyed by `case_id` (and `model_variant` for
`f1_vs_judge`); see `data/cases/README.md` for provenance.

## `latency/` — Section 5.4 / Appendix `app:latency-opts`

| File | Purpose |
|---|---|
| `overhead_bench.py` | Replay sweqa-format MiMo trajectories against a pruner server, measure per-turn overhead (`/prune` latency) vs decode load (`/generate` latency). |
| `plot_paired.py` | Render the paired-bar latency comparison (off-engine vs in-engine head). |

Configure via env vars: `PRUNER_URL`, `TRAJ_ROOT`, `PRUNER_BACKBONE`.

## `stats/` — Result-table generators

| File | Tables |
|---|---|
| `swebench.py` | SWE-Bench Verified main table + ablation breakdowns (resolved %, token savings). |
| `sweqa_pro.py` | SWE-QA / SWE-QA-Pro / Oolong token-savings figures. |

Configure via env vars: `SWEQA_RESULTS_ROOT`, `OOLONG_RESULTS_ROOT`,
`STATS_OUT_DIR`.
