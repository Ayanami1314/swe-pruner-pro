#!/usr/bin/env bash
# Train the pruning head against features extracted from the Qwen3-Coder-Next
# backbone — reproduces the paper's `0507b_size_emb_coder_next_22k` checkpoint.
# Defaults: 10 epochs, lr 3e-5 -> 1.5e-5 cosine, per-sample balanced focal (γ=2),
# length-aware embedding (8 log-spaced buckets, additive at pre_head, zero-init).
set -euo pipefail
: "${FEATURES_DIR:=features/coder_next/packed}"
: "${EVAL_DATA:=data/training_corpus_22k.jsonl}"
: "${LOG_DIR:=logs/0507b_size_emb_coder_next_22k}"
: "${NPROC:=8}"

torchrun --nproc-per-node "${NPROC}" -m swe_pruner_pro.train.train \
    --features-dir "${FEATURES_DIR}" \
    --eval-data "${EVAL_DATA}" \
    --log-dir "${LOG_DIR}" \
    --epochs 10 \
    --lr 3e-5 \
    --min-lr 1.5e-5 \
    --warmup-ratio 0.05 \
    --batch-size 4 \
    --dropout 0.4 \
    --use-size-emb \
    --loss psbf \
    --focal-gamma 2.0
