#!/usr/bin/env bash
# Train the pruning head against features extracted from the MiMo-V2-Flash
# backbone — reproduces the paper's `0507c_size_emb_mimo_v2_flash_22k` checkpoint.
# Same recipe as train_coder_next.sh, only the feature directory and log dir
# differ. The 22k corpus is shared (paper's noquax-filtered fixed-dataset-sonnet-0424).
set -euo pipefail
: "${FEATURES_DIR:=features/mimo_v2_flash/packed}"
: "${EVAL_DATA:=data/training_corpus_22k.jsonl}"
: "${LOG_DIR:=logs/0507c_size_emb_mimo_v2_flash_22k}"
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
