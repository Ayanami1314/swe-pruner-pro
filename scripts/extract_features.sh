#!/usr/bin/env bash
# Extract last-layer hidden states from the agent backbone for from-features training,
# then pack the per-sample shards into memmap .bin files for zero-copy access.
# Requires the patched SGLang (see patches/sglang/README.md).
set -euo pipefail
: "${BACKBONE:?Set BACKBONE to the backbone model path or HF id}"
: "${INPUT_JSONL:=data/training_corpus_22k.jsonl}"
: "${FEATURES_DIR:=features/run0}"
mkdir -p "${FEATURES_DIR}"

python -m swe_pruner_pro.data.extract_features extract \
    "${INPUT_JSONL}" \
    -o "${FEATURES_DIR}/npz" \
    --model "${BACKBONE}" \
    --tp "${TP_SIZE:-8}"

python -m swe_pruner_pro.data.pack_features \
    "${FEATURES_DIR}/npz" \
    -o "${FEATURES_DIR}/packed"

echo "Features ready at ${FEATURES_DIR}/packed"
