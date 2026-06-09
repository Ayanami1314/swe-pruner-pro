#!/usr/bin/env bash
# Re-run the data labelling pipeline. Costs Anthropic API credits.
set -euo pipefail
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY in env}"
: "${HF_CACHE:=hf_cache}"
: "${OUT:=data/relabelled}"
mkdir -p "${OUT}"

python -m swe_pruner_pro.data.parse_trajectories \
    --hf-cache "${HF_CACHE}" \
    --output "${OUT}/parsed_steps.jsonl"

python -m swe_pruner_pro.data.submodular_sample sample \
    --input "${OUT}/parsed_steps.jsonl" \
    --output "${OUT}/sampled_50k.jsonl" \
    --target 50000

python -m swe_pruner_pro.data.label_with_claude \
    --input "${OUT}/sampled_50k.jsonl" \
    --output "${OUT}/labelled.jsonl" \
    --concurrency "${LABEL_CONCURRENCY:-16}"
