#!/bin/bash
# Build & push the Coder-Next ablation image (extra baseline pruners). Built
# on top of the production Coder-Next image; run docker/deploy.sh first.
#
# Usage:
#   REGISTRY=<YOUR_REGISTRY> ./docker/deploy_ablation.sh             # tag = latest
#   REGISTRY=<YOUR_REGISTRY> ./docker/deploy_ablation.sh v1
#
# At deploy time, set EXTRA_PRUNERS to a comma list of backends to load
# (e.g. llmlingua2,longcodezip,bge-reranker,selective-context,self-prune,
# swe-pruner-ref).

set -euo pipefail

REGISTRY="${REGISTRY:-<YOUR_REGISTRY>}"
IMAGE="${REGISTRY}/swe-pruner-coder-next-ablation"
TAG="${1:-latest}"
FULL="${IMAGE}:${TAG}"

# Pre-flight: ablation assets must exist locally — the build runs offline.
# (HF model snapshots are mounted at runtime via HF_HOME, not bundled.)
ASSETS_DIR="${ABLATION_ASSETS_DIR:-ablation_assets}"
for d in "${ASSETS_DIR}/wheels" "${ASSETS_DIR}/tiktoken-cache" "${ASSETS_DIR}/swe-pruner-ref"; do
    if [ ! -d "$d" ] || [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
        echo "ERROR: $d/ is missing or empty."
        echo "Populate ${ASSETS_DIR}/ with wheels/, tiktoken-cache/, swe-pruner-ref/, baselines/, single_turn/."
        exit 1
    fi
done

echo "==> Building ${FULL} ..."
docker build -f docker/Dockerfile.coder-next.ablation -t "${FULL}" .

echo "==> Pushing ${FULL} ..."
docker push "${FULL}"

echo "==> Done: ${FULL}"
