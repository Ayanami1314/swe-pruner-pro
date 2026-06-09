#!/bin/bash
# Build & push the MiMo+ablation image. Built on top of the production MiMo
# image; run docker/deploy_mimo.sh first.

set -euo pipefail

REGISTRY="${REGISTRY:-<YOUR_REGISTRY>}"
IMAGE="${REGISTRY}/swe-pruner-mimo-ablation"
TAG="${1:-latest}"
FULL="${IMAGE}:${TAG}"

ASSETS_DIR="${ABLATION_ASSETS_DIR:-ablation_assets}"
for d in "${ASSETS_DIR}/wheels" "${ASSETS_DIR}/tiktoken-cache" "${ASSETS_DIR}/swe-pruner-ref"; do
    if [ ! -d "$d" ] || [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
        echo "ERROR: $d/ is missing or empty."
        exit 1
    fi
done

echo "==> Building ${FULL} ..."
docker build -f docker/Dockerfile.mimo.ablation -t "${FULL}" .

echo "==> Pushing ${FULL} ..."
docker push "${FULL}"

echo "==> Done: ${FULL}"
