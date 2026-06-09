#!/bin/bash
# Build & push the MiMo-V2-Flash pruner image (in-engine head variant).
set -euo pipefail

REGISTRY="${REGISTRY:-<YOUR_REGISTRY>}"
IMAGE="${REGISTRY}/swe-pruner-mimo"
TAG="${1:-latest}"
FULL="${IMAGE}:${TAG}"

echo "==> Building ${FULL} ..."
docker build -f docker/Dockerfile.mimo.in-engine -t "${FULL}" .

echo "==> Pushing ${FULL} ..."
docker push "${FULL}"

echo "==> Done: ${FULL}"
