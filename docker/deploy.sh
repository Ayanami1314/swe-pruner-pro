#!/bin/bash
# Build & push the Coder-Next pruner image (in-engine head variant).
set -euo pipefail

REGISTRY="${REGISTRY:-<YOUR_REGISTRY>}"
IMAGE="${REGISTRY}/swe-pruner-coder-next"
TAG="${1:-latest}"
FULL="${IMAGE}:${TAG}"

echo "==> Building ${FULL} ..."
docker build -f docker/Dockerfile.coder-next.in-engine -t "${FULL}" .

echo "==> Pushing ${FULL} ..."
docker push "${FULL}"

echo "==> Done: ${FULL}"
