#!/usr/bin/env bash
# Build the pruner image. Set YOUR_REGISTRY to a registry you can push to.
set -euo pipefail
: "${YOUR_REGISTRY:=your-registry.example.com/swe-pruner}"
TAG="${1:-latest}"
docker build -f docker/Dockerfile.pruner -t "${YOUR_REGISTRY}:${TAG}" .
echo "Built ${YOUR_REGISTRY}:${TAG}"
