#!/usr/bin/env bash
# Build a SWE-Pruner-Pro inference image. Dispatches on --variant.
#
# Variants:
#   coder-next-in-engine   (default) Coder-Next backbone, head co-located in SGLang
#   coder-next-off-engine  Coder-Next backbone, head in separate FastAPI process
#   coder-next-ablation    Coder-Next + extra baseline pruners
#   mimo-in-engine         MiMo-V2-Flash backbone, in-engine head
#   mimo-off-engine        MiMo-V2-Flash backbone, off-engine head
#   mimo-ablation          MiMo + extra baseline pruners
#
# Set REGISTRY to your container registry (e.g. ghcr.io/your-org). The
# selected Dockerfile also requires <YOUR_SGLANG_BASE_IMAGE> to be filled in.

set -euo pipefail

VARIANT="coder-next-in-engine"
TAG="${TAG:-latest}"
REGISTRY="${REGISTRY:-<YOUR_REGISTRY>}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$VARIANT" in
    coder-next-in-engine)  DF="docker/Dockerfile.coder-next.in-engine";  NAME="swe-pruner-coder-next" ;;
    coder-next-off-engine) DF="docker/Dockerfile.coder-next.off-engine"; NAME="swe-pruner-coder-next-off-engine" ;;
    coder-next-ablation)   DF="docker/Dockerfile.coder-next.ablation";   NAME="swe-pruner-coder-next-ablation" ;;
    mimo-in-engine)        DF="docker/Dockerfile.mimo.in-engine";        NAME="swe-pruner-mimo" ;;
    mimo-off-engine)       DF="docker/Dockerfile.mimo.off-engine";       NAME="swe-pruner-mimo-off-engine" ;;
    mimo-ablation)         DF="docker/Dockerfile.mimo.ablation";         NAME="swe-pruner-mimo-ablation" ;;
    *) echo "Unknown variant: $VARIANT"; exit 1 ;;
esac

IMAGE="${REGISTRY}/${NAME}:${TAG}"
echo "==> Building ${IMAGE} from ${DF}"
docker build -f "${DF}" -t "${IMAGE}" .

echo "Built ${IMAGE}"
echo "Push with: docker push ${IMAGE}"
