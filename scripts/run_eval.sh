#!/usr/bin/env bash
# Run the SWE-QA / SWE-QA-Pro / Oolong / SWE-Bench evals against a running
# pruner server. Set PRUNER_URL, BASE_URL (model OpenAI endpoint), and MODEL.
set -euo pipefail
: "${PRUNER_URL:?Set PRUNER_URL to the running pruner server}"
: "${BASE_URL:?Set BASE_URL to the model OpenAI-compatible endpoint}"
: "${MODEL:?Set MODEL to the served model id}"
: "${OUT:=results}"
mkdir -p "${OUT}"

case "${1:-all}" in
  sweqa|all)
    python -m swe_pruner_pro.eval.sweqa.agent_eval run \
      --variant sweqa --base-url "${BASE_URL}" --model "${MODEL}" \
      --pruner-url "${PRUNER_URL}" --output "${OUT}/sweqa"
    ;;& # fall through if "all"
  sweqa-pro|all)
    python -m swe_pruner_pro.eval.sweqa.agent_eval run \
      --variant sweqa-pro --base-url "${BASE_URL}" --model "${MODEL}" \
      --pruner-url "${PRUNER_URL}" --output "${OUT}/sweqa-pro"
    ;;&
  oolong|all)
    python -m swe_pruner_pro.eval.oolong.agent_eval run \
      --base-url "${BASE_URL}" --model "${MODEL}" \
      --pruner-url "${PRUNER_URL}" --output "${OUT}/oolong"
    ;;&
  swebench|all)
    # SWE-Bench has 10 settings (2 backbones x 5 prune configs) — see
    # scripts/run_swebench.sh. This branch only runs ours/approach for one
    # backbone (set MODEL_CODER_NEXT or MODEL_MIMO to override paper sweep).
    BASE_URL="${BASE_URL}" PRUNER_URL="${PRUNER_URL}" \
    MODEL_CODER_NEXT="${MODEL}" MODEL_MIMO="${MODEL}" \
    OUT="${OUT}/swebench" ONLY="coder-next:approach" \
    bash "$(dirname "$0")/run_swebench.sh"
    ;;
esac
