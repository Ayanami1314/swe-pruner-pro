#!/usr/bin/env bash
# Reproduce the SWE-Bench Verified table in the paper (Tab. tab:swebench).
# Runs 10 settings: 2 backbones x {Baseline, Approach (ours), LongCodeZip, ReRank, SWE-Pruner}.
#
# Each setting writes a `preds.json` consumable by `sb-cli submit swe-bench_verified test`.
#
# Required env:
#   BASE_URL          OpenAI-compatible endpoint serving the agent backbone
#                     (the patched SGLang from patches/sglang/, i.e. the same
#                     pod as PRUNER_URL with /model-raw/v1 suffix in our setup)
#   PRUNER_URL        Pruner server (FastAPI from src/swe_pruner_pro/serve)
#   MODEL_CODER_NEXT  Model id for Qwen3-Coder-Next as served by BASE_URL
#                     (e.g. "openai/qwen/qwen3-coder-next")
#   MODEL_MIMO        Model id for MiMo-V2-Flash as served by BASE_URL
#                     (e.g. "openai/xiaomi/mimo-v2-flash")
# Optional:
#   OUT               Output root (default: results/swebench)
#   WORKERS           Concurrent instances per setting (default: 8)
#   PRUNE_THRESHOLD   Default 0.5 (read by mini.yaml/swebench.yaml via env)
#   PRUNE_MIN_CHARS   Default 500
#   ONLY              Comma-list to restrict settings, e.g. "mimo:approach,coder-next:baseline"
#
# The ablation backends (longcodezip/rerank/swe_pruner) are routed by the same
# pruner server via the `--ablation-backend` flag — make sure the server was
# launched with the matching `--extra-pruners` config.
set -euo pipefail

: "${BASE_URL:?Set BASE_URL to the agent OpenAI-compatible endpoint}"
: "${PRUNER_URL:?Set PRUNER_URL to the pruner server}"
: "${MODEL_CODER_NEXT:?Set MODEL_CODER_NEXT to the served model id (Qwen3-Coder-Next)}"
: "${MODEL_MIMO:?Set MODEL_MIMO to the served model id (MiMo-V2-Flash)}"
: "${OUT:=results/swebench}"
: "${WORKERS:=8}"
export PRUNE_THRESHOLD="${PRUNE_THRESHOLD:-0.5}"
# Paper uses min_chars=2000 for the "ours" rows (jobs-coder-next-pruner /
# jobs-mimo-pruner) and 500 for the three ablation rows. Baselines
# (--disable-pruner) ignore it. Per-row override below.
export PRUNE_MIN_CHARS="${PRUNE_MIN_CHARS:-500}"
APPROACH_MIN_CHARS="${APPROACH_MIN_CHARS:-2000}"
export OPENAI_API_BASE="${BASE_URL}"
export OPENAI_BASE_URL="${BASE_URL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

mkdir -p "${OUT}"
RUNNER_DIR="$(dirname "$0")/../src/swe_pruner_pro/eval/swebench/mini-swe-agent"

# tag                         backbone          model_var         flags
# Baselines use the plain mini-swe-agent system prompt (extra/swebench-baseline.yaml,
# matching internal `mini-swe-agent-baseline`). All pruner/ablation runs use the
# default extra/swebench.yaml — same prompt the pruner agent saw during the paper
# experiments (matches internal `mini-swe-agent-pruner`). PRUNE_MIN_CHARS=500 is
# the value used by all paper rows (mc500 / 0521 / ablation yamls); older 2000
# configs were superseded.
BASELINE_CFG="src/minisweagent/config/extra/swebench-baseline.yaml"
SETTINGS=(
  "coder-next:baseline        coder-next        MODEL_CODER_NEXT  --disable-pruner --config ${BASELINE_CFG}"
  "coder-next:approach        coder-next        MODEL_CODER_NEXT"
  "coder-next:longcodezip     coder-next        MODEL_CODER_NEXT  --ablation-backend longcodezip"
  "coder-next:rerank          coder-next        MODEL_CODER_NEXT  --ablation-backend rerank"
  "coder-next:swe_pruner      coder-next        MODEL_CODER_NEXT  --ablation-backend swe_pruner"
  "mimo:baseline              mimo              MODEL_MIMO        --disable-pruner --config ${BASELINE_CFG}"
  "mimo:approach              mimo              MODEL_MIMO"
  "mimo:longcodezip           mimo              MODEL_MIMO        --ablation-backend longcodezip"
  "mimo:rerank                mimo              MODEL_MIMO        --ablation-backend rerank"
  "mimo:swe_pruner            mimo              MODEL_MIMO        --ablation-backend swe_pruner"
)

want_run() {
  local tag="$1"
  [[ -z "${ONLY:-}" ]] && return 0
  IFS=',' read -ra wanted <<<"${ONLY}"
  for w in "${wanted[@]}"; do [[ "${w// /}" == "${tag}" ]] && return 0; done
  return 1
}

cd "${RUNNER_DIR}"
for row in "${SETTINGS[@]}"; do
  read -r tag backbone model_var rest <<<"${row}"
  want_run "${tag}" || { echo "[skip] ${tag}"; continue; }
  model="${!model_var}"
  out_dir="${OLDPWD}/${OUT}/${tag//:/__}"
  # Approach rows match the paper's `jobs-{coder-next,mimo}-pruner` runs (min_chars=2000).
  if [[ "${tag}" == *":approach" ]]; then
    row_min_chars="${APPROACH_MIN_CHARS}"
  else
    row_min_chars="${PRUNE_MIN_CHARS}"
  fi
  echo "[run]  ${tag}  model=${model}  min_chars=${row_min_chars}  -> ${out_dir}"
  PRUNE_MIN_CHARS="${row_min_chars}" \
  MSWEA_SILENT_STARTUP=1 \
  python -m minisweagent.run.extra.swebench \
    --subset verified --split test \
    -m "${model}" \
    --pruner-url "${PRUNER_URL}" \
    -w "${WORKERS}" \
    -o "${out_dir}" \
    ${rest:-}
done
cd - >/dev/null

echo
echo "All runs done. Predictions are at:"
for row in "${SETTINGS[@]}"; do
  read -r tag _ _ _ <<<"${row}"
  want_run "${tag}" || continue
  echo "  ${OUT}/${tag//:/__}/preds.json"
done
echo
echo "To score with sb-cli:"
echo "  for d in ${OUT}/*/preds.json; do"
echo "    name=\$(basename \$(dirname \$d))"
echo "    uv run sb-cli submit swe-bench_verified test --predictions_path \"\$d\" --run_id \"\$name\""
echo "  done"
