# SWE-Bench Verified eval (mini-swe-agent fork)

This directory contains a fork of
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) with two
additions on top of the upstream code:

1. **`PrunerClient`** (`mini-swe-agent/src/minisweagent/utils/pruner.py`) —
   an HTTP client that calls the pruner server's `/prune` endpoint
   between turns. The agent's `_apply_pruner` (in
   `agents/default.py`) sends the (history, tool_call, tool_response) to
   the pruner and replaces the observation with the pruned version
   before it goes into the next prompt.

2. **`<output_threshold>X</output_threshold>` XML tags** — the agent can
   embed this tag before its bash block to control prune aggressiveness
   per turn (`0.0` = disable pruning for this call, `0.5` = default,
   `1.0` = aggressive).

The rest of the fork is vanilla mini-swe-agent — left intact so that
swe-bench evaluation matches the upstream harness.

## Setup

```bash
cd mini-swe-agent
uv sync
```

## Running

The pruner config in `src/minisweagent/config/extra/swebench.yaml` reads
its parameters from environment variables (`PRUNER_URL`,
`PRUNE_MIN_CHARS`, `PRUNE_THRESHOLD`). Empty `PRUNER_URL` =
baseline mode (no pruning).

### With pruner

```bash
export PRUNER_URL="http://your-pruner-server"
export PRUNE_MIN_CHARS=500
export PRUNE_THRESHOLD=0.5

uv run python -m minisweagent.run.extra.swebench \
  --subset verified --split test \
  -m openrouter/qwen/qwen3-30b-a3b-instruct-2507 \
  --pruner-url "$PRUNER_URL" \
  -w 8 -o results/swebench-pruner
```

### Baseline (no pruning)

```bash
uv run python -m minisweagent.run.extra.swebench \
  --subset verified --split test \
  -m openrouter/qwen/qwen3-30b-a3b-instruct-2507 \
  --disable-pruner \
  -w 8 -o results/swebench-baseline
```

### Server-side ablation backend

The `--ablation-backend` flag tells the pruner server to route the
request to a non-default backend (`llmlingua2`, `longcodezip`,
`selective_context`, `self_prune`, `rerank`, `swe_pruner`). This
requires that the pruner server has the ablation pruner mounted; see
the open-source release's `serving/` for how to bring up such a server.

```bash
uv run python -m minisweagent.run.extra.swebench \
  --subset verified --split test \
  -m openrouter/qwen/qwen3-30b-a3b-instruct-2507 \
  --pruner-url "$PRUNER_URL" --ablation-backend llmlingua2 \
  -w 8 -o results/swebench-llmlingua2
```

## Evaluating predictions

The runner writes a `preds.json` file in the output directory; submit
that to SWE-Bench's evaluation harness (the upstream
[sb-cli](https://github.com/SWE-agent/sb-cli) tool works directly).
