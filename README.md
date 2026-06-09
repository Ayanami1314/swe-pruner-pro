# SWE-Pruner Pro
## 1. Repository layout, environment, container prep

### Layout

```
src/swe_pruner_pro/
  model/        Pruning head + length-aware embedding
  data/         Trajectory parsing, diversity sampling, Claude labelling,
                hidden-state extraction, packing into memmap shards
  train/        From-features head training (frozen backbone)
  serving/      FastAPI pruner server (talks to a patched SGLang)
  eval/
    sweqa/      SWE-QA / SWE-QA-Pro multi-turn agent + judge
    oolong/     Multi-turn re-cast of Oolong
    swebench/   mini-swe-agent fork with pruner integration
    baselines/  LLMLingua2, Selective Context, RAG, Self-Prune,
                LongCodeZip, SWE-Pruner
  prompts/      Labelling + judge prompts

patches/sglang/ Overlay against sglang 0.5.10.post1 (3 bug fixes
                + binary HS envelope + optional in-engine head hook)
docker/         Six Dockerfiles (Coder-Next / MiMo × in-engine /
                off-engine / ablation) + build & deploy scripts
scripts/        Reproduction shell scripts
data/           Labelled training corpus (22,609 samples) + case bundles
utils/          Figure / stats / motivation-probe / latency scripts
```

### Python environment

```bash
pip install -e .                              # core
pip install -e ".[train,eval,baselines]"     # add training + eval extras
```

Python 3.12, PyTorch 2.9, CUDA 12.x. Multi-GPU training uses `torchrun`; eval
needs HTTP access to a running pruner server.

### Container build (only needed for inference / production serving)

The pruner server is built on top of SGLang. We ship six Dockerfiles under
`docker/`; pick the one matching `<backbone> × <head placement>`:

| Variant                              | Backbone        | Head        |
|--------------------------------------|-----------------|-------------|
| `Dockerfile.coder-next.in-engine`    | Qwen3-Coder-Next| in-engine   |
| `Dockerfile.coder-next.off-engine`   | Qwen3-Coder-Next| off-engine  |
| `Dockerfile.coder-next.ablation`     | Qwen3-Coder-Next| off + 6 baselines |
| `Dockerfile.mimo.in-engine`          | MiMo-V2-Flash   | in-engine   |
| `Dockerfile.mimo.off-engine`         | MiMo-V2-Flash   | off-engine  |
| `Dockerfile.mimo.ablation`           | MiMo-V2-Flash   | off + 6 baselines |

Before building, fill these placeholders in the chosen Dockerfile and any
`deploy*.sh` you use:

| Placeholder                | What to put |
|----------------------------|-------------|
| `<YOUR_SGLANG_BASE_IMAGE>` | base sglang image (we used `lmsys/sglang:0.5.8.post1`; the layer upgrades to 0.5.10.post1 and applies `patches/sglang/`) |
| `<YOUR_REGISTRY>`          | container registry you can push to |
| `<PRESET_MODELS_DIR>`      | mount point for backbone weights inside the container (was `/preset-models`) |
| `<HF_CACHE_DIR>`           | pre-populated HF cache mount (ablation images only) |
| `<YOUR_ABLATION_ASSETS>`   | directory with pre-downloaded wheels for the baselines (ablation images only — see `docker/README.md` for the wheel list) |

Then:

```bash
./docker/build.sh --variant coder-next-in-engine --tag v1
# or the full matrix:
for v in coder-next-{in-engine,off-engine,ablation} \
         mimo-{in-engine,off-engine,ablation}; do
    ./docker/build.sh --variant "$v"
done
```

The SGLang overlay is pure-Python; for a non-Docker workflow just copy it on
top of an installed sglang — see `patches/sglang/README.md`.

---

## 2. Training

### What to edit

`scripts/train_coder_next.sh` and `scripts/train_mimo.sh` are the two recipes
that reproduce the paper checkpoints. Only the four env vars at the top change:

| Var            | Default                                | Override when |
|----------------|----------------------------------------|---------------|
| `FEATURES_DIR` | `features/{coder_next,mimo_v2_flash}/packed` | You extracted features somewhere else. |
| `EVAL_DATA`    | `data/training_corpus_22k.jsonl`        | You want to eval on a different split. |
| `LOG_DIR`      | `logs/...`                      | Naming a new run. |
| `NPROC`        | `8`                                     | Different GPU count. |

The hyperparameters (10 epochs, lr 3e-5 → 1.5e-5 cosine, dropout 0.4,
per-sample balanced focal γ=2, length-aware embedding on) are the
paper-recommended setting; only change them through CLI flags to
`swe_pruner_pro.train.train`.

### How to run

```bash
# 1) (one-time per backbone) extract hidden-state features from the corpus
#    Requires the patched SGLang running locally.
BACKBONE=<path-or-hf-id> TP_SIZE=8 \
    bash scripts/extract_features.sh
# -> features/run0/packed/

# 2) Train the head (5 min on 8×H200)
FEATURES_DIR=features/run0/packed \
    bash scripts/train_coder_next.sh   # or train_mimo.sh
# -> logs/0507b_size_emb_coder_next_22k/best_model.pt + model_config.json
```

(Optional) Re-build the corpus from scratch with `scripts/label_data.sh` —
this calls Claude Sonnet 4.6 and costs Anthropic API credits.

---

## 3. Inference + downstream evals

### What to edit

All eval entry points are HTTP clients; they need three URLs / IDs:

| Env var        | What it is |
|----------------|------------|
| `PRUNER_URL`   | The pruner FastAPI server (one of the Docker images above, or `swe-pruner-server …` run manually). |
| `BASE_URL`     | OpenAI-compatible endpoint serving the agent backbone. In the bundled Docker images this is `${PRUNER_URL}/model-raw/v1`. |
| `MODEL` / `MODEL_CODER_NEXT` / `MODEL_MIMO` | Model id as advertised by `BASE_URL`. |

If running the pruner server outside the Docker image, point it at your head
checkpoint:

```bash
swe-pruner-server \
    --backbone "$BACKBONE_PATH" \
    --checkpoint <your_head_ckpt_dir> \
    --sglang-url "$SGLANG_URL"
```

### How to run

```bash
# SWE-QA / SWE-QA-Pro / Oolong (all use the same eval entrypoint)
PRUNER_URL=… BASE_URL=… MODEL=… bash scripts/run_eval.sh sweqa
PRUNER_URL=… BASE_URL=… MODEL=… bash scripts/run_eval.sh sweqa-pro
PRUNER_URL=… BASE_URL=… MODEL=… bash scripts/run_eval.sh oolong

# SWE-Bench Verified — reproduces the paper sweep (2 backbones × 5 prune configs)
PRUNER_URL=… BASE_URL=… \
MODEL_CODER_NEXT=openai/qwen/qwen3-coder-next \
MODEL_MIMO=openai/xiaomi/mimo-v2-flash \
    bash scripts/run_swebench.sh
# Predictions are written as preds.json per setting; score with sb-cli:
for d in results/swebench/*/preds.json; do
    name=$(basename $(dirname $d))
    uv run sb-cli submit swe-bench_verified test \
        --predictions_path "$d" --run_id "$name"
done
```

The judge step (LLM-as-judge for SWE-QA / SWE-QA-Pro) is invoked inside
`agent_eval.py` and reads `OPENAI_API_KEY` for the judge API.
