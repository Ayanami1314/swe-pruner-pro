# Docker images for SWE-Pruner-Pro inference

This directory ships six Dockerfiles covering the two production backbones
(Coder-Next, MiMo-V2-Flash) crossed with three deployment modes (in-engine head,
off-engine head, ablation with extra baseline pruners).

| Dockerfile | Backbone | Head placement | Extras |
|---|---|---|---|
| `Dockerfile.coder-next.in-engine` | Qwen3-Coder-Next | embedded in SGLang scheduler (saves ~70 MB/req of HS over the wire) | — |
| `Dockerfile.coder-next.off-engine` | Qwen3-Coder-Next | separate FastAPI process; SGLang ships full HS | — |
| `Dockerfile.coder-next.ablation` | Qwen3-Coder-Next | off-engine head + ablation baselines | LLMLingua-2, SelectiveContext, LongCodeZip, BGE-Reranker-v2-m3, self-prune, swe-pruner reference |
| `Dockerfile.mimo.in-engine` | MiMo-V2-Flash | in-engine head, TP=8 DP=2 dp-attention | — |
| `Dockerfile.mimo.off-engine` | MiMo-V2-Flash | off-engine head | — |
| `Dockerfile.mimo.ablation` | MiMo-V2-Flash | off-engine head + ablation baselines | same baselines as Coder-Next ablation |

All six images apply the same SGLang 0.5.10.post1 patches under `patches/sglang/`:
Bug #1 (mixed-batch IndexError), Bug #2 (chunked-prefill HS truncation), and
Bug #3 (HS prefix-cache cap). See `patches/sglang/README.md` for the full bug
report. The flashinfer pin (`==0.6.7.post3`) is required by sglang 0.5.10.

## Placeholders to fill in

Each Dockerfile starts with a header listing the placeholders to set:

- `<YOUR_SGLANG_BASE_IMAGE>` — base SGLang image (sglang 0.5.8 or 0.5.10).
- `<YOUR_REGISTRY>` — your container registry prefix.
- `<PRESET_MODELS_DIR>` — mount point for backbone weights (originally `/preset-models`).
- `<HF_CACHE_DIR>` — pre-populated HuggingFace cache for the ablation images.
- `<YOUR_ABLATION_ASSETS>` — only used by ablation images; directory containing
  pre-downloaded pip wheels, tiktoken cache, swe-pruner-ref source, baselines/.

## Building

```bash
# Single image
./docker/build.sh --variant coder-next-in-engine --tag v1

# Full build matrix
for v in coder-next-in-engine coder-next-off-engine coder-next-ablation \
         mimo-in-engine mimo-off-engine mimo-ablation; do
    ./docker/build.sh --variant "$v"
done
```

Per-variant deploy scripts (`deploy.sh`, `deploy_mimo.sh`, `deploy_ablation.sh`,
`deploy_mimo_ablation.sh`, `deploy_qwen.sh`) build + push in one shot.

## Runtime environment variables

Required:
- `PRUNER_BACKBONE` — backbone model path (mounted into the container).
- `PRUNER_CHECKPOINT` — pruning-head checkpoint directory.

Optional:
- `TP_SIZE` — tensor-parallel size (default 8 on Coder-Next; MiMo is fixed at TP=8 DP=2).
- `SGLANG_PORT`, `PRUNER_PORT` — internal ports (default 30000/9001 + 8001).
- `PRUNER_DEVICE`, `PRUNER_HIDDEN_SIZE`, `PRUNER_MAX_LENGTH`.
- `PRUNER_HEAD_CKPT` — set on in-engine images to enable embedded-head mode
  (auto-sets `PRUNER_EMBEDDED_HEAD=1`).
- `EXTRA_PRUNERS` (ablation only) — comma list of baseline backends.
- `HF_HOME` (ablation only) — pre-populated HF cache mount.

External access (via the bundled nginx on port 80):
- `/model-raw/*` → SGLang
- `/prune-server/*` → pruner FastAPI
- `/*` → pruner FastAPI (default)

## Pinned dependency versions

The Dockerfiles assume the base SGLang image already ships PyTorch, transformers,
numpy, click, ninja, packaging, requests, tabulate, tqdm, nvidia-ml-py, and
sglang 0.5.8.post1 (or 0.5.10) with sgl-kernel — that's what `lmsys/sglang:0.5.8.post1`
gives you. On top of that the Dockerfiles install:

### All images (in-engine, off-engine, ablation)

```
fastapi>=0.115.0
uvicorn>=0.34.0
typer>=0.15.0
pydantic>=2.0.0
rich
tqdm
```

```
sglang==0.5.10.post1            # forced upgrade from 0.5.8.post1 base
sglang-kernel==0.4.1.post1      # renamed from sgl-kernel in 0.5.10
flashinfer-python==0.6.7.post3  # 0.5.10 hard-asserts >=0.6.7.post3 at startup
```

flashinfer ships three coupled wheels that all pin to the same version
(`flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache`). For an
offline build, place all three under `wheels/flashinfer/` and add
`--find-links=wheels/flashinfer` to the install command. The cu129 variant of
`flashinfer-jit-cache` (1.8 GB) is only published on
`https://flashinfer.ai/whl/cu129`, not on standard PyPI mirrors.

The `swe-pruner-pro` package itself (this repo's `pyproject.toml`) is
installed from source in the same layer:

```bash
pip install --no-cache-dir /home/work/swe_pruner_pro/
```

### Ablation images only (`*.ablation`)

These add the seven baseline pruners from §4 of the paper. All wheels live under
`<YOUR_ABLATION_ASSETS>/wheels/` and are installed via `--no-index --find-links`.

```
# Pruner baselines (installed with --no-deps to avoid resolving torch metadata)
llmlingua>=0.2.2
rank-bm25>=0.2.2
FlagEmbedding>=1.4.0          # BGE-Reranker-v2-m3 backend
selective-context>=0.1.4

# Support packages (regular dep resolution; pure-Python deps)
spacy>=3.7,<4.0
nltk>=3.8
tiktoken>=0.7
accelerate>=1.0               # required by transformers 5.x for device_map=...

# Required by swe-pruner reference (model_structure.py hard-codes
# attn_implementation="flash_attention_2") — install via the exact wheel path,
# not "==2.8.3", because the wheel's local-version tag is 2.8.3+cu12torch2.9
# and pip's resolver matches it inconsistently across versions.
flash_attn-2.8.3+cu12torch2.9-cp312-cp312-linux_x86_64.whl

# spaCy English model (downloaded once via download_ablation_assets.sh)
en_core_web_sm==3.7.1
```

### Building the wheel cache for an offline ablation build

```bash
mkdir -p assets/wheels assets/wheels/spacy-model
pip download --no-cache-dir --dest assets/wheels \
    "llmlingua>=0.2.2" "rank-bm25>=0.2.2" \
    "spacy>=3.7,<4.0" "nltk>=3.8" "tiktoken>=0.7" \
    "accelerate>=1.0" "FlagEmbedding>=1.4.0" "selective-context>=0.1.4"

# flash_attn — fetch the cu12torch2.9 cp312 prebuilt wheel directly from
# the release (pip download cannot resolve the local-version tag without help).
curl -L -o assets/wheels/flash_attn-2.8.3+cu12torch2.9-cp312-cp312-linux_x86_64.whl \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9-cp312-cp312-linux_x86_64.whl

# spaCy model wheel
python -m spacy download en_core_web_sm --direct
mv $(python -c 'import en_core_web_sm,os;print(os.path.dirname(en_core_web_sm.__file__))') \
    assets/wheels/spacy-model/   # or just download the wheel from the spaCy release page
```

Then point `<YOUR_ABLATION_ASSETS>` to `assets/` in the Dockerfile.
