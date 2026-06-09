# Ablation Baselines

Six pruning baselines used in the SWE-Pruner Pro paper:

| Module | Class | Notes |
|---|---|---|
| `llmlingua2` | `LLMLingua2Pruner` | Sequence-tagging compressor; query-agnostic. Needs `llmlingua` + `tiktoken`. |
| `selective_context` | `SelectiveContextPruner` | Self-information based. Needs `selective-context` + `spaCy en_core_web_sm`. |
| `rerank` | `RerankPruner` | bge-reranker-v2-m3 over a 40-line sliding window, top_k=3. Vendored module required (see below). |
| `self_prune` | `SelfPrunePruner` | Asks the agent's own backbone (via OpenAI-compatible HTTP) for line numbers to keep. |
| `longcodezip` | `LongCodeZipPruner` | Function-level perplexity ranking (coarse / `rank_only=True`). Vendored module required. |
| `swe_pruner` | `SWEPrunerBackend` | HTTP shim to the published `github.com/Ayanami1314/swe-pruner` Phase-1 service. |

## Usage

In the open-source release these baselines are invoked **client-side by
the agent runners** (SWE-QA / Oolong / SWE-Bench), not by the pruner
server. They share the `BaselinePruner` protocol defined in
`__init__.py`:

```python
result = baseline.prune(
    history=...,
    tool_call={"name": "bash", "arguments": {...}},
    tool_response="<long output>",
    threshold=0.5,
    query="definition of class FooBar",
)
# result.pruned_code, result.original_chars, result.pruned_chars, ...
```

There is no separate "pruner-server-with-extra-pruners" mode in the
public release — the runners construct the chosen baseline once at
startup and call `.prune(...)` directly between turns.

## Vendored dependencies

`longcodezip` and `rerank` import a `code_compressor` and `reranker`
module respectively. Set the env var

```bash
export SWE_PRUNER_BASELINE_LIB_PATH=/path/to/dir/with/those/modules
```

before running the eval. The expected interfaces:

- `code_compressor.CodeCompressor(model_name, device_map)` with method
  `.compress_code_file(code, query, instruction, rate, language, rank_only)`
  returning `{"compressed_code": str}`. Reference implementation:
  the LongCodeZip codebase.
- `reranker.BGEV2M3Reranker(model_name, device)` with method
  `.rerank(query, docs, batch_size)` returning `[(score, doc), ...]`.

## Model / runtime config

- `LLMLINGUA2`: `model_name` defaults to
  `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`. Pre-cache the
  `cl100k_base` tiktoken encoder in your image to avoid runtime
  download.
- `LONGCODEZIP_MODEL_PATH`: ranking LM, default
  `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
- `SWE_PRUNER_REF_URL`: pre-launched ref-service URL (or use
  `SWE_PRUNER_REF_DIR` + `SWE_PRUNER_REF_MODEL` to subprocess-launch).
