# SGLang Patches for SWE-Pruner-Pro

Target version: **sglang 0.5.10.post1**.

These overlays fix three bugs in SGLang's hidden-state (HS) return path that
make end-to-end HS extraction unreliable for production-scale pruning. They
correspond to the patches described in the paper Appendix
(`app:sglang-patches`, `app:latency-opts`).

## What each patch does

### 1. Batch alignment for mixed HS / non-HS requests
Files: `srt/managers/tokenizer_manager.py`

When a batch contains both HS-requesting and non-HS-requesting requests,
upstream's `output_hidden_states` list is shorter than the batch and indexing
crashes the TokenizerManager (which then takes the whole server down). The
patch preallocates a per-request slot (mirroring how the logprob path already
handles this) and adds a defensive bounds check.

### 2. Chunked-prefill HS accumulation
Files: `srt/managers/scheduler_output_processor_mixin.py`,
`srt/managers/scheduler.py`

With chunked prefill, only the **last** chunk's HS was kept and it was sliced
by `len(origin_input_ids)`, so long prompts silently lost earlier chunks and
got positionally-misaligned HS. The mixin now accumulates per-chunk HS into
the request slot using `extend_input_len_per_req`. The scheduler's existing
logprob snapshot guard is widened to also fire when `return_hidden_states=True`
so the per-chunk HS access is well-defined.

### 3. Prefix-cache exemption for HS region
Files: `srt/managers/schedule_batch.py`, `srt/managers/io_struct.py`

Cached prefix tokens carry KV but no HS, so when an earlier identical-prefix
request populated the radix cache past the region the caller wants HS for,
the second request silently received no HS for that span. The patch threads a
new `hidden_states_start_len` field through `GenerateReqInput`,
`TokenizedGenerateReqInput`, and `Req`, then caps `max_prefix_len` at
`hidden_states_start_len` inside `init_next_round_input`, forcing tokens after
the cap to go through prefill (mirrors `logprob_start_len`).

### 4. Binary HS serialization (latency optimization)
Files: `srt/managers/tokenizer_manager.py`

Replaces the JSON-list HS payload with a binary base64 envelope:

```json
{"__binary__": true, "shape": [...], "dtype": "...", "data": "<base64>"}
```

A 16k x 2048 fp32 HS shrinks from ~1-3 GB of JSON text to ~170 MB of base64.
Optional fp16 downcast halves the wire size again. Clients should fall back
to the legacy list format if the field is a list (rolling deploys are safe).

### 5. (Optional) In-engine pruning head
The patches also support running the pruning head **inside** the SGLang
scheduler when a request includes `run_pruning_head=True`. This avoids
shipping full HS over the wire and returns small per-token logits instead.
This path is internally orchestrated by `swe_pruner_pro.serving.pruner_server`
and is not user-facing.

## How to apply

### Option 1: copy overlay onto an installed sglang

```bash
SGLANG_DIR=$(python -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')
cp -r patches/sglang/srt/* "$SGLANG_DIR/srt/"
```

The overlay is pure-python, no rebuild needed. Verify with:

```bash
python3 -m py_compile "$SGLANG_DIR/srt/managers/scheduler.py" \
                      "$SGLANG_DIR/srt/managers/scheduler_output_processor_mixin.py" \
                      "$SGLANG_DIR/srt/managers/tokenizer_manager.py" \
                      "$SGLANG_DIR/srt/managers/schedule_batch.py" \
                      "$SGLANG_DIR/srt/managers/io_struct.py"
```

### Option 2: build a sglang fork with these files committed and `pip install` it.

## Required serving flags

When launching `python -m sglang.launch_server`, add:

- `--enable-return-hidden-states` (required: enables the HS path the patches fix)

Recommended for stability under prefix cache:

- Keep radix cache **enabled** (default). Patch #3 makes HS robust under cache hits.
- For cases where you do not need HS-region exemption, you can also set
  `--chunked-prefill-size -1` to disable chunked prefill entirely (avoids #2 by
  construction, at a throughput cost).

## Validation

Cosine similarity vs a `transformers` reference HS extraction:

- Median: **0.997**
- Mean:   **0.983**

(Slight deviation comes from kernel-level numerical differences, not from
truncation or misalignment, which the patches eliminate.)
