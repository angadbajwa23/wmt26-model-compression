# awq

AWQ INT4 submission for WMT26 Model Compression — constrained track.

- **Base model**: `google/gemma-3-12b-it`
- **Quantization**: AWQ activation-aware scaling + INT4 `W4A16_ASYM` rounding, group_size=128, compressed-tensors format
- **Inference engine**: vLLM with `quantization="compressed-tensors"` 
- **Compression library**: [llm-compressor](https://github.com/vllm-project/llm-compressor) via `AWQModifier` + `QuantizationModifier`
  (legacy [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) is deprecated; llm-compressor is its vLLM-project successor)
- **Model artifact**: [Angad23/gemma-3-12b-it-awq-int4-wmt26](https://huggingface.co/Angad23/gemma-3-12b-it-awq-int4-wmt26)
  — `setup.sh` downloads this automatically into `workdir/model` (override with `HF_MODEL_REPO` / `MODEL_DIR`)

## How AWQ works

AWQ (Activation-aware Weight Quantization) finds the small number of weight
channels whose output activations have the largest magnitude ("salient" channels)
and scales them up before INT4 rounding, then scales them back down after
dequantization.  This is mathematically equivalent to standard dequantization,
but because the rounding error is divided by a large scale factor, the effective
quantization noise on salient channels is much smaller.

Per module, llm-compressor's `AWQModifier` picks the scale via a small grid
search rather than a closed form: for 20 candidate ratios `r ∈ {0/20, ...,
19/20}`, it derives a per-channel scale from calibration activation/weight
statistics, pseudo-quantizes the weights at that scale, replays the cached
calibration activations through the module, and keeps whichever ratio
minimizes `‖Q(W·s)(s⁻¹·X) − W·X‖` (reconstruction error against the original
BF16 output). Still no gradient computation, but it's ~20 forward passes per
module, not one.

> **Note:** `prepare_model.py` currently calls `AWQModifier(duo_scaling="both",
> ...)`. `duo_scaling` is a `bool` field (`True` = derive the scale from both
> activation *and* weight magnitude statistics; `False` = activation magnitude
> only) — `"both"` is not a valid value and raises a `pydantic.ValidationError`
> (confirmed directly against the pinned `llmcompressor==0.7.1.3` /
> `pydantic==2.14.0a1` combination in `.venv-compress`). The uploaded model
> artifact predates this — it was almost certainly built before an unpinned
> `pydantic` drifted forward on a later `uv pip install --prerelease=allow`.
> Re-running `compress.sh` from a clean `workdir/model` today will crash on
> this line; the fix is `duo_scaling=True`, which is what `"both"` was
> presumably intended to mean.

Model size after compression: ~8 GB (down from ~23 GB in BF16).

## Setup

```bash
bash setup.sh
```

Installs runtime and quantization dependencies plus the organizer `modelzip`
package into `./.venv`.

## Compress (offline, run once — optional, not needed to run inference)

`setup.sh` already downloads the finished quantized model from Hugging Face
(see **Model artifact** above), so nothing below is required to run this
submission. It's documented here for reproducibility of how that artifact was
built.

```bash
bash compress.sh
```

`compress.sh` calls `prepare_model.py`, which does two things:

### 1. Calibration data

Same pipeline and pool as [`gptq`](../gptq/README.md#1-calibration-data):
`data/calibration/<pair>/calibration.jsonl`, pre-built by
[`data_prep/build_calibration.py`](../../data_prep/build_calibration.py) from
WMT25 dev paragraphs plus public News Commentary/OPUS TED talks/QED corpora,
deduped, length-filtered, and stratified by sentence length into 2-turn
Gemma3 chat records (`user`: translate prompt, `assistant`: reference
translation).

`compress.sh` passes `CALIB_SAMPLES_PER_PAIR=64` (same as `gptq`), so
`prepare_model.py` takes an evenly-strided 64 examples per language pair —
192 total — applies the Gemma3 chat template with
`add_generation_prompt=False` (the quantizer needs the full prompt+response
token sequence), and tokenizes with a 512-token truncation cap.
(`prepare_model.py`'s own standalone default is 512 examples/pair —
`compress.sh` overrides it down to 64 to match `gptq`/`comet-mixed-precision`.)

### 2. Quantization

A two-stage recipe via llm-compressor's `oneshot()`:

1. **`AWQModifier`** — activation-aware scaling. Finds the small set of
   weight channels whose output activations have the largest magnitude
   ("salient" channels) and scales them up before rounding, then back down
   after dequantization, so the effective rounding error on salient channels
   shrinks. Mappings (`smooth_layer` → `[balance_layers]`, e.g.
   `input_layernorm` → `[q_proj, k_proj, v_proj]`) are built directly from
   the loaded model's module names (`_build_awq_mappings`) instead of
   llm-compressor's default regex patterns — `Gemma3ForConditionalGeneration`'s
   vision tower has its own `q/k/v_proj` modules ahead of the language-model
   layers in module iteration order, which corrupts regex-based grouping
   (AWQ requires exactly one `input_layernorm` per mapping; a global regex
   would dump all 48 decoder layers into a single group). Offloaded to CPU
   between layers; see the `duo_scaling` caveat under **How AWQ works** above.
2. **`QuantizationModifier`** — applies the actual INT4 `W4A16_ASYM` rounding
   to all `Linear` layers post-scaling, group_size=128, excluding `lm_head`
   and the vision tower (text-only MT task; the vision tower's channel dims
   also aren't divisible by 128).

The result is saved in compressed-tensors format (`save_compressed=True`);
tokenizer/processor files are copied straight from the source checkpoint
rather than re-saved, same reason as `gptq` — the calibration tokenizer's
truncation state gets mutated during tokenization and would otherwise bake a
silent 512-token cap into the saved `tokenizer.json`.

Override defaults via environment variables:

| Variable                | Default                          | Description                                                    |
|-------------------------|-----------------------------------|------------------------------------------------------------------|
| `MODEL_ID`               | `google/gemma-3-12b-it`          | Base model repo ID or local path                                |
| `MODEL_CACHE`            | `/mnt/tg/.../models`             | Cache directory for downloaded models                           |
| `MODEL_DIR`              | `workdir/model`                  | Output path for the quantized model                             |
| `CALIB_DIR`              | `../../data/calibration`         | Directory with `<pair>/calibration.jsonl` files                 |
| `CALIB_SAMPLES_PER_PAIR` | `64`                              | Calibration examples per language pair (192 total across 3 pairs)|

## Run

```bash
bash run.sh --lang-pair ces-deu --input input.txt --output output.txt
```

Supported language pairs: `ces-deu`, `eng-zho_Hans`, `eng-ara_EG`

Set `MODEL_DIR` to point at a pre-quantized model artifact if it lives outside
the default `workdir/model` path.
