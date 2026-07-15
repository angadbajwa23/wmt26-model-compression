# gptq

GPTQ INT4 submission for WMT26 Model Compression — constrained track.

- **Base model**: `google/gemma-3-12b-it`
- **Quantization**: GPTQ INT4 (W4A16), group_size=128, sym=True, desc_act=False
- **Inference engine**: vLLM with `quantization="compressed-tensors"` (vLLM auto-routes W4A16 compressed-tensors checkpoints to the Marlin kernel on H100)
- **Compression library**: [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) ≥ 0.7.1
- **Model artifact**: [Angad23/gemma-3-12b-it-gptq-int4-wmt26](https://huggingface.co/Angad23/gemma-3-12b-it-gptq-int4-wmt26)
  — `setup.sh` downloads this automatically into `workdir/model` (override with `HF_MODEL_REPO` / `MODEL_DIR`)

## How GPTQ works

GPTQ (Generalized Post-Training Quantization) quantizes each weight matrix
column-by-column using the Hessian of the layer's reconstruction loss.  After
rounding weight column *i* to INT4, it measures the rounding error and
propagates it to all remaining columns *j > i* via:

```
W[:, j] -= error_i × H_inv[i, j] / H_inv[i, i]
```

This means each already-quantized column actively corrects the weights that
follow it, so the overall layer output changes as little as possible.  The
Hessian inverse is computed once via Cholesky decomposition and reused for all
columns, making the procedure tractable even for large matrices.

`desc_act=False` disables activation-magnitude column reordering, which is
required by vLLM's `gptq_marlin` Marlin-kernel path.  The Marlin kernel fuses
INT4 dequantization with the matrix multiplication into a single CUDA kernel,
delivering roughly 2-3× higher throughput on H100 compared to the generic
GPTQ dequantize-then-multiply path.

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

Calibration examples come from `data/calibration/<pair>/calibration.jsonl`
(one file per language pair), pre-built by
[`data_prep/build_calibration.py`](../../data_prep/build_calibration.py):

- **Sources per pair**: WMT25 dev paragraphs (`data/wmt25/*.paragraphs.jsonl`
  — priority, bypass the length filter below) plus public WMT parallel
  corpora — News Commentary and OPUS TED talks/QED — fetched via
  `data_prep/download_wmt26.sh`.
- **Filtering**: exact dedup on normalized source text, then a 10–120 word
  length filter on the training-corpus examples (wmt25 paragraphs are exempt).
- **Stratified sampling**: examples are bucketed by source length (short
  10–30w / medium 30–60w / long 60w+) and drawn evenly across buckets, so
  calibration isn't dominated by one sentence length.
- **Format**: each example becomes a 2-turn Gemma3 chat record —
  `user: "Translate from {src} to {tgt}: {source}"` /
  `assistant: "{reference translation}"` — written to
  `data/calibration/<pair>/calibration.jsonl` (256 examples/pair by default).

`prepare_model.py` then takes an evenly-strided subset of that pool —
`CALIB_SAMPLES_PER_PAIR` (default 64) examples per language pair, 192 total
across the 3 pairs — applies the Gemma3 chat template to each with
`add_generation_prompt=False` (GPTQ needs the full prompt+response token
sequence, not just the prompt), and tokenizes with a 512-token truncation cap.

### 2. Quantization

The tokenized calibration set is fed to llm-compressor's `oneshot()` with
`GPTQModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head",
"re:model.vision_tower.*"], dampening_frac=0.01)` — the Hessian-based
column-wise algorithm described above, at INT4/group_size=128. `lm_head` and
the vision tower are left unquantized (text-only MT task; the vision tower's
channel dims also aren't divisible by 128). The result is saved in
compressed-tensors format (`save_compressed=True`); tokenizer/processor files
are copied straight from the source checkpoint rather than re-saved, since the
tokenizer instance used for calibration has its truncation state mutated and
would otherwise bake a silent 512-token cap into the saved `tokenizer.json`.

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

