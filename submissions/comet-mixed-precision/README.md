# comet-mixed-precision

COMET-guided mixed-precision quantization for WMT26 Model Compression — constrained track.

- **Base model**: `google/gemma-3-12b-it`
- **Quantization**: per-decoder-layer BF16 / INT8 (W8A16) / INT4 (W4A16_ASYM), group_size=128,
  chosen by measuring each layer's translation-quality (COMET) sensitivity and solving a
  size-budget knapsack — not a uniform precision across all layers, unlike `awq`/`gptq`.
- **Inference engine**: vLLM with `quantization="compressed-tensors"` 
- **Compression library**: [llm-compressor](https://github.com/vllm-project/llm-compressor) via
  `GPTQModifier`'s `config_groups`
- **Model artifact**: [Angad23/gemma-3-12b-it-comet-mixed-precision-wmt26](https://huggingface.co/Angad23/gemma-3-12b-it-comet-mixed-precision-wmt26)
  — `setup.sh` downloads this automatically into `workdir/model` (override with `HF_MODEL_REPO` / `MODEL_DIR`)

## How this differs from `awq`/`gptq`

Those submissions quantize every linear layer to the same precision. Some layers matter far more
to translation quality than others; treating them identically either wastes size budget on
layers that don't need it, or over-compresses layers that do. This submission instead:

1. **Measures per-layer sensitivity** (`sensitivity_sweep.py`): loads the BF16 model once, then
   for each of the 48 decoder layers, in-place round-to-nearest (RTN) fake-quantizes that layer's
   7 linear weights to INT4, translates a held-out dev subset, scores with COMET
   (`wmt22-comet-da` via `pymarian-eval`, matching the official evaluation metric), records the
   quality drop vs. the BF16 baseline, then restores the original weights before moving to the
   next layer.
2. **Solves a budget knapsack** (`solve_budget.py`): given the 48 sensitivity scores and exact
   per-layer byte costs at each tier (read from the base model's safetensors header — no
   hardcoded architecture constants), an exact DP knapsack picks each layer's tier (BF16 / INT8 /
   INT4) to minimize total expected COMET drop subject to a target model size.
3. **Quantizes once** (`prepare_model.py`): a single `GPTQModifier(config_groups=..., ignore=...)`
   call applies Hessian-corrected GPTQ quantization to the INT8 and INT4 tiers in one calibration
   pass; BF16-tier layers are left untouched via `ignore`. Concretely, `config_groups` maps each
   tier to a `QuantizationScheme`: the int8 tier is `num_bits=8, symmetric=True, strategy="group",
   group_size=128` (W8A16), the int4 tier is `num_bits=4, symmetric=False, strategy="group",
   group_size=128` (`W4A16_ASYM`, same convention as `gptq`'s uniform recipe). `ignore` is built
   from `lm_head`, the vision tower, and the exact Linear-layer names of every bf16-tier decoder
   layer (via `layer_utils.discover_decoder_layers`/`layer_linear_names`), so those layers pass
   through `oneshot()` completely untouched.

## Held-out dev set for the sensitivity sweep

The only reference-bearing local data (`data/wmt25/*.paragraphs.jsonl`) is 44–65% already
consumed by `data_prep/build_calibration.py` into `data/calibration/`. Reusing it wholesale for
the sensitivity sweep would measure quality on sentences the quantizer's own calibration pass
already saw. `build_devset.py` excludes any wmt25 record whose reference already appears in
`data/calibration/<pair>/calibration.jsonl` and samples 30 sentences/pair (~90 total) from the
genuinely held-out remainder, written to `data/sensitivity-devset/`.

## Known approximations

- **INT8 sensitivity is estimated, not measured.** The sweep only probes INT4 (the "Fast" depth
  option — full INT4+INT8 sweep would take ~4-5h vs. ~1.5-2h). `solve_budget.py` estimates each
  layer's INT8 drop as `0.5 × measured_INT4_drop` (`--int8-recovery-factor`, a linear
  interpolation heuristic — INT8's quantization step is roughly half of INT4's). Re-running the
  sweep with both precisions probed is the natural follow-up if this submission looks
  competitive.
- **Per-layer sensitivity is measured independently** (one layer quantized at a time, rest BF16).
  The knapsack assumes drops are roughly additive across the final multi-layer assignment —
  interaction effects between simultaneously-quantized layers aren't modeled. This is the
  standard simplification used in layer-sensitivity bit-allocation approaches.
- **The sensitivity probe uses RTN** (no Hessian correction) purely to *rank* layers cheaply. The
  actually deployed weights are produced by GPTQ's Hessian-corrected quantization in
  `prepare_model.py`, a materially better quantizer than the ranking probe.
- **The realized artifact is larger than `solve_budget.py` estimates: ~11 GiB on disk vs. the
  ~8.6 GiB `estimated_size_bytes` in `workdir/tier_assignment.json` the knapsack solved for
  against the 9 GiB budget.** Per-tensor inspection of the saved safetensors header shows the
  int4/int8 tier costs are actually slightly *over*-estimated by `tier_bytes()` (real packed
  weight+scale+zero-point bytes come in a little under budget for both tiers), so that's not the
  cause. The gap traces almost entirely to `language_model.lm_head.weight` — a full ~1.9 GiB BF16
  tensor present in this artifact's `model.safetensors.index.json` despite `config.json`'s
  `tie_word_embeddings: true`, and despite `gptq`'s save (the same
  `model.save_pretrained(..., save_compressed=True)` call) correctly emitting *no* separate
  `lm_head.weight` at all. `fixed_overhead_bytes()` computes its BF16 baseline from the base
  checkpoint's own safetensors header, which — consistent with tied embeddings — has no
  standalone `lm_head.weight` tensor either, so the knapsack never budgets for one.
  llm-compressor's own `save_pretrained` wrapper (`compressed_tensors_utils.py::
  _retie_offloaded_weights`) exists specifically to prevent this — accelerate's offload
  conversion splits tied weights into separate parameters, so llm-compressor calls
  `model.tie_weights()` again right before saving to re-merge them, logging a WARNING and
  falling back to a redundant duplicate if that call fails. Available logs don't capture
  which outcome happened here, but `gptq`'s otherwise-identical save clearly hit the success
  path and this one didn't. Untying that duplicate would bring the artifact to roughly the intended
  ~9.1 GiB — worth fixing before relying on the budget number.

Model size after compression: ~11 GB on disk today (down from ~23 GB in BF16); the untied
`lm_head` fix above would bring it close to the intended ~9 GB target.

## Setup

```bash
bash setup.sh
```

Installs runtime and quantization dependencies plus the organizer `modelzip` package into
`./.venv`. `.venv-compress` gets its own `llmcompressor` + torch/transformers stack, same as
`awq`/`gptq`.

## Compress (offline, run once — optional, not needed to run inference)

`setup.sh` already downloads the finished quantized model from Hugging Face
(see **Model artifact** above), so nothing below is required to run this
submission. It's documented here for reproducibility of how that artifact was
built.

```bash
bash compress.sh
```

Runs, in order: held-out dev-set construction, the 48-layer sensitivity sweep, the budget solve,
and the final mixed-precision GPTQ oneshot. Each stage is skipped if its output already exists —
delete the corresponding file under `workdir/` (or `data/sensitivity-devset/`) to force a rerun
of just that stage.

The final stage (`prepare_model.py`) draws its calibration set from the same
`data/calibration/<pair>/calibration.jsonl` pool as
[`gptq`](../gptq/README.md#1-calibration-data) and
[`awq`](../awq/README.md#1-calibration-data) — built by
[`data_prep/build_calibration.py`](../../data_prep/build_calibration.py) from
WMT25 dev paragraphs plus public News Commentary/OPUS corpora. `CALIB_SAMPLES_PER_PAIR=64`
(default) takes an evenly-strided 64 examples per pair, 192 total, formatted
and tokenized the same way (Gemma3 chat template, `add_generation_prompt=False`,
512-token truncation). This is purely a Hessian-calibration pass for GPTQ, not
a quality measurement, so reusing the same pool the quantizer calibration
already saw is fine — unlike the sensitivity sweep below, which needs
genuinely held-out data to rank layers honestly.

Override defaults via environment variables:

| Variable            | Default                              | Description                              |
|---------------------|---------------------------------------|-------------------------------------------|
| `MODEL_ID`          | `google/gemma-3-12b-it`              | Base model repo ID or local path          |
| `MODEL_CACHE`       | `/mnt/tg/.../models`                 | Cache directory for downloaded models     |
| `MODEL_DIR`         | `workdir/model`                      | Output path for the quantized model       |
| `CALIB_DIR`         | `../../data/calibration`             | Calibration data for the final GPTQ pass  |
| `CALIB_SAMPLES_PER_PAIR` | `64`                              | Calibration sentences per language pair   |
| `DEVSET_DIR`        | `../../data/sensitivity-devset`      | Held-out dev set for the sensitivity sweep|
| `DEVSET_N_PER_PAIR`  | `30`                                 | Dev sentences per pair for the sweep      |
| `BUDGET_GB`          | `9`                                  | Target total model size (binary GiB)      |
| `SENSITIVITY_FILE`   | `workdir/sensitivity.json`           | Per-layer sensitivity scores              |
| `TIER_FILE`          | `workdir/tier_assignment.json`       | Resolved BF16/INT8/INT4 tier per layer    |

## Run

```bash
bash run.sh --lang-pair ces-deu --input input.txt --output output.txt
```

Supported language pairs: `ces-deu`, `eng-zho_Hans`, `eng-ara_EG`

Set `MODEL_DIR` to point at a pre-quantized model artifact if it lives outside the default
`workdir/model` path.
