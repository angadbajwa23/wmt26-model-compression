# bnb-q8

BitsAndBytes q8 organizer baseline for WMT26 Model Compression.

- Base model: `google/gemma-3-12b-it`
- Runtime: PyTorch + Transformers + BitsAndBytes
- Model artifact: BNB q8 model at `workdir/model`, or set `MODEL_DIR` when running.

This is a simple memory/compression baseline, not necessarily the fastest H100 inference route. Engine-native variants such as vLLM AWQ/GPTQ/FP8 or TensorRT-LLM INT4/FP8 should be separate submissions.

## Setup

```bash
bash setup.sh
```

`setup.sh` installs this submission's runtime dependencies and the organizer `modelzip` helper package into `./.venv`. When running outside the organizer repository, set `MODELZIP_SOURCE`.

## Compress

```bash
bash compress.sh
```

`compress.sh` is optional documentation/reproducibility support. It loads `google/gemma-3-12b-it` with `load_in_8bit=True` and saves the prepared artifact under `workdir/model` by default.

## Run

```bash
bash run.sh --lang-pair ces-deu --batch-size 1 --input input.txt --output output.txt
```
