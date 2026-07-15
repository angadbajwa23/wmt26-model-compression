# WMT26 Model Compression

This repository contains organizer tooling, evaluation helpers, and organizer-provided example submissions for the [WMT26 Model Compression shared task](https://www2.statmt.org/wmt26/model-compression.html).

The root of the repository is for organizers. Participant-style systems live under `submissions/`, and each submission directory is self-contained.

## Task Summary

- Constrained track: compress `google/gemma-3-12b-it`.
- Unconstrained track: use any original model below 20B parameters.
- Language pairs: `ces-deu`, `eng-zho_Hans`, and `eng-ara_EG`.
- Evaluation hardware: Ubuntu 24.04 on x86_64 with one NVIDIA H100 GPU up to 80 GB VRAM.
- Evaluation criteria: translation quality, model size/on-disk and VRAM usage, and inference speed.

`google/gemma-3-12b-it` is gated on Hugging Face. Accept the Gemma terms and provide access through `HF_TOKEN`, `huggingface-cli login`, or an authorized cache before downloading the constrained baseline. Organizer submission setup scripts default to the shared model cache `/mnt/tg/data/projects/wmt26/model-compression/models` via `MODEL_CACHE`.

## Repository Layout

```text
wmt26-model-compression/
  modelzip/                 Organizer evaluation package
  evals/                    Organizer sanity/evaluation scripts
  submissions/              Runnable submissions
    baseline/               Uncompressed Gemma baseline
    bnb-q8/                 BitsAndBytes q8 baseline
    bnb-q4/                 BitsAndBytes q4 baseline
```

Participant zip files should be extracted into `submissions/<submission-id>/`. Hugging Face model repositories can be added as submodules in the same location. The evaluator treats the submission directory itself as the runnable model path.

## Submission Contract

Each submission directory must include at least:

```text
setup.sh
run.sh
requirements.txt
README.md
```

`setup.sh` prepares that submission's environment for inference only. The submitted or pre-compressed model artifact should already be present in the submission directory, usually at `workdir/model`, or `run.sh` should honor `MODEL_DIR`. It should not prepare evaluation data; root organizer tooling does that.

`compress.sh` may be included as an optional reproducibility recipe for creating the submitted model artifact. Organizers do not need it to run inference.

Organizer-provided submissions install the shared `modelzip` helper package into their own `.venv` with:

```bash
uv pip install --no-deps -e <organizer-repo-root>
```

That package provides stable submission-facing utilities such as language-pair normalization, language names, prompt helpers, and line-oriented input/output validation. It is installed into each submission environment. Set `MODELZIP_SOURCE` only when the default local repo discovery is not enough. The value may be a local repo directory, wheel path, git URL, or package spec; local directories are installed editable.

`run.sh` is the official inference entry point:

```bash
bash run.sh \
  --lang-pair ces-deu \
  --batch-size 8 \
  --input input.txt \
  --output output.txt
```

The script must produce exactly one output line for each input line. Logs and progress bars must go to stderr or separate files, never into the output file.

The compatibility positional form is allowed but not required:

```bash
bash run.sh ces-deu 8 < input.txt > output.txt
```

## Running Submissions

Install root tooling from the repository root:

```bash
python -m pip install -e .
```

Prepare development/evaluation data:

```bash
python -m modelzip.setup --work workdir --langs ces-deu eng-zho_Hans eng-ara_EG
```

Until WMT26 test data is released, this repo uses the WMT25 General MT blindset for local smoke/speed testing:

```text
https://data.statmt.org/wmt25/general-mt/wmt25.jsonl
```

That file is source-only. It is suitable for setup, inference, line-count validation, and speed testing. For local reference-based scoring, the loader also supports WMT25 post-task reference records with either `refs.refA.ref` or `tgt_text.refA`.

Set up a submission (repeat per submission you want to run):

```bash
bash submissions/baseline/setup.sh
```

Run a sanity check on one submission:

```bash
CUDA_VISIBLE_DEVICES=3 bash evals/sanitycheck.sh submissions/baseline
```

Run sanity checks on all discoverable submissions:

```bash
CUDA_VISIBLE_DEVICES=3 bash evals/sanitycheck.sh
```

Run a local chrF warmup evaluation:

```bash
python -m modelzip.evaluate \
  --work workdir \
  --model submissions/baseline \
  --langs ces-deu \
  --test-names warmup \
  --metrics chrf \
  --batch 1
```

Run the organizer evaluation wrapper over all submissions:

```bash
CUDA_VISIBLE_DEVICES=3 bash evals/evaluate.sh
```

Submission `run.sh` scripts should respect the caller's `CUDA_VISIBLE_DEVICES` and must not overwrite it. To fan out organizer evaluation jobs over multiple GPUs with GNU Parallel, set `PARALLEL_JOBS` and optionally `GPU_IDS`:

```bash
PARALLEL_JOBS=8 GPU_IDS=0,1,2,3,4,5,6,7 bash evals/evaluate.sh
```

If `GPU_IDS` is omitted, the wrapper uses `0..PARALLEL_JOBS-1`. Setup still runs once per submission before parallel evaluation starts.

## Organizer Baselines

`baseline` is the uncompressed Gemma 3 12B baseline using PyTorch and Transformers.

`bnb-q8` and `bnb-q4` are simple BitsAndBytes memory/compression baselines. They are useful reference points, but they are not necessarily the fastest H100 inference route. Engine-native variants such as vLLM AWQ/GPTQ/FP8 or TensorRT-LLM INT4/FP8 should be represented as separate submission directories.
