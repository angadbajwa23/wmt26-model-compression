# Submissions

Each subdirectory in `submissions/` is a self-contained submission to the WMT26 Model Compression shared task.

## Setup

Install all requirements  in repository root:

```bash
pip install -r requirements.txt
```

Install root tooling from the repository root:

```bash
python -m pip install -e .
```

Prepare development/evaluation data:

```bash
python -m modelzip.setup --work workdir --langs ces-deu eng-zho_Hans eng-ara_EG
```

To start a new submission, copy one of the existing baselines (`../organizer_submissions/baseline`, `bnb-q8`, or `bnb-q4`) into a new directory here and replace the requirements, setup, model artifact, and inference code as needed. Participant zip archives should be extracted into their own directory; Hugging Face submission repositories can be added there as submodules. 

A submission directory must contain:

```text
setup.sh
run.sh
requirements.txt
README.md
```

`setup.sh` prepares the submission's runtime environment for inference and also downloads that submission's quantized model artifact from Hugging Face directly into its own `workdir/model` 

To set up and rerun a baseline or submission:

```bash
cd submissions/<name>
bash setup.sh
bash run.sh --lang-pair ces-deu --batch-size 8 --input input.txt --output output.txt
```

Submissions may use helpers from `modelzip.submission` for language-pair normalization, prompt formatting, line-oriented I/O, and Python inference base classes. Install those helpers into the submission venv during `setup.sh`; the baseline scripts demonstrate the editable install. For standalone repositories, set `MODELZIP_SOURCE` before running `setup.sh`.

`run.sh` is the evaluator entry point and must support the following options:

```bash
bash run.sh --lang-pair ces-deu --batch-size 8 --input input.txt --output output.txt
```

The script must write exactly one output line for each input line. Logs, progress bars, and diagnostics must go to stderr or separate files. The evaluator may launch multiple `run.sh` processes in parallel and assign GPUs with `CUDA_VISIBLE_DEVICES`; submission scripts must respect the inherited value.

## Submissions

This directory contains this team's three submissions, all quantizing `google/gemma-3-12b-it`:

- [`gptq`](gptq/README.md): Uniform INT4 (W4A16_ASYM) via GPTQ's Hessian-corrected column-wise quantization.
- [`awq`](awq/README.md): Uniform INT4 (W4A16_ASYM) via AWQ's activation-aware channel scaling.
- [`comet-mixed-precision`](comet-mixed-precision/README.md): Per-layer BF16/INT8/INT4 mix chosen by
  COMET-sensitivity-guided budget allocation, quantized in one GPTQ pass.

Best-performing submission by language pair: `comet-mixed-precision` for `ces-deu` and `eng-zho_Hans`;
`gptq` for `eng-ara_EG`.

See individual `README.md` files for more details.