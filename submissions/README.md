# Submissions

Each subdirectory in `submissions/` is a self-contained submission to the WMT26 Model Compression shared task.

## Setup

To start a new submission, copy one of the existing baselines (`baseline`, `bnb-q8`, or `bnb-q4`) into a new directory and replace the requirements, setup, model artifact, and inference code as needed. Participant zip archives should be extracted into their own directory; Hugging Face submission repositories can be added there as submodules. 

A submission directory must contain:

```text
setup.sh
run.sh
requirements.txt
README.md
```

`setup.sh` prepares the submission's runtime environment for inference only, inlucding setting up its own venv. The submitted or pre-compressed model artifact should already be present, usually at `workdir/model`; otherwise `run.sh` must honor `MODEL_DIR`.

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

## Baselines

Organizer-provided examples:

- `baseline`: uncompressed Gemma 3 12B baseline.
- `bnb-q8`: BitsAndBytes q8 baseline.
- `bnb-q4`: BitsAndBytes q4 baseline.

See individual `README.md` files for more details.