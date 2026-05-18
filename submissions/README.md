# Submissions

Each directory here is a self-contained WMT26 Model Compression submission. Participant zip archives should be extracted here, and Hugging Face submission repositories can be added here as submodules.

A submission directory must contain:

```text
setup.sh
run.sh
requirements.txt
README.md
```

`setup.sh` prepares that submission's runtime environment for inference only. The submitted or pre-compressed model artifact should already be present in the submission directory, usually at `workdir/model`, or `run.sh` should honor `MODEL_DIR`.

`compress.sh` is optional. It is not part of the evaluation contract; it is a documentation/reproducibility recipe for generating the submitted model artifact from a base model.

Submissions may use organizer-provided shared helpers from `modelzip.submission` for language names, language-pair aliases, prompt formatting, line-oriented input/output, and base classes for Python-based inference. Use `LLMBase` for generic causal language models and `Gemma3LLMBase` for Gemma 3 submissions. Install those helpers into the submission venv during `setup.sh`. The organizer examples do this with an editable, no-dependency install from the repository root:

```bash
uv pip install --no-deps -e <organizer-repo-root>
```

For standalone submission repositories, set `MODELZIP_SOURCE` before running `setup.sh`. The value may be a local repo directory, wheel path, git URL, or package spec; local directories are installed editable.

`run.sh` is the evaluator entry point. The official interface is:

```bash
bash run.sh --lang-pair ces-deu --batch-size 8 --input input.txt --output output.txt
```

The script must write exactly one output line for each input line. Logs, progress bars, and diagnostics must go to stderr or separate files, never into the output file.

The evaluator may launch multiple `run.sh` processes in parallel and assign GPUs with `CUDA_VISIBLE_DEVICES`. A submission script must respect the inherited value and should not set or overwrite it internally.

Organizer-provided examples:

- `baseline`: uncompressed Gemma 3 12B baseline.
- `bnb-q8`: BitsAndBytes q8 baseline.
- `bnb-q4`: BitsAndBytes q4 baseline.
