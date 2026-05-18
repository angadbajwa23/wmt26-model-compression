#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
venv_dir="$root_dir/.venv"
model_dir="$root_dir/workdir/model"
modelzip_source="${MODELZIP_SOURCE:-}"

usage() {
    cat <<'EOF'
Usage: bash setup.sh

Create a Python 3.12 uv environment and install BNB q4 inference requirements.

The quantized model is expected at workdir/model, or set MODEL_DIR when running.
Use compress.sh only as an optional recipe for producing the organizer baseline artifact.

Environment:
  MODELZIP_SOURCE           Shared modelzip helpers: local repo dir, wheel/git URL, or package spec
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo "setup.sh does not take command-line options; use environment variables shown below." >&2
    usage >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing it with python3 -m pip" >&2
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        python3 -m pip install uv
    else
        python3 -m pip install --user uv
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but could not be installed. See https://docs.astral.sh/uv/." >&2
    exit 1
fi

python_bin="$venv_dir/bin/python"
if [[ ! -x "$python_bin" ]]; then
    uv venv --python 3.12 "$venv_dir"
fi
uv pip install --python "$python_bin" -r "$root_dir/requirements.txt"

if [[ -z "$modelzip_source" && -f "$root_dir/../../modelzip/submission.py" ]]; then
    modelzip_source=$(cd "$root_dir/../.." && pwd)
fi
if [[ -z "$modelzip_source" ]]; then
    echo "Could not find modelzip helpers. Set MODELZIP_SOURCE or run from the organizer repository." >&2
    exit 1
fi
if [[ -d "$modelzip_source" ]]; then
    if [[ ! -f "$modelzip_source/modelzip/submission.py" ]]; then
        echo "MODELZIP_SOURCE does not contain modelzip/submission.py: $modelzip_source" >&2
        exit 1
    fi
    uv pip install --python "$python_bin" --no-deps -e "$modelzip_source"
else
    uv pip install --python "$python_bin" --no-deps "$modelzip_source"
fi

if [[ -f "$model_dir/config.json" ]]; then
    echo "Inference environment is ready; model found at $model_dir"
else
    echo "Inference environment is ready."
    echo "No model found at $model_dir; provide a pre-compressed model there, set MODEL_DIR at run time, or run compress.sh to generate the organizer baseline artifact." >&2
fi
