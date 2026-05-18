#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MODEL_DIR="${MODEL_DIR:-$root_dir/workdir/model}"
if [[ -x "$root_dir/.venv/bin/python" ]]; then
    python_bin="${PYTHON:-$root_dir/.venv/bin/python}"
else
    python_bin="${PYTHON:-python}"
fi
exec "$python_bin" "$root_dir/inference.py" --model "$MODEL_DIR" "$@"