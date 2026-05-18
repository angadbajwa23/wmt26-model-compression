#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
model_dir="${MODEL_DIR:-$root_dir/workdir/model}"
cache_dir="${MODEL_CACHE:-/mnt/tg/data/projects/wmt26/model-compression/models}"
model_id="${MODEL_ID:-google/gemma-3-12b-it}"

if [[ ! -x "$root_dir/.venv/bin/python" ]]; then
    bash "$root_dir/setup.sh"
fi

python_bin="${PYTHON:-$root_dir/.venv/bin/python}"
"$python_bin" "$root_dir/prepare_model.py" \
    --model-id "$model_id" \
    --cache-dir "$cache_dir" \
    --output "$model_dir" \
    --quantization bnb-q4