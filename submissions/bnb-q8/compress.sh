#!/usr/bin/env bash
set -euo pipefail

# This script is not part of the evaluation contract; it is a documentation/reproducibility recipe for generating the submitted model artifact from a base model.

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$root_dir/.venv/bin/python" "$root_dir/prepare_model.py" \
    --model-id "${MODEL_ID:-/workspace/wmt/wmt26-model-compression/submissions/baseline/workdir/model}" \
    --cache-dir "${MODEL_CACHE:-MODEL_CACHE not set}" \
    --output "${MODEL_DIR:-$root_dir/workdir/model}" \
    --quantization bnb-q8