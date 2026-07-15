#!/usr/bin/env bash
set -euo pipefail
# Offline GPTQ quantization recipe. Run once to produce the model artifact at workdir/model.
# This script is not part of the evaluation contract; it documents how the submitted model was built.

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$root_dir/.venv-compress/bin/python" "$root_dir/prepare_model.py" \
    --model-id "${MODEL_ID:-google/gemma-3-12b-it}" \
    --cache-dir "${MODEL_CACHE:-/mnt/tg/data/projects/wmt26/model-compression/models}" \
    --output "${MODEL_DIR:-$root_dir/workdir/model}" \
    --calib-dir "${CALIB_DIR:-$root_dir/../../data/calibration}" \
    --calib-samples-per-pair "${CALIB_SAMPLES_PER_PAIR:-64}" \
    --bits 4 \
    --group-size 128
