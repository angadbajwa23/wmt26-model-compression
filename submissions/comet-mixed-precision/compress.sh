#!/usr/bin/env bash
set -euo pipefail
# COMET-guided mixed-precision quantization recipe. Run once to produce the model
# artifact at workdir/model. This script is not part of the evaluation contract; it
# documents how the submitted model was built.
#
# Stages (each skipped if its output already exists -- rerun individually by
# deleting the corresponding output file):
#   0. build_devset.py       -> data/sensitivity-devset/<pair>/dev.jsonl
#   1. sensitivity_sweep.py  -> workdir/sensitivity.json
#   2. solve_budget.py       -> workdir/tier_assignment.json
#   3. prepare_model.py      -> workdir/model  (final GPTQ oneshot, mixed tiers)

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
py="$root_dir/.venv-compress/bin/python"

devset_dir="${DEVSET_DIR:-$root_dir/../../data/sensitivity-devset}"
sensitivity_file="${SENSITIVITY_FILE:-$root_dir/workdir/sensitivity.json}"
tier_file="${TIER_FILE:-$root_dir/workdir/tier_assignment.json}"
model_id="${MODEL_ID:-google/gemma-3-12b-it}"
model_cache="${MODEL_CACHE:-/mnt/tg/data/projects/wmt26/model-compression/models}"

if [[ ! -d "$devset_dir" ]] || [[ -z "$(find "$devset_dir" -name dev.jsonl 2>/dev/null)" ]]; then
    echo "=== Stage 0: building held-out sensitivity dev set ==="
    "$py" "$root_dir/build_devset.py" \
        --n-per-pair "${DEVSET_N_PER_PAIR:-30}" \
        --out-dir "$devset_dir"
else
    echo "=== Stage 0: dev set already exists at $devset_dir, skipping ==="
fi

if [[ ! -f "$sensitivity_file" ]]; then
    echo "=== Stage 1: per-layer INT4 COMET-sensitivity sweep ==="
    "$py" "$root_dir/sensitivity_sweep.py" \
        --model-id "$model_id" \
        --cache-dir "$model_cache" \
        --devset-dir "$devset_dir" \
        --output "$sensitivity_file" \
        --group-size 128
else
    echo "=== Stage 1: $sensitivity_file already exists, skipping ==="
fi

if [[ ! -f "$tier_file" ]]; then
    echo "=== Stage 2: solving BF16/INT8/INT4 tier assignment under budget ==="
    "$py" "$root_dir/solve_budget.py" \
        --model-id "$model_id" \
        --cache-dir "$model_cache" \
        --sensitivity "$sensitivity_file" \
        --output "$tier_file" \
        --budget-gb "${BUDGET_GB:-9}" \
        --group-size 128
else
    echo "=== Stage 2: $tier_file already exists, skipping ==="
fi

echo "=== Stage 3: mixed-precision GPTQ oneshot ==="
"$py" "$root_dir/prepare_model.py" \
    --model-id "$model_id" \
    --cache-dir "$model_cache" \
    --output "${MODEL_DIR:-$root_dir/workdir/model}" \
    --calib-dir "${CALIB_DIR:-$root_dir/../../data/calibration}" \
    --calib-samples-per-pair "${CALIB_SAMPLES_PER_PAIR:-64}" \
    --tier-assignment "$tier_file" \
    --group-size 128
