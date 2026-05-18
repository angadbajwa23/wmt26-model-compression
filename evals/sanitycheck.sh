#!/usr/bin/env bash
set -euo pipefail

timeout_duration=${TIMEOUT_DURATION:-900}
lang_pair=${LANG_PAIR:-ces-deu}
batch_size=${BATCH_SIZE:-1}
skip_setup=${SKIP_SETUP:-0}

if [[ $# -gt 0 ]]; then
    candidates=("$@")
else
    candidates=()
    while IFS= read -r run_script; do
        candidates+=("$(dirname "$run_script")")
    done < <(find ./submissions -mindepth 2 -maxdepth 2 -name run.sh 2>/dev/null | sort)
fi

if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No submissions found. Pass submission directories or add submissions/*/run.sh." >&2
    exit 1
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
input_file="$tmp_dir/input.txt"
printf 'Veta1\nVeta2 slovo2 slovo3\nVeta3\n' > "$input_file"
input_lines=$(wc -l < "$input_file")

for submission_dir in "${candidates[@]}"; do
    run_script="$submission_dir/run.sh"
    setup_script="$submission_dir/setup.sh"
    submission_id=$(basename "$(realpath "$submission_dir")")
    echo "===Sanity checking submission ID: $submission_id==="
    if [[ ! -f "$run_script" ]]; then
        echo "ERROR: run.sh not found in $submission_dir"
        continue
    fi

    echo "Submission directory: $(realpath "$submission_dir")"
    du -sh "$submission_dir"

    if [[ "$skip_setup" != 1 && -f "$setup_script" ]]; then
        echo "Running setup.sh for $submission_id"
        if ! timeout "$timeout_duration" bash "$setup_script"; then
            echo "ERROR: setup.sh failed for submission ID: $submission_id"
            continue
        fi
    fi

    output_file="$tmp_dir/$submission_id.out"
    start_time=$(date +%s)
    if ! timeout "$timeout_duration" bash "$run_script" \
        --lang-pair "$lang_pair" \
        --batch-size "$batch_size" \
        --input "$input_file" \
        --output "$output_file"; then
        echo "ERROR: run.sh failed for submission ID: $submission_id"
        continue
    fi
    elapsed_time=$(($(date +%s) - start_time))
    echo "Command executed in $elapsed_time seconds for submission ID: $submission_id"

    if [[ ! -s "$output_file" ]]; then
        echo "ERROR: Output is empty for submission ID: $submission_id"
        continue
    fi
    output_lines=$(wc -l < "$output_file")
    if [[ "$output_lines" -ne "$input_lines" ]]; then
        echo "ERROR: Output line count ($output_lines) does not match input line count ($input_lines)"
        continue
    fi
    echo "SUCCESS: Submission ID $submission_id passed the sanity check."
done

echo "Sanity check completed. See ERROR messages above for any failures."
