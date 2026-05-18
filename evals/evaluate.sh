#!/usr/bin/env bash
set -euo pipefail

work=${WORK_DIR:-$PWD/workdir}
warmup_runs=${WARMUP_RUNS:-3}
full_runs=${FULL_RUNS:-3}
job_name=${JOB_NAME:-${SUB_ID:-local}}
backup=${BACKUP_DIR:-/mnt/tg/data/projects/wmt26/model-compression/evals/backup-v1/$job_name}
skip_setup=${SKIP_SETUP:-0}
parallel_jobs=${PARALLEL_JOBS:-}
gpu_ids=${GPU_IDS:-${EVAL_GPUS:-}}
langs=${LANGS:-"ces-deu eng-zho_Hans eng-ara_EG"}

if [[ $# -gt 0 ]]; then
    submissions=("$@")
else
    submissions=()
    while IFS= read -r run_script; do
        submissions+=("$(dirname "$run_script")")
    done < <(find ./submissions -mindepth 2 -maxdepth 2 -name run.sh 2>/dev/null | sort)
fi

if [[ ${#submissions[@]} -eq 0 ]]; then
    echo "No submissions found. Pass submission directories or add submissions/*/run.sh." >&2
    exit 1
fi

python -m modelzip.setup -w "$work"

echo "Submissions: ${submissions[*]}"
echo "Backup: $backup"

metrics=${METRICS:-"chrf wmt22-comet-da wmt22-cometkiwi-da wmt23-cometkiwi-da-xl"}
read -r -a metric_args <<< "$metrics"
read -r -a lang_args <<< "$langs"

if [[ -n "$gpu_ids" ]]; then
    gpu_ids=${gpu_ids//,/ }
    read -r -a gpu_args <<< "$gpu_ids"
    if [[ ${#gpu_args[@]} -eq 0 ]]; then
        echo "GPU_IDS was set but no GPU IDs were parsed: $gpu_ids" >&2
        exit 1
    fi
    parallel_jobs=${parallel_jobs:-${#gpu_args[@]}}
fi

append_command() {
    local command_file=$1
    shift
    printf '%q ' "$@" >> "$command_file"
    printf '\n' >> "$command_file"
}

discover_test_names() {
    local pair=$1
    local src=${pair%%-*}
    local tgt=${pair#*-}
    local lang_dir="$work/tests/$pair"
    local src_file base suffix

    suffix=".$src-$tgt.$src"
    if [[ ! -d "$lang_dir" ]]; then
        return
    fi
    for src_file in "$lang_dir"/*"$suffix"; do
        [[ -e "$src_file" ]] || continue
        base=$(basename "$src_file")
        printf '%s\n' "${base%$suffix}"
    done | sort
}

run_command_file() {
    local phase=$1
    local command_file=$2

    if [[ ! -s "$command_file" ]]; then
        echo "No commands for $phase"
        return
    fi

    if [[ -n "$parallel_jobs" && "$parallel_jobs" -gt 1 ]]; then
        if [[ -z "$gpu_ids" ]]; then
            gpu_ids=$(seq -s ' ' 0 $((parallel_jobs - 1)))
            read -r -a gpu_args <<< "$gpu_ids"
        fi
        if ! command -v parallel >/dev/null 2>&1; then
            echo "GNU Parallel is required for GPU_IDS/PARALLEL_JOBS mode." >&2
            exit 1
        fi

        local gpu_ids_csv
        gpu_ids_csv=$(IFS=,; echo "${gpu_args[*]}")
        echo "=====Running $phase with GNU Parallel: jobs=$parallel_jobs gpus=$gpu_ids_csv====="
        run_eval_command() {
            local gpu_ids_csv=$1
            local slot=$2
            local command=$3
            local gpu_ids=()
            IFS=, read -r -a gpu_ids <<< "$gpu_ids_csv"
            export CUDA_VISIBLE_DEVICES="${gpu_ids[$(((slot - 1) % ${#gpu_ids[@]}))]}"
            bash -c "$command"
        }
        export -f run_eval_command
        parallel --will-cite --progress --halt soon,fail=1 -j "$parallel_jobs" \
            run_eval_command "$gpu_ids_csv" {%} {} :::: "$command_file"
    else
        echo "=====Running $phase sequentially====="
        while IFS= read -r command; do
            bash -c "$command"
        done < "$command_file"
    fi
}

run_parallel_eval() {
    local tmp_dir full_commands warmup_commands speed_commands
    local submission pair test_name batch_size

    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' EXIT

    full_commands="$tmp_dir/full.commands"
    for submission in "${submissions[@]}"; do
        for pair in "${lang_args[@]}"; do
            while IFS= read -r test_name; do
                append_command "$full_commands" python -m modelzip.evaluate \
                    -w "$work" -B "$backup" -r 1 -M "${metric_args[@]}" -m "$submission" -b 8 -l "$pair" -t "$test_name"
            done < <(discover_test_names "$pair")
        done
    done
    run_command_file "full eval" "$full_commands"

    for batch_size in 1 16 64 256 512; do
        warmup_commands="$tmp_dir/warmup-batch$batch_size.commands"
        speed_commands="$tmp_dir/speed-batch$batch_size.commands"
        for submission in "${submissions[@]}"; do
            append_command "$warmup_commands" python -m modelzip.evaluate \
                -w "$work" -B "$backup" -r "$warmup_runs" -M "${metric_args[@]}" -m "$submission" -b 1 -l ces-deu -t warmup
            append_command "$speed_commands" python -m modelzip.evaluate \
                -w "$work" -B "$backup" -r "$full_runs" -M "${metric_args[@]}" -m "$submission" -b "$batch_size" -l ces-deu -t wmt25-blind
        done
        run_command_file "warmup before batch $batch_size" "$warmup_commands"
        run_command_file "speed eval batch $batch_size" "$speed_commands"
    done
}

for submission in "${submissions[@]}"; do
    setup_script="$submission/setup.sh"
    if [[ "$skip_setup" != 1 && -f "$setup_script" ]]; then
        echo "=====Setup for $submission====="
        bash "$setup_script"
    fi
done

if [[ -n "$parallel_jobs" && "$parallel_jobs" -gt 1 ]]; then
    run_parallel_eval
    exit 0
fi

for submission in "${submissions[@]}"; do
    echo "=====Full eval on $submission with batch size 8====="
    python -m modelzip.evaluate -w "$work" -B "$backup" -r 1 -M "${metric_args[@]}" -m "$submission" -b 8 -l "${lang_args[@]}"

    for batch_size in 1 16 64 256 512; do
        echo "====Warming $submission====="
        python -m modelzip.evaluate -w "$work" -B "$backup" -r "$warmup_runs" -M "${metric_args[@]}" -m "$submission" -b 1 -l ces-deu -t warmup

        echo "=====Speed eval for $submission with batch size $batch_size on ces-deu wmt25-blind====="
        python -m modelzip.evaluate -w "$work" -B "$backup" -r "$full_runs" -M "${metric_args[@]}" -m "$submission" -b "$batch_size" -l ces-deu -t wmt25-blind
    done
done
