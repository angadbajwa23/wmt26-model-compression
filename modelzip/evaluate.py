#!/usr/bin/env python

# 2025-05-09: Initial version by TG Gowda

"""
Evaluation pipeline. The tests and metrics are for demo purposes only.
The official evaluation will use held-out WMT26 test sets.
"""
import argparse
import json
import logging as LOG
import os
import subprocess as sp
from pathlib import Path

from modelzip.config import DEF_BATCH_SIZE, DEF_LANG_PAIRS, TASK_CONF, WORK_DIR, normalize_lang_pair
import shutil
import time, resource

DEF_SHOW_PROGRESS = False
PYMARIAN_CACHE = os.getenv("PYMARIAN_CACHE", "/mnt/tg/data/cache/marian/metric")
PYMARIAN_EXTRA = os.getenv("PYMARIAN_EXTRA", "-c 16")  # default: CPU threads (GPU fused attention NaN on H100)


def get_score(src_file: Path, out_file: Path, ref_file: Path, metric: str):
    if metric == "chrf":
        cmd = f"sacrebleu {ref_file} -i {out_file} -m {metric} -b -lc"
    else:
        cmd = f"pymarian-eval --cache {PYMARIAN_CACHE} {PYMARIAN_EXTRA} -m {metric} -r {ref_file} -t {out_file} -s {src_file} -a only"
    LOG.info(f"Scoring: {cmd}")
    return sp.check_output(cmd, shell=True, text=True).strip()


def get_run_cmd(model_dir: Path) -> list[str]:
    """Return the command for a submission directory containing run.sh."""
    run_script = model_dir / "run.sh"
    assert run_script.exists(), f"run.sh not found in {model_dir}"
    return ["bash", str(run_script)]


def line_count(file: Path) -> int:
    """Returns the number of lines in a file."""
    with open(file, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def evaluate(
    tests_dir: Path,
    model_dir: Path,
    langs=DEF_LANG_PAIRS,
    metrics=TASK_CONF["metrics"],
    test_names=None,
    batch_size: int = DEF_BATCH_SIZE,
    show_progress: bool = DEF_SHOW_PROGRESS,
    backup_dir: Path | None = None,
    run_num: int=1,
):

    run_cmd = get_run_cmd(model_dir)
    model_name = model_dir.name
    for pair in [normalize_lang_pair(lang) for lang in langs]:
        src, tgt = pair.split("-")
        lang_dir = tests_dir / pair
        pair_test_names = test_names
        if not pair_test_names:
            pair_test_names = [f.name.replace(f".{src}-{tgt}.{src}", "") for f in lang_dir.glob(f"*.{src}-{tgt}.{src}")]
            LOG.info(f"No test names specified. Using all available tests for {pair}: {pair_test_names}")
        for test_name in pair_test_names:
            src_file = lang_dir / f"{test_name}.{src}-{tgt}.{src}"
            if not src_file.exists():
                LOG.info(f"{test_name=} is unavailable for {pair}. {src_file} does not exist. Skipping.")
                continue
            ref = lang_dir / f"{test_name}.{src}-{tgt}.{tgt}"
            out = lang_dir / f"{test_name}.{src}-{tgt}.{tgt}.{model_name}.out.batch{batch_size}.run{run_num}"
            stats_file = out.with_suffix(out.suffix + ".stats.json")
            if not out.exists() or out.stat().st_size == 0:
                tmp_file = out.with_suffix(out.suffix + ".tmp")
                tmp_file.unlink(missing_ok=True)

                run_cmd_full = run_cmd + [
                    "--lang-pair",
                    pair,
                    "--batch-size",
                    str(batch_size),
                    "--input",
                    str(src_file),
                    "--output",
                    str(tmp_file),
                ]
                if show_progress:
                    run_cmd_full.append("--progress")
                LOG.info("Running command: %s", " ".join(run_cmd_full))
                try:
                    stats_start_time = time.time()
                    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
                    proc = sp.Popen(run_cmd_full)
                    proc.wait()
                    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
                    stats_end_time = time.time()
                    if proc.returncode != 0:
                        raise sp.CalledProcessError(proc.returncode, run_cmd_full)
                    if not tmp_file.exists() or tmp_file.stat().st_size == 0:
                        LOG.error("Submission did not write output file %s", tmp_file)
                        continue
                    expected_lines = line_count(src_file)
                    observed_lines = line_count(tmp_file)
                    if observed_lines != expected_lines:
                        LOG.error(
                            "Output line count mismatch for %s: expected=%d observed=%d",
                            tmp_file,
                            expected_lines,
                            observed_lines,
                        )
                        tmp_file.unlink(missing_ok=True)
                        continue
                    stats = {
                        "run_num": run_num,
                        "model_name": model_name,
                        "batch_size": batch_size,
                        "out_file": str(out),
                        "job_name": os.getenv("JOB_NAME", os.getenv("SUB_ID", "N/A")),
                        "command": run_cmd_full,
                        "exit_code": proc.returncode,
                        "wall_time_sec": stats_end_time - stats_start_time,
                        "user_time_sec": r1.ru_utime - r0.ru_utime,
                        "sys_time_sec": r1.ru_stime - r0.ru_stime,
                        "max_rss_kb": r1.ru_maxrss,  # max resident set size (KB on Linux)
                        "inblock": r1.ru_inblock - r0.ru_inblock,
                        "oublock": r1.ru_oublock - r0.ru_oublock,
                        "voluntary_ctx_switches": r1.ru_nvcsw - r0.ru_nvcsw,
                        "involuntary_ctx_switches": r1.ru_nivcsw - r0.ru_nivcsw,
                        "start_timestamp": stats_start_time,
                        "end_timestamp": stats_end_time,
                    }
                    with open(stats_file, "a", encoding="utf-8") as sf:
                        sf.write(json.dumps(stats, ensure_ascii=False, indent=None) + "\n")

                    LOG.info(f"Wrote stats to {stats_file}")
                    tmp_file.rename(out)
                    LOG.info(f"Wrote translations to {out}")
                except sp.CalledProcessError as e:
                    LOG.error(f"Error running command: {e}")
                    continue
            for m in metrics:
                if not ref.exists() or ref.stat().st_size == 0:
                    LOG.info("Skipping %s for %s because reference file is missing", m, out)
                    continue
                score_file = out.with_suffix(out.suffix + f".{m}.score")
                if not score_file.exists() or score_file.stat().st_size == 0:
                    try:
                        score = get_score(src_file, out, ref, m)
                        score_file.write_text(score)
                        LOG.info(f"{score_file.name} : {score}")
                    except sp.CalledProcessError as e:
                        LOG.error(f"Error scoring {out} with {m}: {e}")
                        continue
                else:
                    LOG.info(f"Skipping existing score file {score_file}")
            if backup_dir:
                backup_results(lang_dir, backup_dir / lang_dir.name)


def backup_results(from_dir:Path, to_dir:Path):
    """backup from from_dir to to_dir. Update new files. ignore existing and old files"""
    if not from_dir.exists():
        LOG.warning(f"Source directory {from_dir} does not exist")
        return
    LOG.info(f"Backing up results from {from_dir} --> {to_dir}")
    to_dir.mkdir(parents=True, exist_ok=True)
    copied, updated, skipped = 0, 0, 0
    for src in from_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(from_dir)
        dst = to_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not dst.exists() or dst.stat().st_size == 0:
                shutil.copy2(src, dst)
                copied += 1
                LOG.info(f"Copied new file {rel}")
            else:
                if src.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(src, dst)
                    updated += 1
                    LOG.info(f"Updated file {rel}")
                else:
                    skipped += 1
        except OSError as e:
            LOG.error(f"Failed to copy {rel}: {e}")

    LOG.info(f"Backup summary: copied={copied} updated={updated} skipped={skipped}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WMT26 model-compression submissions", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-w", "--work", type=Path, default=WORK_DIR)
    parser.add_argument("-l", "--langs", nargs="+", help="Lang pairs to evaluate", default=DEF_LANG_PAIRS)
    parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=DEF_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "-m", "--model", type=Path, required=True, help="Path to submission directory containing run.sh"
    )
    parser.add_argument("-t", "--test-names", nargs="+", help="Test names to evaluate; e.g. warmup. default: all tests available.", default=[])
    parser.add_argument(
        "-M", "--metrics", nargs="+", default=TASK_CONF["metrics"], help="Metrics to use for evaluation"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pbar", dest="show_progress", action="store_true", default=DEF_SHOW_PROGRESS,
        help="Enable progress bar during evaluation"
    )
    group.add_argument(
        "--no-pbar", dest="show_progress", action="store_false", default=DEF_SHOW_PROGRESS,
        help="Disable progress bar during evaluation"
    )

    job_name = os.environ.get("JOB_NAME") or os.environ.get("SUB_ID") or "local"
    def_backup_name = f"/mnt/tg/data/projects/wmt26/model-compression/evals/backup-v1/{job_name}"
    parser.add_argument(
        "-B", "--backup", type=Path, default=def_backup_name,
        help=f"Backup directory to save or update results. Use shared drive like blob container mount for archival purposes.")

    parser.add_argument("-r", "--runs", type=int, default=1, help="Number of runs to perform")
    args = parser.parse_args()
    tests_dir = args.work / "tests"
    for i in range(1, args.runs + 1):
        LOG.info(f"Starting run {i}/{args.runs}")
        evaluate(tests_dir, args.model, langs=args.langs, batch_size=args.batch_size,
                test_names=args.test_names,
                metrics=args.metrics, show_progress=args.show_progress, backup_dir=args.backup,
                run_num=i)


if __name__ == "__main__":
    main()
