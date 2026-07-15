#!/usr/bin/env python
"""
Summarize quality metrics and latency for a given batch size and run number.

Usage:
    python evals/summarize.py                    # defaults: batch=32, run=2
    python evals/summarize.py -b 8 -r 1
    python evals/summarize.py -w /other/workdir -b 32 -r 2
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

SKIP_TESTS = {"warmup", "wmt25-blind"}
QUALITY_TEST = "wmt25"


def load_scores(work_dir: Path, batch: int, run: int):
    """Returns {(model, lang, test, metric): score}"""
    scores = {}
    pattern = f"*.batch{batch}.run{run}.*.score"
    for sf in work_dir.glob(f"tests/*/{pattern}"):
        parts = sf.name.split(".")
        # filename: <test>.<src>-<tgt>.<tgt>.<model>.out.batch<b>.run<r>.<metric>.score
        # find the metric (second to last part before .score)
        metric = parts[-2]
        lang = sf.parent.name
        name_no_score = sf.name[: -len(f".{metric}.score")]
        # strip known suffixes to find model name
        suffix = f".out.batch{batch}.run{run}"
        base = name_no_score[: name_no_score.index(suffix)]
        # base = <test>.<src>-<tgt>.<tgt>.<model>
        base_parts = base.split(".")
        model = base_parts[-1]
        test = base_parts[0]
        scores[(model, lang, test, metric)] = float(sf.read_text().strip())
    return scores


def load_latency(work_dir: Path, batch: int, run: int):
    """Returns {(model, lang): wall_time_sec} averaged over non-warmup/blind test sets."""
    times = defaultdict(list)
    pattern = f"*.batch{batch}.run{run}.stats.json"
    for sf in work_dir.glob(f"tests/*/{pattern}"):
        test = sf.name.split(".")[0]
        if test in SKIP_TESTS:
            continue
        lang = sf.parent.name
        last_line = sf.read_text().strip().splitlines()[-1]
        d = json.loads(last_line)
        times[(d["model_name"], lang)].append(d["wall_time_sec"])
    return {k: sum(v) / len(v) for k, v in times.items()}


def print_quality(scores, batch, run):
    metrics = ["chrf", "wmt22-comet-da", "wmt22-cometkiwi-da", "wmt23-cometkiwi-da-xl"]
    lang_pairs = sorted({lang for (_, lang, _, _) in scores})
    models = sorted({model for (model, _, _, _) in scores})

    print(f"\n=== Quality Scores (test={QUALITY_TEST}, batch={batch}, run={run}) ===\n")
    header = f"{'lang-pair':<20} {'model':<12}" + "".join(f"  {m:>22}" for m in metrics)
    print(header)
    print("-" * len(header))
    for lang in lang_pairs:
        for model in models:
            row_scores = [scores.get((model, lang, QUALITY_TEST, m), float("nan")) for m in metrics]
            if all(s != s for s in row_scores):  # all NaN = no data
                continue
            cols = "".join(
                f"  {s:>22.4f}" if s == s else f"  {'N/A':>22}" for s in row_scores
            )
            print(f"{lang:<20} {model:<12}{cols}")
        print()


def print_latency(latency, batch, run):
    models = sorted({m for (m, _) in latency})
    lang_pairs = sorted({l for (_, l) in latency})

    print(f"=== Latency / wall_time_sec avg (batch={batch}, run={run}, excl. warmup+blind) ===\n")
    header = f"{'lang-pair':<20}" + "".join(f"  {m:>12}" for m in models)
    print(header)
    print("-" * len(header))
    for lang in lang_pairs:
        row = "".join(
            f"  {latency.get((m, lang), float('nan')):>11.1f}s" for m in models
        )
        print(f"{lang:<20}{row}")


def main():
    parser = argparse.ArgumentParser(description="Summarize quality + latency results")
    parser.add_argument("-w", "--work", type=Path, default=Path("workdir"))
    parser.add_argument("-b", "--batch", type=int, default=32)
    parser.add_argument("-r", "--run", type=int, default=2)
    args = parser.parse_args()

    scores = load_scores(args.work, args.batch, args.run)
    latency = load_latency(args.work, args.batch, args.run)

    if not scores and not latency:
        print(f"No data found for batch={args.batch} run={args.run} in {args.work}")
        return

    print_quality(scores, args.batch, args.run)
    print_latency(latency, args.batch, args.run)


if __name__ == "__main__":
    main()
