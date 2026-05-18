#!/usr/bin/env python3
"""
WMT25 Model Compression Evaluation

This script walks an experiments root directory looking for subfolders named like
  evalXX-MODELNAME/
Inside each of those, it looks for language-pair folders named like
  SOURCE-TARGET/

Within each language-pair folder, it finds all system output files matching the pattern:
      wmt25.SOURCE-TARGET.TARGET.MODEL_TYPE.out.batchYY.runZ

Then it computes the character ratio with the source of the generated outputs, and outputs
plots to show the distribution or outputs which segments are hallucinated (determined by a
threshold). Lastly, it builds a test set that contains no hallucinated segments in any output.

Usage:
  python hallucinations_checks.py \
    --root /path/to/experiments \
    --out /path/to/output.txt
"""
import argparse
import csv
import dataclasses
import os
import re
import sys
from typing import List, Tuple


EVAL_DIR_RE = re.compile(r"^eval(\d{2})-(.+)$")


@dataclasses.dataclass(frozen=True)
class OutputFile:
    model_name: str
    src: str
    tgt: str
    model_type: str
    run: int
    batch: int
    path: str

    @property
    def lang_pair(self) -> str:
        return f"{self.src}-{self.tgt}"


def find_eval_dirs(root: str) -> List[Tuple[str, str]]:
    out = []
    for name in os.listdir(root):
        m = EVAL_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            out.append((name, m.group(2)))  # (dirname, model_name)
    return sorted(out)


def find_langpair_dirs(eval_dir_path: str, lang_pair: str) -> List[str]:
    out = []
    for name in os.listdir(eval_dir_path):
        if name == lang_pair and os.path.isdir(os.path.join(eval_dir_path, name)):
            out.append(name)
    return sorted(out)


def parse_outfiles(langpair_path: str, source_lang: str, tgt_lang: str, model_name: str) -> List[OutputFile]:
    files = []
    OUTFILE_RE = re.compile(
        rf"^wmt25\.{source_lang}-{tgt_lang}\.{tgt_lang}\.([A-Za-z0-9_\-.]+)\.out\.batch(\d+)\.run(\d+)$"
    )
    for name in os.listdir(langpair_path):
        m = OUTFILE_RE.match(name)
        if not m:
            continue
        model_type, batch_s, run_s = m.groups()
        files.append(OutputFile(
            model_name=model_name,
            src=source_lang,
            tgt=tgt_lang,
            model_type=model_type,
            run=int(run_s),
            batch=int(batch_s),
            path=os.path.join(langpair_path, name),
        ))
    return sorted(files, key=lambda f: (f.model_name, f.model_type, f.src, f.tgt, f.run, f.batch))


def read_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        return [ln.rstrip('\n') for ln in fh]


def main():
    ap = argparse.ArgumentParser(description="Check hallucinations in WMT25 outputs.")
    ap.add_argument('--root', required=True, help='Root directory containing evalXX-MODELNAME folders')
    ap.add_argument('--lang-pair', default="ces-deu", help='Input/output languages (default: ces-deu)')
    ap.add_argument('--plot', action='store_true', default=False, help='Plot distributions')
    ap.add_argument('--threshold', type=float, default=2.0, help='Character ratio threshold')
    ap.add_argument('--out', required=True,
                    help='Output path for the txt file with source segments that do not hallucinate in any output')

    args = ap.parse_args()
    all_ratios = []
    hallucinated_segments = set()
    num_processed_outputs = 0
    model_stats = {}

    for eval_dirname, model_name in find_eval_dirs(args.root):
        eval_dir_path = os.path.join(args.root, eval_dirname)
        for lp in find_langpair_dirs(eval_dir_path, args.lang_pair):
            src, tgt = lp.split('-', 1)
            langpair_path = os.path.join(eval_dir_path, lp)
            input_name = f"wmt25.{src}-{tgt}.{src}"
            input_path = os.path.join(langpair_path, input_name)
            if not os.path.exists(input_path):
                sys.exit(f"[ERROR] Missing input for {eval_dirname}/{lp}: {input_path}")
            input_lines = read_lines(input_path)

            meta_name = f"wmt25.{src}-{tgt}.meta"
            meta_path = os.path.join(langpair_path, meta_name)
            meta_lines = read_lines(meta_path)
            if len(input_lines) != len(meta_lines):
                sys.exit(f"[ERROR] Mismatching lines for {input_path} and {meta_path}")

            outfiles = parse_outfiles(langpair_path, src, tgt, model_name)
            if not outfiles:
                print(f"[WARN] No output files in {eval_dirname}/{lp}")
                continue

            for outfile in outfiles:
                out_lines = read_lines(outfile.path)
                if len(out_lines) != len(input_lines):
                    print(
                        f"[WARN] Input and output lines are not matching "
                        f"({len(out_lines)}-{len(input_lines)}). Skipping {outfile.path}")
                    continue
                model_stats[outfile] = 0
                for line_id, (in_line, out_line) in enumerate(zip(input_lines, out_lines)):
                    ratio = len(out_line) / len(in_line)
                    all_ratios.append(ratio)
                    if ratio > args.threshold:
                        hallucinated_segments.add(line_id)
                        model_stats[outfile] += 1
                num_processed_outputs += 1

    if args.plot:
        import matplotlib.pyplot as plt
        import numpy as np

        # Cap values at 3.0
        capped = np.clip(all_ratios, None, 3.0)

        # Plot histogram
        plt.hist(capped, bins=50, edgecolor="black")
        plt.xlabel("Output/source character ratio (capped at 3.0)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Character Ratios")
        plt.show()

    print(
        f"[INFO] {len(hallucinated_segments)} hallucinated segments found out of "
        f"{len(input_lines)} from {num_processed_outputs} output files.")

    with open(args.out, 'w', encoding='utf-8') as f:
        for i, line in enumerate(input_lines):
            if i not in hallucinated_segments:
                f.write(line + "\n")
    with open(args.out + '.meta', 'w', encoding='utf-8') as f:
        for i, line in enumerate(meta_lines):
            if i not in hallucinated_segments:
                f.write(line + "\n")

    with open(args.out + '.stats', 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=["model_name", "model_type", "lang_pair", "batch", "run", "n_hall"])
        writer.writeheader()
        for outfile in model_stats.keys():
            writer.writerow({
                "model_name": outfile.model_name,
                "model_type": outfile.model_type,
                "lang_pair": outfile.lang_pair,
                "batch": outfile.batch,
                "run": outfile.run,
                "n_hall": model_stats[outfile]
            })


if __name__ == '__main__':
    main()
