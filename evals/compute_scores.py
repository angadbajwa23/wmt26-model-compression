#!/usr/bin/env python3
"""
WMT25 Model Compression Evaluation

This script walks an experiments root directory looking for subfolders named like
  evalXX-MODELNAME/
Inside each of those, it looks for language-pair folders named like
  SOURCE-TARGET/

Within each language-pair folder, it:
  • loads the TSV mapping file:  wmt25.SOURCE-TARGET.meta
      - column 1: segment_id (string)
      - column 2: line_number (int)
  • finds all system output files matching the pattern:
      wmt25.SOURCE-TARGET.TARGET.MODEL_TYPE.out.batchYY.runZ

It groups the outputs by (MODEL_TYPE, runZ, batchYY).

For each (MODEL_TYPE, runZ, batchYY) group, it reconstructs *document-level segments* by concatenating
all lines that belong to each segment_id in the order specified by the meta file. If a line
number in the meta is missing beyond the length of the concatenated system lines, an empty line
is inserted.

Finally, it computes COMET scores for each reconstructed segment using the *reference* text from
https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/refs/heads/main/data/wmt25-genmt.jsonl

Output: a CSV with one row per segment containing
  eval_dir, model_name, lang_pair, model_type, run, segment_id, metricx_score, comet_score

Usage:
  python compute_scores.py \
    --root /path/to/experiments \
    --jsonl https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/refs/heads/main/data/wmt25-genmt.jsonl \
    --comet-model Unbabel/XCOMET-XL \
    --metricx-model google/metricx-24-hybrid-xl-v2p6 \
    --out /path/to/output.csv
"""
import argparse
import csv
import dataclasses
import json
import os
import re
import statistics
import sys
import tempfile
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional

import torch
import datasets
import transformers
from metricx24 import models


EVAL_DIR_RE = re.compile(r"^eval(\d{2})-(.+)$")
LANGPAIR_RE = re.compile(r"^([a-z]+)-([a-z]+)$")


@dataclasses.dataclass
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


def load_references(jsonl_path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Loads source texts and references from a JSONL file.
    Returns two `Dict`s, the first one for source texts and the second one for references,
    where the keys are the document id.
    """

    def open_stream(p: str):
        if p.startswith("http://") or p.startswith("https://"):
            import urllib.request
            return urllib.request.urlopen(p)
        return open(p, "rb")

    refs: Dict[str, List[str]] = {}
    srcs: Dict[str, str] = {}
    with open_stream(jsonl_path) as fh:
        for bline in fh:
            if not bline.strip():
                continue
            obj = json.loads(bline)
            # segment id
            doc_id = obj["doc_id"]
            # source
            srcs[doc_id] = obj["src_text"]
            # reference
            refs[doc_id] = []
            for ref_k in obj["refs"].keys():
                refs[doc_id].append(obj["refs"][ref_k]["ref"])
    return srcs, refs


# -------------------------------
# Filesystem parsing
# -------------------------------

def find_eval_dirs(root: str) -> List[Tuple[str, str]]:
    out = []
    for name in os.listdir(root):
        m = EVAL_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            out.append((name, m.group(2)))  # (dirname, model_name)
    return sorted(out)


def find_langpair_dirs(eval_dir_path: str) -> List[str]:
    out = []
    for name in os.listdir(eval_dir_path):
        if os.path.isdir(os.path.join(eval_dir_path, name)) and LANGPAIR_RE.match(name):
            out.append(name)
    return sorted(out)


def parse_outfiles(
        langpair_path: str,
        source_lang: str,
        tgt_lang: str,
        model_name: str,
        run_id: Optional[int] = None) -> List[OutputFile]:
    files = []
    if run_id is None:
        run_regex = r"run(\d+)"
    else:
        run_regex = f"run{run_id}"
    OUTFILE_RE = re.compile(
        rf"^wmt25\.{source_lang}-{tgt_lang}\.{tgt_lang}\.([A-Za-z0-9_\-.]+)\.out\.batch(\d+)\.{run_regex}$"
    )
    for name in os.listdir(langpair_path):
        m = OUTFILE_RE.match(name)
        if not m:
            continue
        if run_id is None:
            model_type, batch_s, run_s = m.groups()
        else:
            model_type, batch_s = m.groups()
            run_s = run_id
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


# -------------------------------
# Segment reconstruction
# -------------------------------

def load_meta(meta_path: str) -> Dict[str, List[int]]:
    seg2lines: Dict[str, List[int]] = OrderedDict()
    with open(meta_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            seg_id = parts[0]
            idx = int(parts[1])
            if seg_id not in seg2lines:
                seg2lines[seg_id] = []
            seg2lines[seg_id].append(idx)
    return seg2lines


def read_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        return [ln.rstrip('\n') for ln in fh]


def reconstruct_segments(seg2lines: Dict[str, List[int]], lines: List[str]) -> Dict[str, str]:
    seg2text: Dict[str, str] = {}
    segment_lines_start_idx = 0
    for seg, idxs in seg2lines.items():
        segment_lines = lines[segment_lines_start_idx:segment_lines_start_idx + len(idxs)]
        buf: List[str] = []
        for i in range(max(idxs)):
            idx = i + 1
            if idx in idxs:
                buf.append(segment_lines[idxs.index(idx)])
            else:
                buf.append('')  # missing line -> empty line
        seg2text[seg] = "\n".join(buf)
        segment_lines_start_idx += len(idxs)
    return seg2text


class CometScore:
    def __init__(self, model_name: str = 'Unbabel/XCOMET-XL', batch_size: int = 8):
        from comet import download_model, load_from_checkpoint
        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)
        self.batch_size = batch_size

    def score(self, sys_texts: List[str], ref_texts: List[str], src_texts: List[str]) -> float:
        data = []
        for mt, ref, src in zip(sys_texts, ref_texts, src_texts):
            data.append({"src": src, "mt": mt, "ref": ref})
        seg_scores = self.model.predict(data, batch_size=self.batch_size, gpus=1)
        return seg_scores.system_score


class MetrixScore:
    def __init__(self, model_name: str = 'google/metricx-24-hybrid-xl-v2p6', batch_size: int = 8):
        self.model_name = model_name
        self.batch_size = batch_size
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")

        self.model = models.MT5ForRegression.from_pretrained(
            model_name, torch_dtype="auto"
        )

        self.model.to(self.device)
        self.model.eval()
        self.max_input_length = 1536

    def score(self, sys_texts: List[str], ref_texts: List[str], src_texts: List[str]) -> float:
        def _make_input(example):
            example["input"] = (
                    "source: "
                    + example["source"]
                    + " candidate: "
                    + example["hypothesis"]
                    + " reference: "
                    + example["reference"]
            )
            return example

        def _tokenize(example):
            return self.tokenizer(
                example["input"],
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
            )

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

        with tempfile.NamedTemporaryFile("w") as tmp:
            training_args = transformers.TrainingArguments(
                output_dir=os.path.dirname(tmp.name),
                per_device_eval_batch_size=self.batch_size,
                dataloader_pin_memory=False,
            )
            trainer = transformers.Trainer(
                model=self.model,
                args=training_args,
            )

            ds = datasets.Dataset.from_dict({
                "source": src_texts, "reference": ref_texts, "hypothesis": sys_texts})
            ds = ds.map(_make_input)
            ds = ds.map(_tokenize)
            ds = ds.map(_remove_eos)
            ds.set_format(
                type="torch",
                columns=["input_ids", "attention_mask"],
                device=self.device,
                output_all_columns=True,
            )
            predictions, _, _ = trainer.predict(test_dataset=ds)

        return statistics.mean(float(p) for p in predictions)


def main():
    ap = argparse.ArgumentParser(description="Build WMT25 segments from batched outputs and score with COMET.")
    ap.add_argument('--root', required=True, help='Root directory containing evalXX-MODELNAME folders')
    ap.add_argument('--jsonl', default="https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/refs/heads/main/data/wmt25-genmt.jsonl",
                    help='Path or URL to wmt25-genmt.jsonl containing refs (and possibly srcs)')
    ap.add_argument('--comet-model', default='Unbabel/XCOMET-XL', help='COMET model name')
    ap.add_argument('--metricx-model', default='google/metricx-24-hybrid-xl-v2p6', help='MetricX model name')
    ap.add_argument('--batch-size', type=int, default=1, help='COMET and MetricX batch size')
    ap.add_argument('--run', type=int, default=None, help='Which run number to evaluate')
    ap.add_argument('--save-jsonl', default=None, help='Optional path to save detailed per-segment results in JSONL')
    ap.add_argument('--out', required=True, help='Output CSV path')

    args = ap.parse_args()

    comet = CometScore(args.comet_model, args.batch_size)
    metrix = MetrixScore(args.metricx_model, args.batch_size)

    # Load references
    print(f"Loading references from: {args.jsonl}")
    srcs_map, refs_map = load_references(args.jsonl)
    if not refs_map or not srcs_map:
        sys.exit("[ERROR] No references loaded from JSONL. Check the URL/file and the field names.")

    rows_for_csv: List[Dict[str, Any]] = []

    for eval_dirname, model_name in find_eval_dirs(args.root):
        eval_dir_path = os.path.join(args.root, eval_dirname)
        for lp in find_langpair_dirs(eval_dir_path):
            src, tgt = lp.split('-', 1)
            langpair_path = os.path.join(eval_dir_path, lp)
            meta_name = f"wmt25.{src}-{tgt}.meta"
            meta_path = os.path.join(langpair_path, meta_name)
            if not os.path.exists(meta_path):
                sys.exit(f"[ERROR] Missing meta for {eval_dirname}/{lp}: {meta_name}")
            seg2lines = load_meta(meta_path)
            outfiles = parse_outfiles(langpair_path, src, tgt, model_name, args.run)
            if not outfiles:
                print(f"[WARN] No output files in {eval_dirname}/{lp}")
                continue

            for outfile in outfiles:
                sys_lines = read_lines(outfile.path)
                seg2text = reconstruct_segments(seg2lines, sys_lines)

                # Build aligned arrays for scoring
                src_texts = []
                hypo_texts = []
                ref_texts = []
                for k in seg2text.keys():
                    if k.split("#")[1] == "_testsuite_":
                        continue
                    splitted_hypo = seg2text[k].split("\n\n")
                    splitted_ref = refs_map[k][0].split("\n\n")
                    splitted_src = srcs_map[k].split("\n\n")
                    assert len(splitted_hypo) == len(splitted_ref) == len(splitted_src)
                    for h, r, s in zip(splitted_hypo, splitted_ref, splitted_src):
                        hypo_texts.append(h)
                        ref_texts.append(r)
                        src_texts.append(s)

                print(
                    f"Scoring {len(hypo_texts)} segments for {outfile} ...")
                comet_score = comet.score(hypo_texts, ref_texts, src_texts)
                metricx_score = metrix.score(hypo_texts, ref_texts, src_texts)

                rows_for_csv.append({
                    "model_name": outfile.model_name,
                    "model_type": outfile.model_type,
                    "lang_pair": outfile.lang_pair,
                    "batch": outfile.batch,
                    "run": outfile.run,
                    "metricx": metricx_score,
                    "comet": comet_score,
                })

    # Write CSV
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["model_name", "model_type", "lang_pair", "batch", "run", "comet", "metricx"])
        writer.writeheader()
        writer.writerows(rows_for_csv)
    print(f"Saved CSV: {args.out} ({len(rows_for_csv)} rows)")

    if args.save_jsonl:
        with open(args.save_jsonl, 'w', encoding='utf-8') as fh:
            for rec in rows_for_csv:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved JSONL: {args.save_jsonl} ({len(rows_for_csv)} records)")


if __name__ == '__main__':
    main()
