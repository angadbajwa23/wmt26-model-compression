#!/usr/bin/env python3
"""
Extract WMT26 General MT blindset records for the 3 model compression language pairs.

Outputs per pair (in --output-dir):
  wmt26.<pair>.paragraphs.jsonl  -- same schema as data/wmt25/*.paragraphs.jsonl
  wmt26.<pair>.src.txt           -- plain source lines, one per line, for run.sh

Language pair mapping and how source language is detected:
  ces-deu      tgt_lang=deu_Latn, Czech source (detected by Czech-specific characters)
  eng-zho_Hans tgt_lang in {zh_CN, zho_Hans}
  eng-ara_EG   tgt_lang=arz_Arab

Only text-only records are kept (multimodal_input_path must be null).
The raw blindset is cached at --cache so repeated runs skip the download.
"""
import argparse
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

BLINDSET_URL = "https://data.statmt.org/wmt26/wmt26_genmt_blindset.jsonl"

# Characters that appear in Czech but never in German — sufficient for source detection.
# ě, š, č, ř, ž, ů and their uppercase forms.
_CZECH_CHARS = frozenset("ěšřžůčĚŠŘŽŮČ")


def _is_czech(text: str) -> bool:
    return any(c in _CZECH_CHARS for c in text)


def _is_text_only(rec: dict) -> bool:
    return not rec.get("multimodal_input_path")


PAIR_FILTERS = {
    "ces-deu": lambda r: r["tgt_lang"] == "deu_Latn" and _is_czech(r["source_doc"]),
    "eng-zho_Hans": lambda r: r["tgt_lang"] in ("zh_CN", "zho_Hans"),
    "eng-ara_EG": lambda r: r["tgt_lang"] == "arz_Arab",
}


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached blindset: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    print(f"Saved: {dest}")
    return dest


def load(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_pair(records: list[dict], pair: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"wmt26.{pair}.paragraphs.jsonl"
    src_path = out_dir / f"wmt26.{pair}.src.txt"

    with open(jsonl_path, "w", encoding="utf-8") as jf, \
         open(src_path, "w", encoding="utf-8") as sf:
        for idx, rec in enumerate(records, start=1):
            # Flatten internal newlines — run.sh expects one translation per line
            src = rec["source_doc"].strip().replace("\n", " ")
            jf.write(json.dumps({
                "doc_id": rec["doc_id"],
                "paragraph_id": idx,
                "src_text": src,
                "refs": {},
            }, ensure_ascii=False) + "\n")
            sf.write(src + "\n")

    print(f"  {pair}: {len(records):4d} records  →  {jsonl_path.name}  {src_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=BLINDSET_URL, help="Blindset URL")
    parser.add_argument("--cache", type=Path, default=Path("data/wmt26/wmt26_genmt_blindset.jsonl"),
                        help="Local cache path for the raw blindset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/wmt26"),
                        help="Directory to write per-pair output files")
    parser.add_argument("--all-langs", action="store_true",
                        help="Print full tgt_lang breakdown from the blindset and exit")
    args = parser.parse_args()

    cache = download(args.url, args.cache)
    all_records = load(cache)
    print(f"Loaded {len(all_records):,} total records\n")

    if args.all_langs:
        counts = Counter(r["tgt_lang"] for r in all_records)
        for lang, n in sorted(counts.items()):
            mm = sum(1 for r in all_records if r["tgt_lang"] == lang and r.get("multimodal_input_path"))
            print(f"  {lang:20s}  total={n:5d}  multimodal={mm:4d}  text-only={n-mm:4d}")
        return

    print("Extracting text-only records per pair:")
    for pair, keep in PAIR_FILTERS.items():
        matched = [r for r in all_records if _is_text_only(r) and keep(r)]
        write_pair(matched, pair, args.output_dir)

    print("\nDone. Use these files for local evaluation:")
    print("  MODELZIP_DATA_DIR=data/wmt26  (see modelzip/config.py)")


if __name__ == "__main__":
    main()
