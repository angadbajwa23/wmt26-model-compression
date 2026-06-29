#!/usr/bin/env python3
"""
Convert mtdata recipe output to paragraph-level JSONL matching the wmt25 format.

mtdata get-recipe --no-merge produces per-dataset files under:
  <recipe_dir>/train/<dataset_name>.{src,tgt}[.gz]  (separate files)
  or <recipe_dir>/train/<dataset_name>.tsv[.gz]      (tab-separated)

Output format per line (same as data/wmt25/*.paragraphs.jsonl):
  {"doc_id": "...", "paragraph_id": 1, "src_text": "...", "refs": {"refA": "..."}}

Usage:
  python3 data_prep/convert_mtdata_to_jsonl.py \\
      --input-dir data/wmt26-train \\
      --output-dir data/wmt26

The script auto-detects language pairs from recipe directory names and writes:
  data/wmt26/wmt26.cs-de_DE.paragraphs.jsonl
  data/wmt26/wmt26.en-zh_CN.paragraphs.jsonl
  data/wmt26/wmt26.en-ar_EG.paragraphs.jsonl
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

# Map recipe language-pair patterns -> output filename suffix (matching wmt25 naming)
LANG_PAIR_MAP = {
    ("ces", "deu"): "cs-de_DE",
    ("deu", "ces"): "cs-de_DE",  # same file, will flip src/tgt
    ("eng", "zho_Hans"): "en-zh_CN",
    ("zho_Hans", "eng"): "en-zh_CN",
    ("eng", "ara_EG"): "en-ar_EG",
    ("ara_EG", "eng"): "en-ar_EG",
}

# Canonical src language for each pair (so we always write src->tgt correctly)
CANONICAL_SRC = {
    "cs-de_DE": "ces",
    "en-zh_CN": "eng",
    "en-ar_EG": "eng",
}


def open_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def detect_lang_pair(recipe_dir: Path) -> tuple[str, str] | None:
    """Infer (src_lang, tgt_lang) from recipe directory name."""
    name = recipe_dir.name.lower()
    for src, tgt in LANG_PAIR_MAP:
        if src.lower() in name and tgt.lower() in name:
            return src, tgt
    return None


def iter_bitext(dataset_dir: Path):
    """Yield (src_line, tgt_line) pairs from a dataset directory."""
    # Try TSV format first
    for tsv in sorted(dataset_dir.glob("*.tsv")) + sorted(dataset_dir.glob("*.tsv.gz")):
        with open_file(tsv) as fh:
            for line in fh:
                line = line.rstrip("\n")
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    yield parts[0].strip(), parts[1].strip()
        return

    # Try separate src/tgt files
    src_files = sorted(dataset_dir.glob("*.src")) + sorted(dataset_dir.glob("*.src.gz"))
    tgt_files = sorted(dataset_dir.glob("*.tgt")) + sorted(dataset_dir.glob("*.tgt.gz"))

    # Match by stem name
    src_map = {f.name.replace(".src.gz", "").replace(".src", ""): f for f in src_files}
    tgt_map = {f.name.replace(".tgt.gz", "").replace(".tgt", ""): f for f in tgt_files}

    for stem in sorted(src_map):
        if stem not in tgt_map:
            print(f"  WARNING: no matching tgt for {src_map[stem].name}", file=sys.stderr)
            continue
        src_lines = open_file(src_map[stem]).read().splitlines()
        tgt_lines = open_file(tgt_map[stem]).read().splitlines()
        if len(src_lines) != len(tgt_lines):
            print(
                f"  WARNING: line count mismatch in {stem}: "
                f"src={len(src_lines)} tgt={len(tgt_lines)}, skipping",
                file=sys.stderr,
            )
            continue
        yield from zip(src_lines, tgt_lines)


def convert_recipe(recipe_dir: Path, output_file: Path, flip: bool) -> int:
    """Convert one recipe directory to JSONL. Returns number of records written."""
    train_dir = recipe_dir / "train"
    if not train_dir.exists():
        print(f"  WARNING: no train/ subdir in {recipe_dir}, skipping", file=sys.stderr)
        return 0

    count = 0
    recipe_name = recipe_dir.name

    with open(output_file, "a", encoding="utf-8") as out:
        for dataset_path in sorted(train_dir.iterdir()):
            if not dataset_path.is_dir():
                continue
            dataset_name = dataset_path.name
            pair_count = 0
            for src_line, tgt_line in iter_bitext(dataset_path):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if not src_line or not tgt_line:
                    continue
                if flip:
                    src_line, tgt_line = tgt_line, src_line
                record = {
                    "doc_id": f"{recipe_name}_{dataset_name}_{count + 1}",
                    "paragraph_id": 1,
                    "src_text": src_line,
                    "refs": {"refA": tgt_line},
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                pair_count += 1
            if pair_count:
                print(f"    {dataset_name}: {pair_count:,} pairs", file=sys.stderr)

    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path, help="Directory with downloaded recipe subdirs")
    ap.add_argument("--output-dir", required=True, type=Path, help="Directory to write JSONL files")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Group recipe dirs by output pair
    pair_recipes: dict[str, list[tuple[Path, bool]]] = {}
    for recipe_dir in sorted(args.input_dir.iterdir()):
        if not recipe_dir.is_dir():
            continue
        detected = detect_lang_pair(recipe_dir)
        if detected is None:
            print(f"Skipping {recipe_dir.name} (language pair not recognized)", file=sys.stderr)
            continue
        src_lang, tgt_lang = detected
        pair_key = LANG_PAIR_MAP[(src_lang, tgt_lang)]
        canonical_src = CANONICAL_SRC[pair_key]
        flip = src_lang != canonical_src
        pair_recipes.setdefault(pair_key, []).append((recipe_dir, flip))

    if not pair_recipes:
        print("No matching recipe directories found.", file=sys.stderr)
        sys.exit(1)

    for pair_key, recipes in sorted(pair_recipes.items()):
        output_file = args.output_dir / f"wmt26.{pair_key}.paragraphs.jsonl"
        # Clear existing file
        output_file.write_text("", encoding="utf-8")
        total = 0
        print(f"\n=== {pair_key} -> {output_file.name} ===", file=sys.stderr)
        for recipe_dir, flip in recipes:
            print(f"  Recipe: {recipe_dir.name} (flip={flip})", file=sys.stderr)
            n = convert_recipe(recipe_dir, output_file, flip)
            total += n
        print(f"  Total: {total:,} records written", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
