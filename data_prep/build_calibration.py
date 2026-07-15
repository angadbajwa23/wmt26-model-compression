#!/usr/bin/env python3
"""
Build GPTQ/AWQ calibration data for the 3 WMT26 model compression language pairs.

Per pair (--n-per-pair examples, split evenly across 3 length buckets):
  1. WMT25 dev paragraphs — highest priority, bypass length filter
  2. High-value training corpora — fill remaining bucket slots

Preprocessing:
  - Exact deduplication on normalized source text
  - Length filter on source: 10–120 whitespace-split words (training data only)
  - Stratified sampling across short / medium / long buckets

Output: data/calibration/<pair>/calibration.jsonl
  Each line: {"pair": "...", "messages": [user_turn, assistant_turn]}
  where each turn uses Gemma3 content-block format (list with {"type":"text","text":"..."}).

To tokenize in a quantization script:
  msgs = [json.loads(l)["messages"] for l in open("calibration.jsonl")]
  inputs = processor.apply_chat_template(
      msgs, tokenize=True, return_dict=True,
      return_tensors="pt", add_generation_prompt=False, padding=True,
  )
"""

import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modelzip.submission import make_translation_prompt

TRAIN_DIR = Path("data/wmt26-train")
WMT25_DIR = Path("data/wmt25")
OUT_DIR = Path("data/calibration")

# (min_words, max_words) applied to training data source; wmt25 dev bypasses this
TRAIN_LEN_BOUNDS = (10, 120)

# Length bucket boundaries (words). Targets are computed at runtime from --n-per-pair.
BUCKET_BOUNDS = [
    (10,  30),   # short
    (30,  60),   # medium
    (60, 999),   # long  (999 = no upper cap; wmt25 paragraphs land here)
]

PAIR_CONF = {
    "ces-deu": {
        "train_dir":   "ces-deu",
        "src_ext":     "ces",
        "tgt_ext":     "deu",
        "wmt25_file":  "wmt25.cs-de_DE.paragraphs.jsonl",
        "prompt_pair": "ces-deu",
        "corpora": [
            "Statmt-news_commentary-18.1-ces-deu",
            "OPUS-neulab_tedtalks-v1-ces-deu",
            "OPUS-qed-v2.0a-ces-deu",
        ],
    },
    "eng-zho_Hans": {
        "train_dir":   "eng-zho",
        "src_ext":     "eng",
        "tgt_ext":     "zho",
        "wmt25_file":  "wmt25.en-zh_CN.paragraphs.jsonl",
        "prompt_pair": "eng-zho_Hans",
        "corpora": [
            "Statmt-news_commentary-18.1-eng-zho",
            "OPUS-ted2013-v1.1-eng-zho",
            "OPUS-ted2020-v1-eng-zho",
            "OPUS-qed-v2.0a-eng-zho",
        ],
    },
    "eng-ara_EG": {
        "train_dir":   "eng-ara",
        "src_ext":     "eng",
        "tgt_ext":     "ara",
        "wmt25_file":  "wmt25.en-ar_EG.paragraphs.jsonl",
        "prompt_pair": "eng-ara_EG",
        "corpora": [
            "OPUS-qed-v2.0a-ara-eng",
            "Statmt-news_commentary-18.1-ara-eng",
            "Statmt-tedtalks-2_clean-eng-ara",
            "OPUS-ted2020-v1-ara-eng",
            "OPUS-neulab_tedtalks-v1-ara-eng",
        ],
    },
}


def _bucket_targets(n: int) -> list[int]:
    """Split n as evenly as possible across the 3 buckets, extra goes to the last."""
    base, remainder = divmod(n, len(BUCKET_BOUNDS))
    return [base] * (len(BUCKET_BOUNDS) - remainder) + [base + 1] * remainder


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_wmt25(conf: dict) -> list[tuple[str, str, bool, str]]:
    """Return (src, ref, is_priority=True, source) for all wmt25 dev examples with refs."""
    path = WMT25_DIR / conf["wmt25_file"]
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            src = rec["src_text"].strip()
            ref = (rec.get("refs") or {}).get("refA")
            if isinstance(ref, dict):
                ref = ref.get("ref")
            if src and ref:
                out.append((src, ref.strip(), True, "wmt25-dev"))
    return out


def _load_corpus(conf: dict, corpus: str) -> list[tuple[str, str, bool, str]]:
    """Return (src, ref, is_priority=False, source) from a gzipped parallel corpus."""
    parts = TRAIN_DIR / conf["train_dir"] / "train-parts"
    src_path = parts / f"{corpus}.{conf['src_ext']}.gz"
    tgt_path = parts / f"{corpus}.{conf['tgt_ext']}.gz"
    if not src_path.exists() or not tgt_path.exists():
        print(f"    WARNING: {corpus} not found, skipping")
        return []
    out = []
    with gzip.open(src_path, "rt", encoding="utf-8") as sf, \
         gzip.open(tgt_path, "rt", encoding="utf-8") as tf:
        for src_line, tgt_line in zip(sf, tf):
            src, tgt = src_line.strip(), tgt_line.strip()
            if src and tgt:
                out.append((src, tgt, False, corpus))
    return out


# ---------------------------------------------------------------------------
# Filtering & sampling
# ---------------------------------------------------------------------------

def _dedup_key(text: str) -> str:
    return hashlib.md5(" ".join(text.lower().split()).encode()).hexdigest()


def _word_count(text: str) -> int:
    return len(text.split())


def _bucket_idx(n_words: int) -> int | None:
    for i, (lo, hi) in enumerate(BUCKET_BOUNDS):
        if lo <= n_words < hi:
            return i
    return None


def deduplicate(examples: list[tuple[str, str, bool, str]]) -> list[tuple[str, str, bool, str]]:
    seen: set[str] = set()
    out = []
    for src, tgt, pri, src_name in examples:
        key = _dedup_key(src)
        if key not in seen:
            seen.add(key)
            out.append((src, tgt, pri, src_name))
    return out


def length_filter(examples: list[tuple[str, str, bool, str]]) -> list[tuple[str, str, bool, str]]:
    lo, hi = TRAIN_LEN_BOUNDS
    out = []
    for src, tgt, pri, src_name in examples:
        # wmt25 priority examples bypass the length filter
        if pri or (lo <= _word_count(src) <= hi):
            out.append((src, tgt, pri, src_name))
    return out


def stratified_sample(
    examples: list[tuple[str, str, bool, str]],
    n_per_pair: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    from collections import Counter
    targets = _bucket_targets(n_per_pair)

    buckets: list[list] = [[] for _ in BUCKET_BOUNDS]
    for ex in examples:
        idx = _bucket_idx(_word_count(ex[0]))
        if idx is not None:
            buckets[idx].append(ex)

    selected = []
    source_counts: Counter = Counter()
    for i, ((lo, hi), target_n) in enumerate(zip(BUCKET_BOUNDS, targets)):
        pool = buckets[i]
        priority = [ex for ex in pool if ex[2]]
        rest = [ex for ex in pool if not ex[2]]
        rng.shuffle(rest)
        chosen = (priority + rest)[:target_n]
        print(f"    bucket {i} ({lo:3d}–{hi:3d}w): {len(chosen):3d}/{target_n}")
        for ex in chosen:
            source_counts[ex[3]] += 1
        selected.extend((s, t) for s, t, _, __ in chosen)

    print(f"  Per-source breakdown:")
    for src_name, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {count:4d}  {src_name}")

    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _make_messages(prompt_pair: str, src: str, ref: str) -> list[dict]:
    """Gemma3 content-block format: user prompt + assistant reference."""
    prompt = make_translation_prompt(prompt_pair, src)
    return [
        {"role": "user",      "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": ref}]},
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_pair(pair: str, conf: dict, n_per_pair: int, rng: random.Random) -> list[dict]:
    print(f"  Loading sources:")
    all_examples: list[tuple[str, str, bool]] = []

    wmt25 = _load_wmt25(conf)
    print(f"    {len(wmt25):6,}  wmt25-dev (priority)")
    all_examples.extend(wmt25)

    for corpus in conf["corpora"]:
        loaded = _load_corpus(conf, corpus)
        print(f"    {len(loaded):6,}  {corpus}")
        all_examples.extend(loaded)

    before = len(all_examples)
    all_examples = deduplicate(all_examples)
    all_examples = length_filter(all_examples)
    print(f"  After dedup + length filter: {len(all_examples):,} / {before:,}")

    print(f"  Stratified sampling (target {n_per_pair}):")
    sampled = stratified_sample(all_examples, n_per_pair, rng)
    print(f"  Total selected: {len(sampled)}")

    return [
        {"pair": pair, "messages": _make_messages(conf["prompt_pair"], src, ref)}
        for src, ref in sampled
    ]


class _Tee:
    """Write to stdout and a log file simultaneously."""
    def __init__(self, log_fh):
        self._stdout = sys.stdout
        self._log = log_fh

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._log.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._log.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-per-pair", type=int, default=256,
        help="Number of calibration examples per language pair (default: 256)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--pairs", nargs="+", default=list(PAIR_CONF), metavar="PAIR",
        help=f"Pairs to build (default: all). Choices: {list(PAIR_CONF)}",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "build_log.txt"

    with open(log_path, "w", encoding="utf-8") as log_fh:
        sys.stdout = _Tee(log_fh)
        try:
            rng = random.Random(args.seed)

            for pair in args.pairs:
                if pair not in PAIR_CONF:
                    sys.exit(f"Unknown pair {pair!r}, choices: {list(PAIR_CONF)}")

                print(f"\n{'='*60}\n{pair}")
                records = build_pair(pair, PAIR_CONF[pair], args.n_per_pair, rng)

                out_dir = args.output_dir / pair
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "calibration.jsonl"
                with open(out_path, "w", encoding="utf-8") as fh:
                    for rec in records:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  Saved: {out_path}")

            print("\nDone.")
        finally:
            sys.stdout = sys.__stdout__

    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
