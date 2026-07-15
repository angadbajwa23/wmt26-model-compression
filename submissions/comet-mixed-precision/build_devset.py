#!/usr/bin/env python3
"""
Build a held-out dev subset for the COMET-guided layer-sensitivity sweep.

data/wmt25/wmt25.<file>.paragraphs.jsonl is the only reference-bearing data available
locally, but data_prep/build_calibration.py already folds ALL of it into
data/calibration/<pair>/calibration.jsonl as priority examples (see that script's
_load_wmt25 + stratified_sample: wmt25 records always win bucket slots over training
corpus examples). Reusing wmt25 wholesale for the sensitivity sweep would measure
quality on sentences the quantizer's calibration pass already saw.

This script excludes any wmt25 record whose reference text already appears in the
pair's calibration.jsonl (assistant-turn content is the exact, unmodified reference —
see build_calibration.py::_make_messages), then samples an even stride of the
remainder per pair.

Output: data/sensitivity-devset/<pair>/dev.jsonl
  Each line: {"pair": "...", "src": "...", "ref": "..."}
"""
import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
WMT25_DIR = REPO_ROOT / "data" / "wmt25"
CALIB_DIR = REPO_ROOT / "data" / "calibration"
OUT_DIR = REPO_ROOT / "data" / "sensitivity-devset"

# Mirrors data_prep/build_calibration.py::PAIR_CONF's wmt25_file mapping.
PAIR_TO_WMT25_FILE = {
    "ces-deu": "wmt25.cs-de_DE.paragraphs.jsonl",
    "eng-zho_Hans": "wmt25.en-zh_CN.paragraphs.jsonl",
    "eng-ara_EG": "wmt25.en-ar_EG.paragraphs.jsonl",
}

DEFAULT_N_PER_PAIR = 30
DEFAULT_SEED = 42


def _load_wmt25(path: Path) -> list[tuple[str, str]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            src = rec["src_text"].strip()
            ref = (rec.get("refs") or {}).get("refA")
            if isinstance(ref, dict):
                ref = ref.get("ref")
            if src and ref:
                out.append((src, ref.strip()))
    return out


def _calibration_ref_set(calib_path: Path) -> set[str]:
    """Exact reference texts already used to calibrate the quantizer for this pair."""
    refs = set()
    if not calib_path.exists():
        return refs
    with open(calib_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            for turn in rec["messages"]:
                if turn["role"] != "assistant":
                    continue
                content = turn["content"]
                text = "".join(b["text"] for b in content if b.get("type") == "text")
                refs.add(text.strip())
    return refs


def build_pair(pair: str, n_per_pair: int, rng: random.Random) -> list[dict]:
    wmt25_path = WMT25_DIR / PAIR_TO_WMT25_FILE[pair]
    calib_path = CALIB_DIR / pair / "calibration.jsonl"

    all_examples = _load_wmt25(wmt25_path)
    calibrated_refs = _calibration_ref_set(calib_path)

    held_out = [(s, r) for s, r in all_examples if r not in calibrated_refs]
    overlap = len(all_examples) - len(held_out)
    print(f"  {pair}: {len(all_examples)} wmt25 records, {overlap} already in calibration.jsonl, "
          f"{len(held_out)} held out")

    if len(held_out) < n_per_pair:
        print(f"  WARNING: only {len(held_out)} held-out records available for {pair}, "
              f"requested {n_per_pair}")

    rng.shuffle(held_out)
    step = max(1, len(held_out) // max(1, n_per_pair))
    selected = held_out[::step][:n_per_pair]

    return [{"pair": pair, "src": s, "ref": r} for s, r in selected]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a held-out (non-calibration) dev subset for the layer-sensitivity sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-per-pair", type=int, default=DEFAULT_N_PER_PAIR,
                        help="Number of held-out dev sentences to sample per language pair")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    total = 0
    for pair in PAIR_TO_WMT25_FILE:
        selected = build_pair(pair, args.n_per_pair, rng)
        pair_dir = args.out_dir / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        out_path = pair_dir / "dev.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in selected:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  -> wrote {len(selected)} records to {out_path}")
        total += len(selected)

    print(f"Total held-out dev sentences: {total}")


if __name__ == "__main__":
    main()
