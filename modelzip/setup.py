#!/usr/bin/env python
"""Prepare local evaluation data for WMT26 model-compression runs."""

import argparse
import logging as LOG
from pathlib import Path

from modelzip.config import DEF_LANG_PAIRS, TASK_CONF, WORK_DIR, normalize_lang_pair

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def setup_eval(work_dir: Path, langs=None):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = work_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    langs = [normalize_lang_pair(lang) for lang in (langs or DEF_LANG_PAIRS)]
    for lang_pair in langs:
        src, tgt = lang_pair.split("-")
        lang_dir = tests_dir / lang_pair
        lang_dir.mkdir(parents=True, exist_ok=True)
        for test_name, get_fn in TASK_CONF["langs"][lang_pair].items():
            src_file = lang_dir / f"{test_name}.{src}-{tgt}.{src}"
            ref_file = lang_dir / f"{test_name}.{src}-{tgt}.{tgt}"
            meta_file = lang_dir / f"{test_name}.{src}-{tgt}.meta"
            if src_file.exists() and src_file.stat().st_size > 0 and (ref_file.exists() or meta_file.exists()):
                LOG.info("Test files exist for %s:%s", lang_pair, test_name)
                continue
            LOG.info("Fetching %s via: %s", test_name, get_fn)
            lines = get_fn()
            assert isinstance(lines, list), f"Expected list of lines, got {type(lines)}"
            assert len(lines) > 0, f"No lines returned for {test_name} in {lang_pair}"
            if isinstance(lines[0], str):
                src_file.write_text("\n".join(lines), encoding="utf-8")
                LOG.info("Created source file %s; refs are missing", src_file)
            elif isinstance(lines[0], (list, tuple)):
                n_fields = len(lines[0])
                srcs = [x[0] for x in lines]
                src_file.write_text("\n".join(srcs), encoding="utf-8")
                LOG.info("Created source file %s", src_file)
                if n_fields > 1:
                    refs = [x[1] for x in lines]
                    if all(ref is None for ref in refs):
                        LOG.info("Refs are missing")
                    else:
                        assert all(ref is not None for ref in refs), "Some references are None"
                        ref_file.write_text("\n".join(refs), encoding="utf-8")
                        LOG.info("Created ref file %s", ref_file)
                if n_fields > 2:
                    meta = [x[2] for x in lines]
                    meta_file.write_text("\n".join(meta), encoding="utf-8")
                    LOG.info("Created meta file %s", meta_file)
            else:
                raise TypeError(f"Unexpected line format {type(lines[0])} for {test_name} in {lang_pair}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare WMT26 model-compression evaluation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-w", "--work", type=Path, default=WORK_DIR, help="Work directory")
    parser.add_argument("-l", "--langs", nargs="+", help="Language pairs to setup")
    parser.add_argument(
        "-t",
        "--task",
        choices=["eval"],
        default="eval",
        help="Compatibility option; root setup only prepares evaluation data",
    )
    args = parser.parse_args()
    setup_eval(work_dir=args.work, langs=args.langs)


if __name__ == "__main__":
    main()
