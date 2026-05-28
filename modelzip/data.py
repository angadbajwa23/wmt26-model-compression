import json
import logging
import shutil
import subprocess as sp
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable

LOG = logging.getLogger(__name__)


@dataclass
class CmdGetter:
    cmd: str

    def __call__(self) -> list[list[str]]:
        lines = sp.check_output(self.cmd, shell=True, text=True).strip().replace("\r", "").split("\n")
        return [line.strip().split("\t") for line in lines if line.strip()]


@dataclass
class WmtJsonlData:
    pair: str
    url: str
    cache_name: str

    CACHE_DIR: ClassVar[Path] = Path.home() / ".cache" / "wmt-modelzip"

    @property
    def cache_file(self) -> Path:
        source = Path(self.url).expanduser()
        cache_file = self.CACHE_DIR / self.cache_name
        if source.exists():
            return source
        if cache_file.exists() and cache_file.stat().st_size > 0:
            return cache_file

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
        LOG.info("Downloading %s to %s", self.url, cache_file)
        with urllib.request.urlopen(self.url) as response, open(tmp_file, "wb") as out:
            shutil.copyfileobj(response, out)
        tmp_file.replace(cache_file)
        return cache_file

    def records(self) -> Iterable[dict]:
        seen_pairs = set()
        count = 0
        with open(self.cache_file, "r", encoding="utf-8") as lines:
            for line in lines:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec_pair = f"{rec['src_lang']}-{rec['tgt_lang']}"
                seen_pairs.add(rec_pair)
                if rec_pair == self.pair:
                    count += 1
                    yield rec
        if count == 0:
            LOG.warning("No records found for %s. Seen pairs: %s", self.pair, sorted(seen_pairs))

    @staticmethod
    def _split_text(text: str) -> list[str]:
        paragraphs = [part.strip().replace("\n", " ") for part in text.strip().split("\n\n")]
        return [part for part in paragraphs if part]

    @staticmethod
    def _get_reference(rec: dict) -> str | None:
        refs = rec.get("refs") or {}
        if "refA" in refs:
            ref = refs["refA"]
            return ref.get("ref") if isinstance(ref, dict) else ref

        tgt_text = rec.get("tgt_text") or {}
        ref = tgt_text.get("refA")
        return ref if isinstance(ref, str) else None

    def __call__(self) -> list[list[str | None]]:
        rows = []
        for rec in self.records():
            src_parts = self._split_text(rec["src_text"])
            ref_text = self._get_reference(rec)
            ref_parts = self._split_text(ref_text) if ref_text else [None] * len(src_parts)
            if len(ref_parts) != len(src_parts):
                LOG.warning(
                    "Reference/source paragraph mismatch for %s: src=%d ref=%d; dropping refs for this doc",
                    rec["doc_id"],
                    len(src_parts),
                    len(ref_parts),
                )
                ref_parts = [None] * len(src_parts)

            for idx, (src, ref) in enumerate(zip(src_parts, ref_parts), start=1):
                rows.append([src, ref, f"{rec['doc_id']}\t{idx}"])
        return rows


class Wmt25BlindData(WmtJsonlData):
    URL = "https://data.statmt.org/wmt25/general-mt/wmt25.jsonl"

    def __init__(self, pair: str, url: str | None = None):
        super().__init__(pair=pair, url=url or self.URL, cache_name="wmt25.jsonl")


class Wmt25ReferenceData(WmtJsonlData):
    URL = "https://github.com/wmt-conference/wmt25-general-mt/raw/refs/heads/main/data/wmt25-genmt.jsonl"

    def __init__(self, pair: str, url: str | None = None):
        super().__init__(pair=pair, url=url or self.URL, cache_name="wmt25-genmt.jsonl")


@dataclass
class LocalParagraphData:
    """Read pre-split paragraph-level JSONL files (doc_id, paragraph_id, src_text, refs)."""

    path: str | Path

    def __call__(self) -> list[list[str | None]]:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Local data file not found: {path}")
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                src = rec["src_text"].replace("\n", " ").strip()
                refs = rec.get("refs") or {}
                ref = refs.get("refA")
                if isinstance(ref, dict):
                    ref = ref.get("ref")
                if isinstance(ref, str):
                    ref = ref.replace("\n", " ").strip()
                meta = f"{rec['doc_id']}\t{rec.get('paragraph_id', 0)}"
                rows.append([src, ref, meta])
        return rows
