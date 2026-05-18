import logging as LOG
import os
from pathlib import Path

from .data import CmdGetter, Wmt25BlindData, Wmt25ReferenceData, WmtJsonlData
from .submission import DEF_BATCH_SIZE, DEF_LANG_PAIRS, LANG_PAIR_ALIASES, normalize_lang_pair


LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HF_CACHE = Path(os.getenv("HF_HOME", default=Path.home() / ".cache" / "huggingface")) / "hub"
WORK_DIR = "./workdir"

# reusing hf cache for data; not perfect but for simpler config sake
WmtJsonlData.CACHE_DIR = HF_CACHE / "wmt26-modelzip"

WMT25_BLIND_URL = os.getenv("MODELZIP_WMT25_BLIND_URL", Wmt25BlindData.URL)
WMT25_REF_URL = os.getenv("MODELZIP_WMT25_REF_URL", Wmt25ReferenceData.URL)
WMT26_DATA_URL = os.getenv("MODELZIP_WMT26_DATA_URL", "")

TASK_CONF = {
    "langs": {
        "ces-deu": {
            "warmup": CmdGetter("printf 'ahoj světe\tHallo Welt\n'"),
            "wmt25-blind": Wmt25BlindData("cs-de_DE", url=WMT25_BLIND_URL),
            "wmt25-ref": Wmt25ReferenceData("cs-de_DE", url=WMT25_REF_URL),
        },
        "eng-zho_Hans": {
            "warmup": CmdGetter("printf 'hello world\t你好，世界\n'"),
            "wmt25-blind": Wmt25BlindData("en-zh_CN", url=WMT25_BLIND_URL),
            "wmt25-ref": Wmt25ReferenceData("en-zh_CN", url=WMT25_REF_URL),
        },
        "eng-ara_EG": {
            "warmup": CmdGetter("printf 'hello world\tمرحبا بالعالم\n'"),
            "wmt25-blind": Wmt25BlindData("en-ar_EG", url=WMT25_BLIND_URL),
            "wmt25-ref": Wmt25ReferenceData("en-ar_EG", url=WMT25_REF_URL),
        },
    },
    "metrics": ["chrf", "wmt22-comet-da"],  # "wmt22-cometkiwi-da" is a gated model
}
