from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO

DEF_BATCH_SIZE = 1
DEF_MAX_NEW_TOKENS = int(os.getenv("MODEL_MAX_NEW_TOKENS", "1024"))
DEF_MAX_NEW_TOKENS_OVER_INPUT = int(os.getenv("MODEL_MAX_NEW_TOKENS_OVER_INPUT", "64"))

DEF_LANG_PAIRS = ["ces-deu", "eng-zho_Hans", "eng-ara_EG"]

LANGS_MAP = {
    "ces": "Czech",
    "deu": "German",
    "zho_Hans": "Simplified Chinese",
    "zh_CN": "Simplified Chinese",
    "eng": "English",
    "ara_EG": "Egyptian Arabic",
    "ar_EG": "Egyptian Arabic",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "zh": "Simplified Chinese",
    "ar": "Egyptian Arabic",
}

LANG_PAIR_ALIASES = {
    "cs-de": "ces-deu",
    "cs-de_DE": "ces-deu",
    "ces-deu": "ces-deu",
    "en-zh": "eng-zho_Hans",
    "en-zh_CN": "eng-zho_Hans",
    "eng-zho": "eng-zho_Hans",
    "eng-zho_Hans": "eng-zho_Hans",
    "en-ar": "eng-ara_EG",
    "en-ar_EG": "eng-ara_EG",
    "eng-ara": "eng-ara_EG",
    "eng-ara_EG": "eng-ara_EG",
}

TRANSLATE_PROMPT = (
    "Translate the following text from {src} to {tgt}. "
    "Return only the translation, with no explanation, labels, or quotes.\n\n"
    "{text}\n"
)


def normalize_lang_pair(pair: str) -> str:
    normalized = LANG_PAIR_ALIASES.get(pair, pair)
    if normalized not in DEF_LANG_PAIRS:
        known = ", ".join(sorted(DEF_LANG_PAIRS))
        aliases = ", ".join(sorted(LANG_PAIR_ALIASES))
        raise ValueError(f"Unsupported language pair {pair!r}. Known pairs: {known}. Aliases: {aliases}")
    return normalized


def language_names(pair: str) -> tuple[str, str]:
    src, tgt = normalize_lang_pair(pair).split("-")
    return LANGS_MAP[src], LANGS_MAP[tgt]


def make_translation_prompt(pair: str, text: str, template: str = TRANSLATE_PROMPT) -> str:
    src, tgt = language_names(pair)
    return template.format(src=src, tgt=tgt, text=text)


def read_source_lines(input_file: TextIO) -> list[str]:
    lines = input_file.read().splitlines()
    if not lines:
        raise ValueError("Input file is empty. Please provide some input.")
    return lines


def write_output_lines(output_file: TextIO, lines: Iterable[str]) -> None:
    output_file.write("\n".join(line.replace("\n", " ") for line in lines) + "\n")


def validate_line_count(inputs: list[str], outputs: list[str]) -> None:
    if len(outputs) != len(inputs):
        raise ValueError(f"Output length {len(outputs)} does not match input length {len(inputs)}")


def parse_inference_args(
    *,
    default_model: Path,
    description: str = "Run translation using a submission model",
    default_prompt: str = TRANSLATE_PROMPT,
    default_batch_size: int = DEF_BATCH_SIZE,
    default_max_new_tokens: int = DEF_MAX_NEW_TOKENS,
    default_max_new_tokens_over_input: int = DEF_MAX_NEW_TOKENS_OVER_INPUT,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pos_lang_pair", nargs="?", help="Compatibility positional language pair, e.g. ces-deu")
    parser.add_argument("pos_batch_size", nargs="?", type=int, help="Compatibility positional batch size")
    parser.add_argument("--lang-pair", help="Language pair to translate, e.g. ces-deu")
    parser.add_argument("--batch-size", type=int, help="Batch size for translation")
    parser.add_argument(
        "-m",
        "--model",
        "--model-dir",
        dest="model",
        type=Path,
        default=default_model,
        help="Path to a Hugging Face Transformers-compatible model directory",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r", encoding="utf-8", errors="replace"),
        default=sys.stdin,
        help="Input file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w", encoding="utf-8", errors="replace"),
        default=sys.stdout,
        help="Output file",
    )
    parser.add_argument("-pb", "--progress", action="store_true", help="Show progress bar")
    parser.add_argument("--max-new-tokens", type=int, default=default_max_new_tokens)
    parser.add_argument(
        "--max-new-tokens-over-input",
        type=int,
        default=default_max_new_tokens_over_input,
        help="Additional generated-token budget beyond the longest source segment in each batch",
    )
    parser.add_argument("-pt", "--prompt", type=str, default=default_prompt, help="Prompt template for translation")
    args = parser.parse_args()

    lang_pair = args.lang_pair or args.pos_lang_pair
    if not lang_pair:
        parser.error("provide --lang-pair or the compatibility positional language pair")
    try:
        args.lang_pair = normalize_lang_pair(lang_pair)
    except ValueError as exc:
        parser.error(str(exc))
    args.batch_size = args.batch_size or args.pos_batch_size or default_batch_size
    return args


class LLMBase:
    def __init__(
        self,
        model_dir: Path,
        use_chat_template: bool = True,
        prompt_template: str = TRANSLATE_PROMPT,
        progress_bar: bool = False,
        max_new_tokens: int = DEF_MAX_NEW_TOKENS,
        max_new_tokens_over_input: int = DEF_MAX_NEW_TOKENS_OVER_INPUT,
    ):
        self.model_dir = Path(model_dir)
        self.use_chat_template = use_chat_template
        self.prompt_template = prompt_template
        self.progress_bar = progress_bar
        self.max_new_tokens = max_new_tokens
        self.max_new_tokens_over_input = max_new_tokens_over_input
        self._config = None
        self._tokenizer = None
        self._model = None

    @property
    def config(self):
        if self._config is None:
            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True)
        return self._config

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM

            loader_args: dict[str, Any] = dict(device_map="auto", torch_dtype="auto", local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_dir, **loader_args)
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True, local_files_only=True)
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def length_tokenizer(self):
        return self.tokenizer

    @property
    def input_device(self):
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        return next(self.model.parameters()).device

    def translate_lines(self, pair: str, lines: list[str], batch_size: int = DEF_BATCH_SIZE) -> list[str]:
        from tqdm.auto import tqdm

        pair = normalize_lang_pair(pair)
        indexed = [(idx, line, self.source_length(line)) for idx, line in enumerate(lines) if line.strip()]
        results = [""] * len(lines)
        if not indexed:
            return results

        indexed.sort(key=lambda item: item[2], reverse=True)
        batches = [indexed[i : i + batch_size] for i in range(0, len(indexed), batch_size)]
        out_indexed = []
        pbar = tqdm(total=len(indexed), disable=not self.progress_bar)
        for batch in batches:
            ids, texts, source_lengths = zip(*batch)
            max_new_tokens = min(self.max_new_tokens, max(source_lengths) + self.max_new_tokens_over_input)
            hyps = self.translate_batch(pair, list(texts), max_new_tokens=max_new_tokens)
            out_indexed.extend(zip(ids, hyps))
            pbar.update(len(batch))
        pbar.close()

        for idx, text in out_indexed:
            results[idx] = text
        return results

    def translate_batch(self, pair: str, texts: list[str], max_new_tokens: int) -> list[str]:
        prompts = [self.make_prompt(pair, text) for text in texts]
        return self.generate_causal_lm(prompts, max_new_tokens)

    def make_prompt(self, pair: str, text: str) -> str:
        return make_translation_prompt(pair, text, self.prompt_template)

    def source_length(self, text: str) -> int:
        return len(self.length_tokenizer(text, add_special_tokens=False)["input_ids"])

    def move_inputs(self, inputs):
        inputs = inputs.to(self.input_device) if hasattr(inputs, "to") else inputs
        if isinstance(inputs, dict):
            return {key: value.to(self.input_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        return inputs

    def generate_causal_lm(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        import torch

        if self.use_chat_template:
            chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
            prompts = [
                self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                for chat in chats
            ]
        inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=not self.use_chat_template,
        )
        inputs = self.move_inputs(inputs)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        return [
            text.replace("\n", " ")
            for text in self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        ]


class Gemma3LLMBase(LLMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._processor = None

    @property
    def model(self):
        if self._model is None:
            try:
                from transformers import Gemma3ForConditionalGeneration
            except ImportError as exc:  # pragma: no cover - depends on the installed transformers version
                raise RuntimeError(
                    "Gemma 3 requires transformers with Gemma3ForConditionalGeneration support"
                ) from exc

            loader_args: dict[str, Any] = dict(device_map="auto", torch_dtype="auto", local_files_only=True)
            self._model = Gemma3ForConditionalGeneration.from_pretrained(self.model_dir, **loader_args)
            self._model.eval()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
            tokenizer = getattr(self._processor, "tokenizer", None)
            if tokenizer is not None:
                tokenizer.padding_side = "left"
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
        return self._processor

    @property
    def length_tokenizer(self):
        return self.processor.tokenizer

    def translate_batch(self, pair: str, texts: list[str], max_new_tokens: int) -> list[str]:
        prompts = [self.make_prompt(pair, text) for text in texts]
        return self.generate_gemma3(prompts, max_new_tokens)

    def generate_gemma3(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        import torch

        messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}] for prompt in prompts]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            padding=True,
        )
        inputs = self.move_inputs(inputs)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        return [
            text.replace("\n", " ")
            for text in self.processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        ]


def default_model_path(script_file: str | Path) -> Path:
    script_dir = Path(script_file).parent
    return Path(os.getenv("MODEL_DIR", os.getenv("MODELZIP_MODEL_DIR", script_dir / "workdir" / "model")))


def run_inference(args: argparse.Namespace, llm_cls: type[LLMBase] = LLMBase, *, use_chat_template: bool = True) -> None:
    llm = llm_cls(
        args.model,
        use_chat_template=use_chat_template,
        prompt_template=args.prompt,
        progress_bar=args.progress,
        max_new_tokens=args.max_new_tokens,
        max_new_tokens_over_input=args.max_new_tokens_over_input,
    )
    lines = read_source_lines(args.input)
    outputs = llm.translate_lines(args.lang_pair, lines, batch_size=args.batch_size)
    validate_line_count(lines, outputs)
    write_output_lines(args.output, outputs)
