#!/usr/bin/env python
import logging as LOG

from modelzip.submission import (
    DEF_MAX_NEW_TOKENS,
    DEF_MAX_NEW_TOKENS_OVER_INPUT,
    Gemma3LLMBase,
    TRANSLATE_PROMPT,
    default_model_path,
    parse_inference_args,
    run_inference,
)

USE_CHAT_TEMPLATE = True

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BnbQ8LLM(Gemma3LLMBase):
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, use_fast=True, local_files_only=True, fix_mistral_regex=True
            )
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer


def main():
    run_inference(parse_args(), BnbQ8LLM, use_chat_template=USE_CHAT_TEMPLATE)


def parse_args():
    return parse_inference_args(
        default_model=default_model_path(__file__),
        description="Run translation using the BNB q8 Gemma baseline",
        default_prompt=TRANSLATE_PROMPT,
        default_max_new_tokens=DEF_MAX_NEW_TOKENS,
        default_max_new_tokens_over_input=DEF_MAX_NEW_TOKENS_OVER_INPUT,
    )


if __name__ == "__main__":
    main()
