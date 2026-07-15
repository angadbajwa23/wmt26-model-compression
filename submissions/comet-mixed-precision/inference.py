#!/usr/bin/env python
import logging as LOG

from modelzip.submission import (
    DEF_BATCH_SIZE,
    DEF_MAX_NEW_TOKENS,
    DEF_MAX_NEW_TOKENS_OVER_INPUT,
    TRANSLATE_PROMPT,
    LLMBase,
    default_model_path,
    make_translation_prompt,
    normalize_lang_pair,
    parse_inference_args,
    run_inference,
)

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class VLLMTranslator(LLMBase):
    """vLLM-backed translator with batched generation.  Subclasses set QUANTIZATION."""

    QUANTIZATION: str | None = None
    MAX_NUM_SEQS: int = 512
    MAX_MODEL_LEN: int = 4096
    GPU_MEMORY_UTILIZATION: float = 0.90

    def __init__(self, model_dir, **kwargs):
        super().__init__(model_dir, **kwargs)
        self._vllm = None

    @property
    def vllm(self):
        if self._vllm is None:
            from vllm import LLM

            LOG.info("Loading vLLM model from %s (quantization=%s)", self.model_dir, self.QUANTIZATION)
            init_kwargs = dict(
                model=str(self.model_dir),
                dtype="bfloat16",
                max_num_seqs=self.MAX_NUM_SEQS,
                max_model_len=self.MAX_MODEL_LEN,
                gpu_memory_utilization=self.GPU_MEMORY_UTILIZATION,
                trust_remote_code=False,
                enforce_eager=False,
            )
            if self.QUANTIZATION:
                init_kwargs["quantization"] = self.QUANTIZATION
            self._vllm = LLM(**init_kwargs)
        return self._vllm

    def translate_lines(self, pair: str, lines: list[str], batch_size: int = DEF_BATCH_SIZE) -> list[str]:
        from vllm import SamplingParams

        pair = normalize_lang_pair(pair)
        results = [""] * len(lines)
        active = [(idx, line) for idx, line in enumerate(lines) if line.strip()]
        if not active:
            return results

        prompts = [make_translation_prompt(pair, line) for _, line in active]

        tokenizer = self.vllm.get_tokenizer()
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        lengths = [len(tokenizer.encode(p)) for p in prompts]
        max_new_tokens = min(
            self.max_new_tokens,
            max(lengths) + self.max_new_tokens_over_input,
        )

        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        outputs = self.vllm.generate(formatted, sampling_params)

        for (idx, _), output in zip(active, outputs):
            results[idx] = output.outputs[0].text.replace("\n", " ")

        return results


class VLLMCometMixed(VLLMTranslator):
    # llm-compressor saves in compressed-tensors format with per-layer config_groups
    # (BF16/INT8/INT4 tiers chosen by the COMET-guided sensitivity sweep, see
    # prepare_model.py). vLLM's CompressedTensorsConfig detects config_groups from
    # config.json and picks a per-layer scheme automatically at load time — no
    # inference-side changes needed vs. the uniform-precision awq/gptq submissions.
    QUANTIZATION = "compressed-tensors"


def main():
    args = parse_inference_args(
        default_model=default_model_path(__file__),
        description="Run translation using vLLM with COMET-guided mixed-precision (compressed-tensors) quantization",
        default_prompt=TRANSLATE_PROMPT,
        default_max_new_tokens=DEF_MAX_NEW_TOKENS,
        default_max_new_tokens_over_input=DEF_MAX_NEW_TOKENS_OVER_INPUT,
    )
    run_inference(args, VLLMCometMixed)


if __name__ == "__main__":
    main()
