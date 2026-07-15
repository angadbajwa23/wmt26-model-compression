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
        self._max_num_seqs = self.MAX_NUM_SEQS

    @property
    def vllm(self):
        if self._vllm is None:
            from vllm import LLM

            LOG.info(
                "Loading vLLM model from %s (quantization=%s, max_num_seqs=%d)",
                self.model_dir, self.QUANTIZATION, self._max_num_seqs,
            )
            init_kwargs = dict(
                model=str(self.model_dir),
                dtype="bfloat16",
                max_num_seqs=self._max_num_seqs,
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

        # Cap vLLM's continuous-batching concurrency at the requested batch size instead of
        # always running at MAX_NUM_SEQS, so the batch-size sweep in evals/evaluate.sh actually
        # measures different concurrency levels (batch=1 ~= single-request latency) rather than
        # always reporting full-concurrency throughput regardless of the requested batch size.
        self._max_num_seqs = max(1, min(batch_size, self.MAX_NUM_SEQS))

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


class VLLMAwq(VLLMTranslator):
    # compressed-tensors format saved by llm-compressor.
    # vLLM detects W4A16 group quantization from the config and routes to the
    # Marlin kernel (fused dequantize+GEMM) for H100/A100/L40 throughput.
    QUANTIZATION = "compressed-tensors"


def main():
    args = parse_inference_args(
        default_model=default_model_path(__file__),
        description="Run translation using vLLM with AWQ INT4 (awq_marlin) quantization",
        default_prompt=TRANSLATE_PROMPT,
        default_max_new_tokens=DEF_MAX_NEW_TOKENS,
        default_max_new_tokens_over_input=DEF_MAX_NEW_TOKENS_OVER_INPUT,
    )
    run_inference(args, VLLMAwq)


if __name__ == "__main__":
    main()
