#!/usr/bin/env python
"""
Offline W4A16 GPTQ quantization for Gemma 3 12B using llm-compressor.

HOW GPTQ WORKS
--------------
GPTQ quantizes one weight matrix at a time using second-order (Hessian) information.
For each layer it does:

  1. Compute the approximate Hessian H = 2 * X * X^T from calibration forward passes,
     where X is the layer's input activations.

  2. Iterate column-by-column through the weight matrix W. For column i:
       a. Round w_i to the nearest INT4 value.
       b. Measure the rounding error e_i = w_i - quant(w_i).
       c. Propagate e_i to all remaining columns via the Hessian inverse:
              W[:, j>i] -= e_i * (H^{-1})_{i,j>i} / (H^{-1})_{i,i}

  This column-wise error compensation is GPTQ's core advantage over RTN: quantization
  errors are explicitly corrected rather than minimised purely via scaling.

  At INT4 group_size=128, GPTQ typically matches or beats AWQ on perplexity.

WHY compressed-tensors (not legacy GPTQ format)
------------------------------------------------
llm-compressor (the vLLM project's successor to auto-gptq) saves in compressed-tensors
format. vLLM detects this from config.json and uses the Marlin kernel for fused
dequantize+GEMM on L40/A100/H100.  inference.py sets QUANTIZATION="compressed-tensors".

CALIBRATION DATA
----------------
64 examples per language pair (192 total) of full Gemma3 chat sequences:
  user: "Translate from X to Y: <source>"
  assistant: "<reference translation>"
Built by data_prep/build_calibration.py → data/calibration/<pair>/calibration.jsonl
"""
import argparse
import json
import shutil
from pathlib import Path

DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 128
DEFAULT_CALIB_SAMPLES = 64


def resolve_model_source(model_id: str, cache_dir: Path) -> Path:
    explicit_path = Path(model_id).expanduser()
    if explicit_path.exists():
        return explicit_path
    if explicit_path.is_absolute():
        raise FileNotFoundError(f"Model path does not exist: {explicit_path}")

    cached_model_dir = cache_dir.expanduser() / model_id
    if (cached_model_dir / "config.json").exists():
        return cached_model_dir

    cached_model_dir.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_id, local_dir=cached_model_dir)
    (cached_model_dir / "._DOWNLOAD_OK").touch()
    return cached_model_dir


def load_calibration_messages(calib_dir: Path, n_samples_per_pair: int) -> list[list[dict]]:
    """Load messages from calibration.jsonl files built by data_prep/build_calibration.py.

    Searches calib_dir recursively for calibration.jsonl files (one per language pair).
    Each record has {"pair": "...", "messages": [user_turn, assistant_turn]} where turns
    use Gemma3 content-block format.  Returns n_samples_per_pair examples from each file,
    evenly strided, then concatenated — guaranteeing equal representation per language pair.
    """
    jsonl_files = sorted(calib_dir.glob("**/calibration.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No calibration.jsonl files found under {calib_dir}.\n"
            "Run:  python3 data_prep/build_calibration.py\n"
            "to generate data/calibration/<pair>/calibration.jsonl"
        )

    all_selected: list[list[dict]] = []
    for jsonl_file in jsonl_files:
        messages = []
        with open(jsonl_file, encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                if "messages" in obj:
                    messages.append(obj["messages"])
        step = max(1, len(messages) // n_samples_per_pair)
        selected = messages[::step][:n_samples_per_pair]
        print(f"  {jsonl_file.parent.name}: {len(selected)} samples "
              f"(available: {len(messages)})")
        all_selected.extend(selected)

    print(f"Loaded {len(all_selected)} calibration examples total "
          f"({n_samples_per_pair} per language pair × {len(jsonl_files)} pairs)")
    return all_selected


def _flatten_and_format(tokenizer, messages_list: list[list[dict]]) -> list[str]:
    """Apply Gemma3 chat template to each message list and return formatted strings.

    Converts content-block turns (Gemma3 processor format) to plain-string turns
    (AutoTokenizer format) then applies the chat template with add_generation_prompt=False
    because the assistant turn is already included — we want the quantizer to see
    the full prompt+response token sequence.
    """
    formatted = []
    for messages in messages_list:
        flat = []
        for turn in messages:
            content = turn["content"]
            if isinstance(content, list):
                content = "".join(
                    block["text"] for block in content if block.get("type") == "text"
                )
            flat.append({"role": turn["role"], "content": content})
        text = tokenizer.apply_chat_template(
            flat,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted.append(text)
    return formatted


def prepare_quantized(
    model_source: Path,
    output_dir: Path,
    calib_messages: list[list[dict]],
    bits: int,
    group_size: int,
) -> None:
    if (output_dir / "config.json").exists():
        print(f"Model already exists at {output_dir}; skipping quantization.")
        return

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError as exc:
        raise RuntimeError(
            "llmcompressor is not installed. Run: pip install llmcompressor"
        ) from exc

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # torch 2.8.0's torch.accelerator API predates max_memory_allocated/get_memory_info,
    # which llmcompressor's CompressionLogger calls unconditionally after each layer.
    # Shim them onto torch.cuda equivalents so compression doesn't crash mid-run.
    if torch.accelerator.is_available() and not hasattr(torch.accelerator, "max_memory_allocated"):
        torch.accelerator.max_memory_allocated = lambda device_id: torch.cuda.max_memory_allocated(device_id)
        torch.accelerator.get_memory_info = lambda device_id: torch.cuda.mem_get_info(device_id)

    print(f"Loading tokenizer from {model_source} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_source),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Applying chat template to {len(calib_messages)} calibration examples ...")
    calib_texts = _flatten_and_format(tokenizer, calib_messages)

    # Tokenize ourselves and hand oneshot() an already-tokenized dataset (one with
    # an "input_ids" column). llmcompressor's dataset loader detects this and skips
    # its own tokenize+add-labels postprocessing (see llmcompressor/datasets/utils.py,
    # get_processed_dataset: `if "input_ids" in dataset.column_names: return dataset`).
    #
    # If we instead pass raw text, llmcompressor tokenizes it AND appends a `labels`
    # column (copied from input_ids) meant for finetuning-style loss supervision.
    # That forces every calibration forward pass to hit Gemma3ForConditionalGeneration's
    # loss branch (`if labels is not None`), which does boolean-mask indexing on
    # `labels`/`attention_mask`. Under llmcompressor's sequential GPTQ pipeline, the
    # `labels` tensor for the final subgraph is never onloaded off the meta device
    # (only hidden-state-producing modules are), so that indexing crashes with
    # "Cannot copy out of meta tensor; no data!" GPTQ calibration only needs hidden
    # states, not a loss, so skipping label creation avoids the branch entirely.
    tokenized = tokenizer(
        calib_texts,
        truncation=True,
        max_length=512,
        padding=False,
    )
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    })

    print(f"Loading base model from {model_source} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_source),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # GPTQ: calibration-aware Hessian-based quantization.
    # GPTQModifier computes per-layer Hessians from calibration activations and
    # applies column-wise error correction during INT4 rounding.
    # No separate QuantizationModifier needed — GPTQModifier handles both.
    #
    # Vision tower excluded: text-only MT task + non-divisible column dims (4304).
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16_ASYM",
        ignore=["lm_head", "re:model.vision_tower.*"],
        dampening_frac=0.01,
    )

    # Gemma3ForConditionalGeneration's loss head uses nonzero() for attention mask
    # indexing, which is data-dependent and can't be evaluated during torch.fx
    # meta tracing. Assume all mask elements are nonzero — safe since calibration
    # only needs hidden states, not the loss output.
    import torch.fx.experimental._config as _fx_cfg
    _fx_cfg.meta_nonzero_assume_all_nonzero = True

    print(
        f"Running GPTQ W4A16_ASYM quantization "
        f"(bits={bits}, group_size={group_size}, n_calib={len(calib_texts)}) ..."
    )
    oneshot(
        model=model,
        recipe=recipe,
        dataset=dataset,
        num_calibration_samples=len(calib_texts),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), save_compressed=True)

    # Copy tokenizer/processor files straight from the source checkpoint instead
    # of tokenizer.save_pretrained(): the tokenizer instance above just ran
    # tokenizer(..., truncation=True, max_length=512), which mutates the
    # underlying fast tokenizer's truncation state and would otherwise get
    # baked into the saved tokenizer.json, silently truncating long inputs at
    # inference time. Copying the pristine files avoids that, and also avoids
    # tokenizer_config.json key-name drift between the transformers version
    # used here and the one vLLM loads with (e.g. "extra_special_tokens" vs.
    # "model_specific_special_tokens" for the boi/eoi/image tokens), which
    # otherwise breaks Gemma3Processor's tokenizer.image_token_id lookup.
    #
    # config.json still reports Gemma3ForConditionalGeneration (the multimodal
    # architecture used at calibration time, even though only the language
    # backbone is quantized/served), so vLLM's multimodal model loader also
    # needs the image processor config files alongside the tokenizer.
    for tokenizer_file in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
        "chat_template.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        src = model_source / tokenizer_file
        if src.exists():
            shutil.copy2(src, output_dir / tokenizer_file)

    (output_dir / "._OK").touch()
    print(f"Quantized model saved to {output_dir}")


def main() -> None:
    repo_dir = Path(__file__).parent
    default_calib = (repo_dir / "../../data/calibration").resolve()

    parser = argparse.ArgumentParser(
        description="Quantize Gemma 3 12B to W4A16 INT4 using GPTQ via llm-compressor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default="google/gemma-3-12b-it",
                        help="HuggingFace repo ID or local path to the base model")
    parser.add_argument("--cache-dir", type=Path,
                        default="/mnt/tg/data/projects/wmt26/model-compression/models",
                        help="Directory used to cache downloaded models")
    parser.add_argument("--output", type=Path, default="workdir/model",
                        help="Destination directory for the quantized model")
    parser.add_argument("--calib-dir", type=Path, default=default_calib,
                        help="Directory with calibration.jsonl files (one per language pair sub-dir)")
    parser.add_argument("--calib-samples-per-pair", type=int, default=DEFAULT_CALIB_SAMPLES,
                        help="Number of calibration examples per language pair (total = this × number of pairs)")
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS, choices=[4],
                        help="Weight quantization bit-width")
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE,
                        help="Per-group quantization granularity (128 is standard for Marlin)")
    args = parser.parse_args()

    source = resolve_model_source(args.model_id, args.cache_dir)
    calib_messages = load_calibration_messages(args.calib_dir, args.calib_samples_per_pair)
    prepare_quantized(source, args.output, calib_messages, args.bits, args.group_size)


if __name__ == "__main__":
    main()
