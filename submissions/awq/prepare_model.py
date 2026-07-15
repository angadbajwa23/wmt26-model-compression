#!/usr/bin/env python
"""
Offline W4A16 quantization for Gemma 3 12B using llm-compressor.

AutoAWQ was deprecated in 2025 and adopted by the vLLM project as llm-compressor
(https://github.com/vllm-project/llm-compressor).  This script uses llm-compressor's
oneshot() API to apply INT4 weight quantization (W4A16) with group_size=128.

HOW W4A16 WORKS
---------------
Weights are quantized to INT4, activations remain in FP16/BF16.
llm-compressor's W4A16 scheme uses GPTQ-style layer-wise calibration:
  1. Run calibration data through each Linear layer.
  2. Minimize the layer output error after rounding weights to INT4.
  3. Store INT4 weights + per-group FP16 scales in compressed-tensors format.

The saved model uses the compressed-tensors layout (not the legacy GEMM layout
from autoawq).  vLLM detects this from config.json and uses the Marlin kernel
for fused dequantize+GEMM on H100/A100/L40.

CALIBRATION DATA
----------------
512 examples per language pair (1536 total) of full Gemma3 chat sequences:
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
DEFAULT_CALIB_SAMPLES = 512


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


def _build_awq_mappings(model, AWQMapping):
    """Build explicit per-layer AWQ mappings from the model's actual module names.

    Regex-based mappings fail for Gemma3ForConditionalGeneration because the vision
    tower's q/k/v_proj modules are iterated before the language model layers, which
    pushes the grouping ancestor to the top-level 'model' node — causing all 48
    input_layernorm modules to land in one group (AWQ requires exactly one per mapping).

    Using exact module names avoids this by bypassing regex grouping entirely.
    """
    all_names = {name for name, _ in model.named_modules()}
    mappings = []

    for name in sorted(all_names):
        if not name.endswith(".input_layernorm"):
            continue
        if "vision_tower" in name:
            continue
        prefix = name[: -len(".input_layernorm")]

        q = f"{prefix}.self_attn.q_proj"
        k = f"{prefix}.self_attn.k_proj"
        v = f"{prefix}.self_attn.v_proj"
        o = f"{prefix}.self_attn.o_proj"
        pre_ff = f"{prefix}.pre_feedforward_layernorm"
        gate = f"{prefix}.mlp.gate_proj"
        up = f"{prefix}.mlp.up_proj"
        down = f"{prefix}.mlp.down_proj"

        if all(n in all_names for n in [q, k, v]):
            mappings.append(AWQMapping(f"{prefix}.input_layernorm", [q, k, v]))
        if v in all_names and o in all_names:
            mappings.append(AWQMapping(v, [o]))
        if pre_ff in all_names and all(n in all_names for n in [gate, up]):
            mappings.append(AWQMapping(pre_ff, [gate, up]))
        if up in all_names and down in all_names:
            mappings.append(AWQMapping(up, [down]))

    print(f"Built {len(mappings)} explicit AWQ mappings across {len(mappings) // 4} layers")
    return mappings


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
        from llmcompressor.modifiers.transform.awq import AWQModifier, AWQMapping
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError as exc:
        raise RuntimeError(
            "llmcompressor is not installed. Run: pip install llmcompressor>=0.5.0"
        ) from exc

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    # `labels`/`attention_mask`. Under llmcompressor's sequential AWQ pipeline, the
    # `labels` tensor for the final subgraph is never onloaded off the meta device
    # (only hidden-state-producing modules are), so that indexing crashes with
    # "Cannot copy out of meta tensor; no data!" AWQ calibration only needs hidden
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
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # AWQ: find salient channels via activation magnitude, scale them before
    # rounding so they retain precision, then quantize all Linear layers to W4A16.
    # AWQModifier handles activation-aware scaling transforms;
    # QuantizationModifier applies the actual INT4 rounding after transforms.
    #
    # Build explicit per-layer mappings by inspecting the loaded model's module names.
    # Regex patterns like "re:.*input_layernorm$" match all layers at once; AWQ requires
    # exactly one smooth_layer per mapping. The vision tower (SigLIP) also has q/k/v_proj
    # modules, which corrupt the grouping when regex is used globally.
    awq_mappings = _build_awq_mappings(model, AWQMapping)
    recipe = [
        AWQModifier(mappings=awq_mappings, duo_scaling="both",
                    offload_device=torch.device("cpu")),
        QuantizationModifier(
            targets=["Linear"],
            scheme="W4A16_ASYM",
            ignore=["lm_head", "re:model.vision_tower.*"],
        ),
    ]

    # Gemma3ForConditionalGeneration's loss head uses nonzero() for attention mask
    # indexing, which is data-dependent and can't be evaluated during torch.fx
    # meta tracing. Assume all mask elements are nonzero — safe since calibration
    # only needs hidden states, not the loss output.
    import torch.fx.experimental._config as _fx_cfg
    _fx_cfg.meta_nonzero_assume_all_nonzero = True

    print(
        f"Running AWQ W4A16_ASYM quantization "
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
        description="Quantize Gemma 3 12B to W4A16 INT4 using llm-compressor",
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
