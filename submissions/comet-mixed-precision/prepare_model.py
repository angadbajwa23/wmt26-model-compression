#!/usr/bin/env python
"""
Stage 3 (final) of the COMET-guided mixed-precision recipe for Gemma 3 12B.

Reads workdir/tier_assignment.json (produced by solve_budget.py) and runs a single
GPTQ oneshot calibration pass that quantizes each decoder layer to the precision
tier chosen for it: layers assigned "int4" get GPTQModifier's Hessian-corrected
W4A16_ASYM treatment (same convention as submissions/gptq), layers assigned "int8"
get W8A16 symmetric-group treatment, and layers assigned "bf16" are left in the
`ignore` list (untouched). llmcompressor's GPTQModifier supports all three tiers in
ONE call via `config_groups` + `ignore` -- no need for separate quantizer passes.

CALIBRATION DATA
----------------
Reuses the same data/calibration/<pair>/calibration.jsonl set as awq/gptq (built by
data_prep/build_calibration.py) -- this is a Hessian-calibration pass, not a quality
measurement, so there's no leakage concern here (same convention as the uniform
awq/gptq submissions already use).
"""
import argparse
import json
import shutil
from pathlib import Path

from layer_utils import discover_decoder_layers, layer_linear_names

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
    """Load messages from calibration.jsonl files built by data_prep/build_calibration.py."""
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
    """Apply Gemma3 chat template to each message list and return formatted strings."""
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


def load_tier_assignment(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run sensitivity_sweep.py then solve_budget.py first."
        )
    data = json.loads(path.read_text())
    return {"bf16": data.get("bf16", []), "int8": data.get("int8", []), "int4": data.get("int4", [])}


def prepare_quantized(
    model_source: Path,
    output_dir: Path,
    calib_messages: list[list[dict]],
    tier_assignment: dict[str, list[int]],
    group_size: int,
) -> None:
    if (output_dir / "config.json").exists():
        print(f"Model already exists at {output_dir}; skipping quantization.")
        return

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    except ImportError as exc:
        raise RuntimeError(
            "llmcompressor is not installed. Run: pip install llmcompressor"
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

    # Pre-tokenize and hand oneshot() an already-tokenized dataset (one with an
    # "input_ids" column). This is required to avoid the "Cannot copy out of meta
    # tensor; no data!" crash -- see submissions/gptq/prepare_model.py for the full
    # explanation. Passing raw text instead would make llmcompressor tokenize it AND
    # append a `labels` column, forcing every calibration forward pass through
    # Gemma3's loss branch, which crashes under the sequential GPTQ pipeline.
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
        device_map="cuda:0",
    )

    layer_prefixes = discover_decoder_layers(model)
    print(f"Discovered {len(layer_prefixes)} decoder layers")

    def names_for(layer_idxs: list[int]) -> list[str]:
        names = []
        for L in layer_idxs:
            names.extend(layer_linear_names(layer_prefixes[L]))
        return names

    bf16_layers = tier_assignment["bf16"]
    int8_layers = tier_assignment["int8"]
    int4_layers = tier_assignment["int4"]
    print(f"Tier assignment: bf16={len(bf16_layers)} int8={len(int8_layers)} int4={len(int4_layers)}")

    ignore = ["lm_head", "re:model.vision_tower.*"] + names_for(bf16_layers)

    config_groups = {}
    if int8_layers:
        config_groups["int8_group"] = QuantizationScheme(
            targets=names_for(int8_layers),
            weights=QuantizationArgs(num_bits=8, type="int", symmetric=True,
                                      strategy="group", group_size=group_size),
        )
    if int4_layers:
        # W4A16_ASYM convention, matching submissions/gptq's uniform-INT4 recipe.
        config_groups["int4_group"] = QuantizationScheme(
            targets=names_for(int4_layers),
            weights=QuantizationArgs(num_bits=4, type="int", symmetric=False,
                                      strategy="group", group_size=group_size),
        )
    if not config_groups:
        raise ValueError("tier_assignment has no int8/int4 layers -- nothing to quantize")

    recipe = GPTQModifier(
        config_groups=config_groups,
        ignore=ignore,
        dampening_frac=0.01,
    )

    # Gemma3ForConditionalGeneration's loss head uses nonzero() for attention mask
    # indexing, which is data-dependent and can't be evaluated during torch.fx
    # meta tracing (GPTQModifier's sequential pipeline traces the model graph).
    # Assume all mask elements are nonzero -- safe since calibration only needs
    # hidden states, not the loss output. Same workaround as submissions/gptq.
    import torch.fx.experimental._config as _fx_cfg
    _fx_cfg.meta_nonzero_assume_all_nonzero = True

    print(f"Running mixed-precision GPTQ quantization (group_size={group_size}, "
          f"n_calib={len(calib_texts)}) ...")
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
    # inference time. Copying the pristine files avoids that (same convention
    # as submissions/gptq/prepare_model.py).
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
        description="Quantize Gemma 3 12B to a COMET-guided BF16/INT8/INT4 mix using llm-compressor",
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
    parser.add_argument("--tier-assignment", type=Path, default=repo_dir / "workdir/tier_assignment.json",
                        help="Output of solve_budget.py")
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE,
                        help="Per-group quantization granularity for the int8/int4 tiers")
    args = parser.parse_args()

    source = resolve_model_source(args.model_id, args.cache_dir)
    calib_messages = load_calibration_messages(args.calib_dir, args.calib_samples_per_pair)
    tier_assignment = load_tier_assignment(args.tier_assignment)
    prepare_quantized(source, args.output, calib_messages, tier_assignment, args.group_size)


if __name__ == "__main__":
    main()
