#!/usr/bin/env python
import argparse
from pathlib import Path


DEFAULT_QUANTIZATION = "bnb-q8"


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


def quantization_config(kind: str):
    from transformers import BitsAndBytesConfig

    if kind == "bnb-q8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if kind == "bnb-q4":
        return BitsAndBytesConfig(load_in_4bit=True)
    raise ValueError(f"Unsupported quantization kind: {kind}")


def prepare_model(model_source: Path, output_dir: Path, kind: str):
    if (output_dir / "config.json").exists():
        print(f"Model already prepared at {output_dir}")
        return

    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    config = AutoConfig.from_pretrained(model_source, local_files_only=True)
    if getattr(config, "model_type", "") == "gemma3":
        from transformers import Gemma3ForConditionalGeneration

        processor_or_tokenizer = AutoProcessor.from_pretrained(model_source, local_files_only=True)
        model_cls = Gemma3ForConditionalGeneration
    else:
        processor_or_tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True, use_fast=True)
        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(
        model_source,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config(kind),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    processor_or_tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    (output_dir / "._OK").touch()
    print(f"Prepared {kind} model at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare a BNB-quantized Gemma baseline")
    parser.add_argument("--model-id", default="google/gemma-3-12b-it")
    parser.add_argument("--cache-dir", type=Path, default="/mnt/tg/data/projects/wmt26/model-compression/models")
    parser.add_argument("--output", type=Path, default="workdir/model")
    parser.add_argument("--quantization", choices=["bnb-q8", "bnb-q4"], default=DEFAULT_QUANTIZATION)
    args = parser.parse_args()

    source = resolve_model_source(args.model_id, args.cache_dir)
    prepare_model(source, args.output, args.quantization)


if __name__ == "__main__":
    main()
