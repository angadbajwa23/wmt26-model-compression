#!/usr/bin/env python
"""
Stage 1 of the COMET-guided mixed-precision recipe: measure how much translation
quality drops when EACH decoder layer, in isolation, is quantized to INT4.

Loads gemma-3-12b-it once in BF16 and, for each layer, in-place round-to-nearest
(RTN) fake-quantizes that layer's 7 linear weights to INT4 (group_size=128,
asymmetric — matching the W4A16_ASYM convention used by the awq/gptq submissions),
translates the held-out dev subset (see build_devset.py), scores it with COMET
(wmt22-comet-da via pymarian-eval, matching modelzip/evaluate.py's official metric),
then restores the original weights before moving to the next layer.

RTN (no Hessian correction) is a cheap proxy used only to RANK layers by relative
sensitivity — Stage 3 (prepare_model.py) applies GPTQ's Hessian-corrected
quantization to the layers actually assigned to a quantized tier.

Output: workdir/sensitivity.json
  {"baseline_comet": 0.8731, "layers": {"0": 0.0031, "1": 0.0002, ...}}
"""
import argparse
import json
import subprocess as sp
import tempfile
from pathlib import Path

import torch

from layer_utils import discover_decoder_layers, layer_linear_names

PYMARIAN_METRIC = "wmt22-comet-da"
DEF_MAX_NEW_TOKENS = 1024
DEF_MAX_NEW_TOKENS_OVER_INPUT = 64

# Mirrors modelzip/submission.py — duplicated here so this compress-time script
# stays self-contained in the .venv-compress environment (which does not install
# modelzip; see submissions/gptq/prepare_model.py for the same convention).
TRANSLATE_PROMPT = (
    "Translate the following text from {src} to {tgt}. "
    "Return only the translation, with no explanation, labels, or quotes.\n\n"
    "{text}\n"
)
LANGS_MAP = {
    "ces": "Czech", "deu": "German",
    "zho_Hans": "Simplified Chinese",
    "eng": "English",
    "ara_EG": "Egyptian Arabic",
}
PAIR_LANGS = {
    "ces-deu": ("ces", "deu"),
    "eng-zho_Hans": ("eng", "zho_Hans"),
    "eng-ara_EG": ("eng", "ara_EG"),
}


def make_translation_prompt(pair: str, text: str) -> str:
    src_code, tgt_code = PAIR_LANGS[pair]
    return TRANSLATE_PROMPT.format(src=LANGS_MAP[src_code], tgt=LANGS_MAP[tgt_code], text=text)


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


def load_devset(devset_dir: Path) -> list[dict]:
    records = []
    for jsonl_file in sorted(devset_dir.glob("*/dev.jsonl")):
        with open(jsonl_file, encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
    if not records:
        raise FileNotFoundError(
            f"No dev.jsonl files found under {devset_dir}.\nRun: python3 build_devset.py"
        )
    return records


def fake_quantize_int4_group(w, group_size: int = 128):
    """Asymmetric per-group round-to-nearest INT4 quantize-dequantize round trip."""
    orig_shape = w.shape
    assert orig_shape[-1] % group_size == 0, (
        f"in_features={orig_shape[-1]} not divisible by group_size={group_size}"
    )
    g = w.reshape(-1, group_size).float()
    lo = g.min(dim=-1, keepdim=True).values
    hi = g.max(dim=-1, keepdim=True).values
    scale = (hi - lo).clamp_min(1e-5) / 15.0
    zero = (-lo / scale).round()
    q = (g / scale + zero).round().clamp(0, 15)
    dq = (q - zero) * scale
    return dq.reshape(orig_shape).to(w.dtype)


def translate_batch(model, tokenizer, records: list[dict], batch_size: int = 8) -> list[str]:
    tokenizer.padding_side = "left"
    hyps: list[str] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        prompts = [make_translation_prompt(r["pair"], r["src"]) for r in batch]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        max_new_tokens = min(DEF_MAX_NEW_TOKENS, input_len + DEF_MAX_NEW_TOKENS_OVER_INPUT)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        gen = out[:, input_len:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        hyps.extend(d.replace("\n", " ").strip() for d in decoded)
    return hyps


def pymarian_comet(src_texts: list[str], hyp_texts: list[str], ref_texts: list[str], metric: str = PYMARIAN_METRIC) -> float:
    """Same pymarian-eval invocation as modelzip/evaluate.py::get_score(metric='wmt22-comet-da')."""
    import os

    cache = os.getenv("PYMARIAN_CACHE", "/mnt/tg/data/cache/marian/metric")
    extra = os.getenv("PYMARIAN_EXTRA", "-c 16")
    # Held-out dev records are paragraph-level and can contain embedded "\n"
    # (dialogues, numbered lists, ...). hyp_texts is already single-line
    # (translate_batch strips newlines from generations); src/ref must be
    # sanitized the same way here, or an embedded newline injects an extra
    # physical line into src.txt/ref.txt but not hyp.txt, desynchronizing the
    # three files' line counts and making pymarian-eval fail.
    def _one_line(texts: list[str]) -> list[str]:
        return [t.replace("\n", " ") for t in texts]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src_f, hyp_f, ref_f = tmp / "src.txt", tmp / "hyp.txt", tmp / "ref.txt"
        src_f.write_text("\n".join(_one_line(src_texts)) + "\n", encoding="utf-8")
        hyp_f.write_text("\n".join(_one_line(hyp_texts)) + "\n", encoding="utf-8")
        ref_f.write_text("\n".join(_one_line(ref_texts)) + "\n", encoding="utf-8")
        cmd = (
            f"pymarian-eval --cache {cache} {extra} -m {metric} "
            f"-r {ref_f} -t {hyp_f} -s {src_f} -a only"
        )
        out = sp.check_output(cmd, shell=True, text=True).strip()
    return float(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-layer INT4 COMET-drop sensitivity sweep for gemma-3-12b-it",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default="google/gemma-3-12b-it")
    parser.add_argument("--cache-dir", type=Path,
                        default="/mnt/tg/data/projects/wmt26/model-compression/models")
    parser.add_argument("--devset-dir", type=Path,
                        default=(Path(__file__).parent / "../../data/sensitivity-devset").resolve())
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "workdir/sensitivity.json")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to probe (default: all). Useful for smoke tests.")
    parser.add_argument("--metric", default=PYMARIAN_METRIC,
                        help="pymarian-eval metric. wmt22-cometkiwi-da is ~2.7x faster than wmt22-comet-da "
                             "(reference-free QE, good for smoke tests); wmt23-cometkiwi-da-xl is slowest.")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = resolve_model_source(args.model_id, args.cache_dir)
    devset = load_devset(args.devset_dir)
    print(f"Loaded {len(devset)} held-out dev sentences from {args.devset_dir}")

    print(f"Loading tokenizer from {source} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {source} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(source), local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    # Weights are mutated in place every loop iteration (quantize layer N, restore, move to
    # N+1), which invalidates transformers' auto-compile guards on every call and blows past
    # torch._dynamo's recompile_limit within a handful of layers (FailOnRecompileLimitHit).
    # Compilation buys nothing here anyway since the graph can't be reused across iterations.
    model.generation_config.disable_compile = True

    layers = discover_decoder_layers(model)
    print(f"Discovered {len(layers)} decoder layers")
    if args.layers:
        wanted = {int(x) for x in args.layers.split(",")}
        layers = {i: p for i, p in layers.items() if i in wanted}
        print(f"Restricting sweep to layers: {sorted(layers)}")

    srcs = [r["src"] for r in devset]
    refs = [r["ref"] for r in devset]

    sensitivity: dict[str, float] = {}
    baseline_comet = None
    if args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("metric") == args.metric:
            baseline_comet = prior.get("baseline_comet")
            sensitivity = prior.get("layers", {})
            print(f"Resuming from {args.output}: {len(sensitivity)} layer(s) already scored, "
                  f"baseline_comet={baseline_comet:.4f}")
        else:
            print(f"{args.output} exists but was computed with metric={prior.get('metric')!r}, "
                  f"not {args.metric!r}; ignoring it and starting fresh.")

    if baseline_comet is None:
        print("Scoring BF16 baseline ...")
        baseline_hyps = translate_batch(model, tokenizer, devset, batch_size=args.batch_size)
        baseline_comet = pymarian_comet(srcs, baseline_hyps, refs, metric=args.metric)
        print(f"Baseline COMET ({args.metric}): {baseline_comet:.4f}")
    else:
        print(f"Reusing cached baseline COMET ({args.metric}): {baseline_comet:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for i, (layer_idx, prefix) in enumerate(sorted(layers.items())):
        if str(layer_idx) in sensitivity:
            print(f"[{i + 1}/{len(layers)}] layer {layer_idx}: already scored "
                  f"(drop={sensitivity[str(layer_idx)]:+.4f}), skipping")
            continue

        names = layer_linear_names(prefix)
        modules = [model.get_submodule(n) for n in names]
        backups = [m.weight.data.clone() for m in modules]

        for m in modules:
            m.weight.data.copy_(fake_quantize_int4_group(m.weight.data, args.group_size))

        hyps = translate_batch(model, tokenizer, devset, batch_size=args.batch_size)
        comet = pymarian_comet(srcs, hyps, refs, metric=args.metric)
        drop = baseline_comet - comet
        sensitivity[str(layer_idx)] = drop
        print(f"[{i + 1}/{len(layers)}] layer {layer_idx}: COMET={comet:.4f}  drop={drop:+.4f}")

        for m, backup in zip(modules, backups):
            m.weight.data.copy_(backup)

        # persist incrementally so a crash/interrupt doesn't lose completed layers
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"metric": args.metric, "baseline_comet": baseline_comet, "layers": sensitivity}, fh, indent=2)

    print(f"Wrote sensitivity scores for {len(sensitivity)} layers to {args.output}")


if __name__ == "__main__":
    main()
