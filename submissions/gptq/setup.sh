#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
modelzip_source="${MODELZIP_SOURCE:-$(cd "$root_dir/../.." && pwd)}"

# ── Inference venv (.venv) ──────────────────────────────────────────────────
# torch 2.8.0 + vllm; used by run.sh at submission time
inference_venv="$root_dir/.venv"
if [[ ! -d "$inference_venv" ]]; then
    uv venv --python 3.11 "$inference_venv"
fi
source "$inference_venv/bin/activate"
uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0"
uv pip install --index-strategy unsafe-best-match --no-build-isolation \
    -r "$root_dir/requirements.txt"
uv pip install --no-deps -e "$modelzip_source"
deactivate

# ── Compress venv (.venv-compress) ─────────────────────────────────────────
# full llmcompressor with its own torch/transformers; used by compress.sh only
# L40 is sm_89 (Ada Lovelace); cu128 builds support CUDA 12.4+ so all good
compress_venv="$root_dir/.venv-compress"
if [[ ! -d "$compress_venv" ]]; then
    uv venv --python 3.11 "$compress_venv"
fi
source "$compress_venv/bin/activate"
# Let llmcompressor install with all its own deps (torch, transformers, etc.)
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
    -r "$root_dir/requirements-compress.txt"
# Force torch back to the version proven to work on this machine (L40, driver 550, CUDA 12.4)
uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0"
deactivate

# ── Model artifact ───────────────────────────────────────────────────────────
# Downloads the pre-quantized model from Hugging Face into workdir/model unless
# it (or a MODEL_DIR override) is already present.
model_dir="${MODEL_DIR:-$root_dir/workdir/model}"
hf_repo="${HF_MODEL_REPO:-Angad23/gemma-3-12b-it-gptq-int4-wmt26}"
if [[ -f "$model_dir/config.json" ]]; then
    echo "Model already present at $model_dir, skipping download"
else
    echo "Downloading model artifact: $hf_repo -> $model_dir"
    "$inference_venv/bin/python" - "$hf_repo" "$model_dir" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo_id, local_dir=local_dir, ignore_patterns=["._OK", "*.bak"])
PYEOF
fi

echo ""
echo "Setup complete."
echo "  Inference venv : $inference_venv  (torch 2.8.0 + vllm)"
echo "  Compress venv  : $compress_venv  (llmcompressor, torch forced to 2.8.0)"
