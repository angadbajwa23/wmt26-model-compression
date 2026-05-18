#!/usr/bin/env bash
set -euo pipefail

echo "baseline is the uncompressed Gemma baseline; no compression step is needed."
echo "Provide the original Gemma model at workdir/model, or set MODEL_DIR when running inference."