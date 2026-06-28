#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
venv_dir="$root_dir/.venv"
modelzip_source="${MODELZIP_SOURCE:-$(cd "$root_dir/../.." && pwd)}"

if [[ ! -d "$venv_dir" ]]; then
    uv venv --python 3.11 "$venv_dir"
fi
source "$venv_dir/bin/activate"
uv pip install -r "$root_dir/requirements.txt"
uv pip install --no-deps -e "$modelzip_source"
