#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME=/tmp/helico/hf
python -m pip install --disable-pip-version-check huggingface_hub s3fs
if [[ "$1" == mirror ]]; then
    exec python -m helico.coreweave_worker mirror --config "$2"
fi
python -m pip install --disable-pip-version-check -e '.[scale]'
exec python -m helico.coreweave_worker train --config "$2"
