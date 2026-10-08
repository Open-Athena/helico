#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME=/tmp/helico/hf
if [[ "$1" == mirror ]]; then
    python -m pip install --disable-pip-version-check huggingface_hub s3fs
    exec python -m helico.coreweave_worker mirror --config "$2"
fi
mkdir -p /tmp/helico
# The official PyTorch image marks system Python externally managed. Install
# only the environment builder into a private target, then use a real venv.
python -m pip install --disable-pip-version-check --target /tmp/helico/tools uv==0.9.18
UV=/tmp/helico/tools/bin/uv
"$UV" venv --system-site-packages --python "$(command -v python)" /tmp/helico/venv
"$UV" export --frozen --extra scale --no-dev --no-emit-project --no-hashes -o /tmp/helico/requirements.txt
"$UV" pip install --python /tmp/helico/venv/bin/python -r /tmp/helico/requirements.txt
"$UV" pip install --python /tmp/helico/venv/bin/python --no-deps -e .
exec /tmp/helico/venv/bin/python -m helico.coreweave_worker train --config "$2"
