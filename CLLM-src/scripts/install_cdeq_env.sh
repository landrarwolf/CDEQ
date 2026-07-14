#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-cllm-cdeq}"
source /opt/anaconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
if [[ -n "${TORCH_WHEEL:-}" ]]; then
  python -m pip install "${TORCH_WHEEL}"
else
  python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
fi
python -m pip install -r requirements-cdeq-runtime.txt -c constraints-cdeq.txt
python -m pip install flash-attn==2.4.1 --no-build-isolation
python -m pip check
python -m pip freeze | sort > environment-pip-freeze.txt
conda list --explicit > environment-conda-explicit.txt
