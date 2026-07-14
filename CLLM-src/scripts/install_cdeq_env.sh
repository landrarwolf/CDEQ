#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-cllm-cdeq}"
source /opt/anaconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi
conda activate "${ENV_NAME}"
# FlashAttention 2.4.1 still imports ``pkg_resources`` while preparing its
# wheel. Newer setuptools releases removed that compatibility module, and pip
# 26 misclassifies the official Ninja wheel on this host.
python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.42.0
if [[ -n "${TORCH_WHEEL:-}" ]]; then
  python -m pip install "${TORCH_WHEEL}"
else
  python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
fi
python -m pip install -r requirements-cdeq-runtime.txt -c constraints-cdeq.txt
if [[ "${SKIP_FLASH_ATTN:-0}" == "1" ]]; then
  printf '%s\n' sdpa > environment-attention-backend.txt
  echo "FlashAttention installation was skipped; quality runs will use SDPA." >&2
elif python -m pip install flash-attn==2.4.1 --no-build-isolation; then
  printf '%s\n' flash_attention_2 > environment-attention-backend.txt
else
  printf '%s\n' sdpa > environment-attention-backend.txt
  if [[ "${STRICT_FLASH_ATTN:-0}" == "1" ]]; then
    exit 1
  fi
  echo "FlashAttention 2.4.1 is unavailable; quality runs may use SDPA." >&2
fi
python -m pip check
python -m pip freeze | sort > environment-pip-freeze.txt
conda list --explicit > environment-conda-explicit.txt
