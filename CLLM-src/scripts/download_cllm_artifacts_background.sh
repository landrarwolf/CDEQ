#!/usr/bin/env bash
set -u

STAGE=/private/tmp/cllm-stage
REMOTE=pc-cot-120
REMOTE_ABEL=/home/ljc/models/cllm/Abel-7B-001
REMOTE_CLLM=/home/ljc/models/cllm/consistency-llm-7b-math
REMOTE_CODE=/home/ljc/Code/Consistency_LLM_CDEQ
SSH_OPTIONS=(-o UpdateHostKeys=no -o ServerAliveInterval=10 -o ServerAliveCountMax=6)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

local_size() {
  stat -f '%z' "$1" 2>/dev/null || printf '0\n'
}

remote_ok() {
  local remote_path=$1
  local expected_size=$2
  local expected_sha=$3
  local result
  result=$(ssh "${SSH_OPTIONS[@]}" "${REMOTE}" \
    "test -f '${remote_path}' && printf '%s ' \"\$(stat -c '%s' '${remote_path}')\" && sha256sum '${remote_path}' | awk '{print \$1}'" \
    2>/dev/null) || return 1
  [[ "${result}" == "${expected_size} ${expected_sha}" ]]
}

download_and_stage() {
  local label=$1
  local url=$2
  local local_path=$3
  local expected_size=$4
  local expected_sha=$5
  local remote_path=$6

  mkdir -p "$(dirname "${local_path}")"
  if remote_ok "${remote_path}" "${expected_size}" "${expected_sha}"; then
    log "${label}: remote artifact already verified; skipping download"
    rm -f "${local_path}" "${local_path}.aria2"
    return
  fi

  log "${label}: resuming download"
  while [[ -f "${local_path}.aria2" || "$(local_size "${local_path}")" != "${expected_size}" ]]; do
    aria2c -c -x 16 -s 16 -k 1M \
      --file-allocation=none --summary-interval=20 \
      --console-log-level=warn \
      -d "$(dirname "${local_path}")" -o "$(basename "${local_path}")" \
      "${url}" || log "${label}: aria2 interrupted; retrying in 5 seconds"
    sleep 5
  done

  local actual_sha
  actual_sha=$(shasum -a 256 "${local_path}" | awk '{print $1}')
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    log "${label}: FATAL local SHA-256 mismatch: ${actual_sha}"
    exit 1
  fi
  log "${label}: local size and SHA-256 verified"

  while ! remote_ok "${remote_path}" "${expected_size}" "${expected_sha}"; do
    log "${label}: rsyncing to remote"
    rsync -a --partial --inplace --checksum --progress \
      -e "ssh -o UpdateHostKeys=no -o ServerAliveInterval=10 -o ServerAliveCountMax=6" \
      "${local_path}" "${REMOTE}:${remote_path}" || \
      log "${label}: rsync interrupted; retrying in 5 seconds"
    sleep 5
  done
  log "${label}: remote size and SHA-256 verified"
  rm -f "${local_path}" "${local_path}.aria2"
  log "${label}: removed verified local temporary copy"
}

mkdir -p "${STAGE}"
rm -f "${STAGE}/download.done"
log "background artifact queue started"

download_and_stage \
  abel-shard-3 \
  'https://huggingface.co/GAIR/Abel-7B-001/resolve/3439c5a654dac2320d228d11a0c5590346e81d1a/pytorch_model-00003-of-00003.bin?download=true' \
  "${STAGE}/abel-aria2/pytorch_model-00003-of-00003.bin" \
  7181510149 \
  60b8958b936d62ccfa15f5d60e0be35173c2a5c34238a1192435d2ae9824903a \
  "${REMOTE_ABEL}/pytorch_model-00003-of-00003.bin"

download_and_stage \
  abel-shard-1 \
  'https://huggingface.co/GAIR/Abel-7B-001/resolve/3439c5a654dac2320d228d11a0c5590346e81d1a/pytorch_model-00001-of-00003.bin?download=true' \
  "${STAGE}/abel/.cache/huggingface/download/tYEjMy-9HxF6eZaoFpYvo3axcg4=.74400d14928ac9bb3219a5c76d947b9c0b86cf095e0cdc68d4a14cfdc1bd60a8.incomplete" \
  9878525722 \
  74400d14928ac9bb3219a5c76d947b9c0b86cf095e0cdc68d4a14cfdc1bd60a8 \
  "${REMOTE_ABEL}/pytorch_model-00001-of-00003.bin"

download_and_stage \
  abel-shard-2 \
  'https://huggingface.co/GAIR/Abel-7B-001/resolve/3439c5a654dac2320d228d11a0c5590346e81d1a/pytorch_model-00002-of-00003.bin?download=true' \
  "${STAGE}/abel-aria2/pytorch_model-00002-of-00003.bin" \
  9894793766 \
  36bf491ad0f8f850a6f0bbff4862e107a64eb77c7f55d0682671e4c824291140 \
  "${REMOTE_ABEL}/pytorch_model-00002-of-00003.bin"

download_and_stage \
  cllm-7b-math \
  'https://huggingface.co/cllm/consistency-llm-7b-math/resolve/904a1eefdf8e33a3440ddea35a55dd75cead648c/pytorch_model.bin?download=true' \
  "${STAGE}/cllm/pytorch_model.bin" \
  13477502757 \
  1159bb6bd9087b22d8538e43029c8bda731f07bcc69b558fc2686ed50381e00a \
  "${REMOTE_CLLM}/pytorch_model.bin"

log "running pinned remote artifact verification"
ssh "${SSH_OPTIONS[@]}" "${REMOTE}" \
  "cd '${REMOTE_CODE}' && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate cllm-cdeq && \
   python -m llm_cdeq.verify_artifacts --root '${REMOTE_ABEL}' \
     --revision 3439c5a654dac2320d228d11a0c5590346e81d1a \
     --manifest configs/llm_cdeq/artifacts/abel-7b-001.json \
     --output /home/ljc/experiments/cllm-cdeq/artifacts/abel-7b-001.json && \
   python -m llm_cdeq.verify_artifacts --root '${REMOTE_CLLM}' \
     --revision 904a1eefdf8e33a3440ddea35a55dd75cead648c \
     --manifest configs/llm_cdeq/artifacts/cllm-7b-math.json \
     --output /home/ljc/experiments/cllm-cdeq/artifacts/cllm-7b-math.json"

touch "${STAGE}/download.done"
log "ALL ARTIFACTS VERIFIED"
