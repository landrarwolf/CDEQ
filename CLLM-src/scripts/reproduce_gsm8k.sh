#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/llm_cdeq/gsm8k.yaml}"
TARGET="/home/ljc/models/cllm/Abel-7B-001"
CLLM="/home/ljc/models/cllm/consistency-llm-7b-math"
OUTPUT="/home/ljc/experiments/cllm-cdeq/official-reproduction"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attention_2}"
mkdir -p "${OUTPUT}"

python applications/chat_cli_cllm.py \
  --model_path "${CLLM}" --cllm_type gsm8k --debug \
  --attention_backend "${ATTENTION_BACKEND}" \
  > "${OUTPUT}/official_demo.log" 2>&1

python eval/gsm8k/acc.py \
  --model_dir "${CLLM}" --dev_set gsm8k --prompt_type math-single \
  --max_new_tokens_for_consistency 16 --max_tokens 1024 \
  --use_consistency_decoding --sample_num 1 --seed 42 \
  --attention_backend "${ATTENTION_BACKEND}" \
  --output_file_name "${OUTPUT}/demo.jsonl"

python eval/gsm8k/acc.py \
  --model_dir "${CLLM}" --dev_set gsm8k --prompt_type math-single \
  --max_new_tokens_for_consistency 16 --max_tokens 1024 \
  --use_consistency_decoding --seed 42 \
  --attention_backend "${ATTENTION_BACKEND}" \
  --output_file_name "${OUTPUT}/full.jsonl"

python -m llm_cdeq.profile --config "${CONFIG}" --method cllm \
  --sample-limit 500 --attention-backend "${ATTENTION_BACKEND}" \
  --output "${OUTPUT}/speed.json"

python -m llm_cdeq.verify_jacobi --config "${CONFIG}" --blocks 100 \
  --attention-backend "${ATTENTION_BACKEND}" \
  --output "${OUTPUT}/abel_jacobi_equivalence.json"
