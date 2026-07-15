# CDEQ+-Jacobi on CLLM/GSM8K

This directory keeps the official CLLM source at upstream commit
`22775363f8563c63620e71f7a204e90a51d6a379`. The isolated implementation in
`llm_cdeq/` freezes Abel-7B and its LM head, distills continuous Jacobi hidden
trajectories into a one-step bottleneck adapter, and supports the four
`Init × CT` ablations. It does not change the existing CDEQ entrypoint.

## Pinned inputs

| Artifact | Revision |
| --- | --- |
| `cllm/consistency-llm-7b-math` | `904a1eefdf8e33a3440ddea35a55dd75cead648c` |
| `GAIR/Abel-7B-001` | `3439c5a654dac2320d228d11a0c5590346e81d1a` |
| GSM8K Jacobi data | `d29940a2d0ac1dea42e92d758598c64c0041a3c1` |

The model, dataset, hidden cache, experiment outputs, and checkpoints stay on
`pc-cot-120`; they are ignored by Git and must not be synced into this repo.

## Environment

The supported runtime is Python 3.10 with PyTorch 2.1.2, Transformers 4.36.2,
Accelerate 0.25.0, Datasets 2.15.0, and FlashAttention 2.4.1. On the remote
server:

```bash
cd /home/ljc/Code/Consistency_LLM_CDEQ
bash scripts/install_cdeq_env.sh cllm-cdeq
```

On a server with slow PyTorch egress, stage the Linux wheel first and set
`TORCH_WHEEL=/absolute/path/to/torch-2.1.2+cu121-...whl` for the same installer.

`requirements-cdeq.txt` records the complete direct dependency pins.
`requirements-cdeq-runtime.txt` intentionally excludes PyTorch and
FlashAttention because those two packages have separate CUDA/build steps. The
official transitive versions needed by this subset are fixed in
`constraints-cdeq.txt`. The installer writes the resolved environment to
`environment-pip-freeze.txt` and
`environment-conda-explicit.txt`.
If FlashAttention cannot be built, use `evaluation.attention_backend: sdpa`
for quality checks; all speed comparisons must later use one common backend.
The installer records the detected choice in `environment-attention-backend.txt`;
set `STRICT_FLASH_ATTN=1` when a missing FlashAttention build should be fatal.
Set `SKIP_FLASH_ATTN=1` after a recorded build failure to regenerate the exact
environment locks without retrying the unavailable wheel.
The build frontend is pinned to pip 24.0, setuptools 69.5.1, and wheel 0.42.0
because FlashAttention 2.4.1 requires the legacy `pkg_resources` compatibility
module and pip 26 misclassifies the official Ninja wheel on the target host.

## Stable CLI

Run all commands from `CLLM-src/`:

```bash
python -m llm_cdeq.prepare_states --config configs/llm_cdeq/gsm8k.yaml

python -m llm_cdeq.train --config configs/llm_cdeq/gsm8k.yaml --init 0 --ct 0
python -m llm_cdeq.train --config configs/llm_cdeq/gsm8k.yaml --init 1 --ct 0
python -m llm_cdeq.train --config configs/llm_cdeq/gsm8k.yaml --init 0 --ct 1
python -m llm_cdeq.train --config configs/llm_cdeq/gsm8k.yaml --init 1 --ct 1

python -m llm_cdeq.evaluate \
  --config configs/llm_cdeq/gsm8k.yaml \
  --checkpoint /home/ljc/experiments/cllm-cdeq/init1_ct1_seed42/best.pt \
  --mode both

python -m llm_cdeq.profile \
  --config configs/llm_cdeq/gsm8k.yaml --method cllm
python -m llm_cdeq.profile \
  --config configs/llm_cdeq/gsm8k.yaml --method cdeq \
  --checkpoint /home/ljc/experiments/cllm-cdeq/init1_ct1_seed42/best.pt
```

The official reproduction suite is:

```bash
bash scripts/reproduce_gsm8k.sh configs/llm_cdeq/gsm8k.yaml
```

It runs a one-example demo, full GSM8K accuracy, the 500-example CLLM/AR speed
profile, and 100-block Abel vanilla-Jacobi/greedy-AR endpoint equivalence.
After staging each Hub local directory, `python -m llm_cdeq.verify_artifacts`
checks every file against the exact revision and Hub Git/LFS etag and writes a
machine-readable verification manifest.
Pass the matching pinned manifest under `configs/llm_cdeq/artifacts/`; this also
rejects missing shards, unexpected files, and wrong byte sizes. For example:

```bash
python -m llm_cdeq.verify_artifacts \
  --root /home/ljc/models/cllm/Abel-7B-001 \
  --revision 3439c5a654dac2320d228d11a0c5590346e81d1a \
  --manifest configs/llm_cdeq/artifacts/abel-7b-001.json \
  --output /home/ljc/experiments/cllm-cdeq/artifacts/abel-7b-001.json
```

The resumable local-to-remote artifact queue can run in a detached
`cllm-artifacts` screen session. Its status command is safe to run repeatedly:

```bash
bash /Users/landrarwolf/Documents/CDEQ/CLLM-src/scripts/cllm_artifact_status.sh
```

The queue downloads one Abel shard at a time to respect local disk limits,
checks the pinned size and SHA-256, retries interrupted rsync transfers, verifies
the remote digest, and only then removes the local temporary weight. It writes
`/private/tmp/cllm-stage/download.done` only after both complete model manifests
pass on the remote server.

## Cache contract

`prepare_states` converts every token state `y^(k)` to the frozen Abel final
hidden slice

```text
hidden[prompt_len - 1 : prompt_len + block_len - 1].
```

The frozen LM head must map that slice to `y^(k+1)`; the final slice must map
back to the endpoint. The official `augTrue` JSON stores repeated/interleaved
candidate states rather than one chronological list. Cache construction
deduplicates those candidates, evaluates the frozen Abel one-step transition,
and retains the longest strictly aligned path ending at the self-mapping
endpoint. Off-path augmented candidates are recorded and discarded; a block
with no valid path is rejected. Cache construction first counts complete
`data_id` groups and assigns a seeded, group-level hashed split without ever
splitting a problem across train/validation.

Each 128-example safetensors shard contains:

- `states [N,K,16,4096]` in BF16;
- `state_mask [N,K]` and per-example rho `time_grid [N,K]`;
- `trajectory_tokens [N,K,16]` and `endpoint_tokens [N,16]`;
- `token_mask [N,16]`, keeping the first EOS and masking its tail.

The cache builder hashes complete `data_id` groups into train/validation and
round-robins across groups while selecting blocks. In the first sampling round,
each problem contributes at most one accepted block; later rounds are used only
when a requested limit exceeds the available group count. This prevents a few
very large augmented groups from dominating a subset while preserving strict
cross-split isolation.

The split manifest records schema version, complete config and revisions, split
hash, shard counts, alignment rejection counts, and metadata JSONL. The frozen
LM-head weight is cached once outside the shards so adapter training never
loads the 7B backbone.

## Model and checkpoint contract

The default adapter uses `4096 → 512`, a tokenwise `(512+1) → 1536 → 512`
updater, and `512 → 4096`. Its output is

```text
s_hat = s + (1 - t/T) * up(G(down(s), t/T) - down(s)),
```

which is exactly identity at `t=T`. The initializer has its own optimizer,
direct endpoint SmoothL1 supervision, and is detached before entering the
consistency updater. The discrete objective is
`0.1 × adjacent EMA MSE + 0.9 × endpoint SmoothL1`; CT replaces discrete
states by interpolated hidden states with progressive `r<t` sampling.

Checkpoints are versioned packages, not bare state dicts. They contain online
and EMA weights, both optimizer states, full config and upstream revisions,
data split hash, parameter count, epoch/global step, and best validation
metrics. Evaluation refuses upstream or data-split mismatches.

## Staged experiment commands

First, overfit 64 blocks for all variants:

```bash
for init in 0 1; do
  for ct in 0 1; do
    python -m llm_cdeq.train --config configs/llm_cdeq/gsm8k.yaml \
      --init "$init" --ct "$ct" --train-limit 64 --validation-limit 64 \
      --epochs 40 --overfit --output-dir /home/ljc/experiments/cllm-cdeq/overfit
  done
done
```

Then run the 2k/512 baseline. The CLI exposes `--rank`, `--learning-rate`,
`--local-weight`, `--token-ce-weight`, `--init-learning-rate`, `--init-steps`,
`--ct-q`, `--ct-d`, and `--seed` for the fixed search spaces in the plan. After
selecting one baseline configuration, use it unchanged across all four 10k/1k
ablations. Only baseline and Init+CT are repeated with seeds 42/43/44 after the
feasibility gate passes.

## Tests

```bash
python -m pytest -q tests/llm_cdeq
python -m py_compile llm_cdeq/*.py
```

Tests cover time-grid endpoints, continuous interpolation, progressive pair
ordering, EOS masks, hidden shift, exact `t=T` identity, initializer detach,
EMA, parameter budget, safetensors manifests, and checkpoint round-trip.

## Current feasibility result

The first preregistered 2k/512 run did not pass the baseline gate after the full
rank/LR/loss-weight search, so the 10k/1k four-way ablation was not started.
See [`reports/FEASIBILITY_REPORT.md`](reports/FEASIBILITY_REPORT.md) for exact
metrics, the negative-result diagnosis, compact official reproduction evidence,
and the generated trajectory/training curves.
