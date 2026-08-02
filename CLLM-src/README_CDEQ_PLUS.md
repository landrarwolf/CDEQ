# CDEQ+-Jacobi on CLLM/GSM8K

This directory keeps the official CLLM source at upstream commit
`22775363f8563c63620e71f7a204e90a51d6a379`. The active implementation freezes
the official `cllm/consistency-llm-7b-math` checkpoint and executes, on every
refinement round:

```text
current tokens
  -> full official CLLM forward
  -> canonical shifted hidden
  -> causal Transformer residual corrector
  -> frozen official LM head
  -> next tokens
```

The next round always sends those tokens through the complete official CLLM
backbone again. The old Abel-cache/tokenwise-MLP implementation remains readable
only as a `legacy/failed diagnostic`; it is not a valid wrapped CLLM operator.
Neither path changes the original DEQ/CDEQ entrypoint.

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

Active official-CLLM wrapped gate:

```bash
python -m llm_cdeq.prepare_cllm_states \
  --config configs/llm_cdeq/gsm8k_wrapped_cllm.yaml \
  --limit 64 --device cuda --attention-backend sdpa

python -m llm_cdeq.train_wrapped \
  --config configs/llm_cdeq/gsm8k_wrapped_cllm.yaml \
  --init 0 --ct 0 --overfit --device cuda
```

The preserved 64-block command is an explicit train-set overfit gate. Without
`--overfit`, `train_wrapped` requires separate train and validation manifests.
It rejects Init/CT until the base wrapped gate passes, rejects the legacy Abel
cache schema, and never loads Abel as a fallback operator.

The held-out 512/128 ten-epoch pilot uses an isolated config and directories:

```bash
python -m llm_cdeq.prepare_cllm_states \
  --config configs/llm_cdeq/gsm8k_wrapped_cllm_pilot.yaml \
  --train-limit 512 --validation-limit 128 \
  --device cuda --attention-backend sdpa

python -m llm_cdeq.train_wrapped \
  --config configs/llm_cdeq/gsm8k_wrapped_cllm_pilot.yaml \
  --init 0 --ct 0 --epochs 10 --device cuda

python -m llm_cdeq.evaluate_wrapped \
  --config configs/llm_cdeq/gsm8k_wrapped_cllm_pilot.yaml \
  --checkpoint /home/ljc/experiments/cllm-cdeq/\
wrapped-official-cllm-512x128-10ep-seed42/best.pt \
  --mode both --weights both --sample-limit 8 --device cuda
```

Fresh training refuses a non-empty output directory. Resume only from the same
run's complete `last.pt`; `--epochs 10` remains the total target, not ten more
epochs. The first evaluator uses full-prefix re-prefill after each committed
block and is therefore a correctness harness, not a fair speed benchmark.

The 2026-08-02 pilot completed 10 epochs/640 steps on a disjoint 512/128 cache.
EMA passed every held-out cache gate: endpoint error improved 24.32%, token
agreement changed from 70.70% to 70.12%, and safety/EOS/repetition violations
were zero. The eight-question correctness smoke then scored 6/8 for the
963-backbone-call official fixed point, but 0/8 for official one-step, wrapped
online, and wrapped EMA. Wrapped online/EMA repeated within blocks on 7/8 and
8/8 questions respectively, while held-out exact block match remained 0.78%.
The engineering path is validated, but this Base one-step model is not an
end-to-end quality pass; Init/CT remains disabled pending a stronger Base.

The following commands are preserved only for reproducing the failed MLP-only
diagnostic and its checkpoints:

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

The Stage-A adaptive hypothesis has a separate cache-only oracle diagnostic:

```bash
python -m llm_cdeq.adaptive \
  --config configs/llm_cdeq/gsm8k.yaml \
  --checkpoint /home/ljc/experiments/cllm-cdeq/halving-phase1/\
r512_lr0.001_l0.2/init0_ct0_seed42/best.pt \
  --split validation --sample-limit 512 --max-calls 4 --weights ema \
  --output-file /home/ljc/experiments/cllm-cdeq/adaptive-oracle/2k-best.json
```

It continuously projects each student state onto the piecewise-linear teacher
trajectory and reports per-call endpoint error, token agreement, projection
progress/distance, call counts, and stop reasons. Split-hash mismatches are a
hard error unless `--allow-split-mismatch` explicitly marks a diagnostic smoke
run. The matched 512-block gate failed: calls 2--4 moved farther from the
endpoint and teacher trajectory, so progress-head and rollout training remain
disabled rather than being added to the existing checkpoint/training path.

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

## Active official CLLM cache contract

`prepare_cllm_states` reuses only each GSM8K prompt, `data_id`, and deterministic
initial `y0`. Starting from `y0`, it repeatedly runs the full official CLLM
single-step operator until token fixed point. Each state stores:

- `canonical_hidden [N,K,16,4096]` in BF16;
- official input/output token blocks and their state/EOS masks;
- endpoint hidden/tokens and the per-trajectory rho time grid;
- official CLLM/tokenizer revisions and attention backend;
- shard, LM-head, data-split, and full backbone checksums.

`LMHead(canonical_hidden)` must match the official Jacobi-shifted tokens at every
valid state. The schema is
`llm_cdeq_official_cllm_hidden_cache_v1`, under
`/home/ljc/data/cllm/hidden_state_cache/gsm8k_official_cllm_v1`; the wrapped
trainer refuses the legacy Abel schema.

## Legacy Abel cache contract

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

## Active wrapped model and checkpoint contract

The active residual corrector is:

```text
4096 -> 512
  + learned time embedding
  + learned 16-position embedding
  + 1 causal self-attention block (8 heads)
  + FFN 512 -> 2048 -> 512
512 -> 4096 (zero initialized)
```

For the official CLLM single-step hidden `b`, it computes
`A(b,t)=b+(1-t/T)Delta(b,t)`. Both the corrector and wrapper explicitly bypass
all adapter work at `t=T`, and `disable_adapter` bypasses initializer and
corrector. The block's first hidden position is preserved because its token is
already fixed by prompt prefill. The initializer, when later enabled, may run
only at round zero and is detached before the corrector.

The short-gate objective is
`0.1 L_local + 0.9 L_endpoint + 0.1 L_safe + 0.05 L_token`, with
`safe_margin=0`. Wrapped checkpoints use
`llm_cdeq_wrapped_checkpoint_v1` and include corrector/EMA/optimizer weights,
the full config, split hash, parameter counts, best metrics, and backbone
checksums. The official backbone and LM head are frozen and absent from the
optimizer.

## Legacy MLP checkpoint contract

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

## Legacy staged experiment commands

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

Tests additionally cover the official single-step canonical shift, immutable
prompt KV cache, zero/terminal/disable exact equivalence, per-round backbone NFE,
first-round-only initializer behavior, causal block interaction, official-cache
schema isolation, wrapped checkpoint round-trip, safe/token losses, and full
official-backbone checksum preservation. The 7B integration test is opt-in with
`RUN_OFFICIAL_CLLM_TESTS=1`; the checksum variant also sets
`RUN_FULL_BACKBONE_CHECKSUM=1`.

## Current feasibility result

The active official-CLLM 64-block gate passed on 2026-07-15 with SDPA:

- official cache alignment: 100%, 64/64 accepted, trajectory length 3--8;
- zero residual, `t=T`, and `disable_adapter`: exact official-step equality;
- endpoint relative error: `1.212116 -> 0.890527` (26.53% improvement);
- endpoint token agreement: `66.8945% -> 83.6914%`;
- samplewise safe violation, EOS collapse, repeated-block collapse: all 0%;
- corrector: 7,616,512 parameters, 0.1130% of the 6,738,677,760-parameter
  official CLLM backbone;
- full official backbone checksum before/after the freeze test:
  `2d7c82daaa9ddd454712f4a44b1e07027f03e61fb5ee82da2d46cb3b6e4ed1fb`.

The run stopped at the planned 200 steps. It did not start 2k/512 training or
adaptive progress-head work. See
[`reports/WRAPPED_CLLM_64_GATE.md`](reports/WRAPPED_CLLM_64_GATE.md).

For historical context, the legacy MLP-only feasibility result was negative:

The first preregistered 2k/512 run did not pass the baseline gate after the full
rank/LR/loss-weight search, so the 10k/1k four-way ablation was not started.
See [`reports/FEASIBILITY_REPORT.md`](reports/FEASIBILITY_REPORT.md) for exact
metrics, the negative-result diagnosis, compact official reproduction evidence,
and the generated trajectory/training curves.
