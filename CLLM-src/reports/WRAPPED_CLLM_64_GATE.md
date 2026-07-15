# Official CLLM + CDEQ+ 64-block Gate

Date: 2026-07-15

## Scope

This gate validates the corrected composite operator only. It does not include
2k/512 training, full GSM8K evaluation, initializer/CT ablations, progress-head
training, or adaptive rollout.

Each inference round is:

```text
official CLLM forward -> canonical shifted hidden
-> causal Transformer corrector -> frozen official LM head
```

The next round must re-run the full official CLLM backbone from decoded tokens.

## Reproducible runtime

- Host: `pc-cot-120`
- Environment: `cllm-cdeq`, Python 3.10, PyTorch 2.1.2,
  Transformers 4.36.2
- GPU: one RTX 3090
- Attention backend: SDPA
- Official checkpoint: `cllm/consistency-llm-7b-math` revision
  `904a1eefdf8e33a3440ddea35a55dd75cead648c`
- Config: `configs/llm_cdeq/gsm8k_wrapped_cllm.yaml`

## Correctness gates

- Remote unit suite: 52 passed, 1 integration-only test skipped by default.
- Official 7B integration test: passed.
- Full backbone checksum freeze test: passed.
- Zero residual: exact official hidden/logits/tokens.
- `t=T`: exact official hidden/logits/tokens.
- `disable_adapter=True`: exact official hidden/logits/tokens.
- Candidate block KV cache does not change the reusable prompt cache.
- Initializer is skipped for this gate and is structurally limited to round 0.

Backbone checksum before and after the explicit freeze test:

```text
2d7c82daaa9ddd454712f4a44b1e07027f03e61fb5ee82da2d46cb3b6e4ed1fb
```

## Cache

Remote path:

```text
/home/ljc/data/cllm/hidden_state_cache/gsm8k_official_cllm_v1
```

- Schema: `llm_cdeq_official_cllm_hidden_cache_v1`
- Accepted: 64/64 blocks
- Hidden/token alignment: 100%
- Rejected: 0
- Trajectory lengths: 3 (20), 4 (17), 5 (13), 6 (9), 7 (4), 8 (1)
- Hidden shape: `[64,17,16,4096]`, BF16
- Total cache size including frozen LM head: 395MB

The old Abel cache was not read or overwritten.

## 200-step overfit result

| Metric | Official CLLM single step | Wrapped corrector | Change |
| --- | ---: | ---: | ---: |
| Endpoint relative error | 1.212116 | 0.890527 | -26.53% |
| Endpoint token agreement | 66.8945% | 83.6914% | +16.80 pp |
| Exact block match | 0.00% | 18.75% | +18.75 pp |
| Safe violation rate | - | 0.00% | pass |
| EOS collapse | - | 0.00% | pass |
| Repeated-block collapse | - | 0.00% | pass |

The best hidden error observed at the final planned validation was used. The
run completed exactly 200 optimizer steps and did not trigger patience.

## Parameter and NFE accounting

- Official CLLM backbone: 6,738,677,760 parameters
- Trainable corrector: 7,616,512 parameters
- Trainable fraction: 0.1130%
- Normal wrapped round: CLLM backbone NFE=1, corrector NFE=1
- Initializer NFE: 0 for this gate; later 0 or 1 per rollout
- Prompt prefill: reported separately

## Decision

All 64-block gates passed. Per the preregistered scope, the process stops here:
`long_training_allowed` remains false in the gate report. The next phase must be
planned separately before creating a 2k/512 cache or resuming adaptive few-step
research.
