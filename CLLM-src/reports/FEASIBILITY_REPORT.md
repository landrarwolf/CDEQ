# CDEQ+-Jacobi GSM8K feasibility report

Date: 2026-07-15

This report records the first preregistered CDEQ+-to-CLLM feasibility run. Model
weights, hidden caches, checkpoints, raw trajectories, and full runtime logs remain
on `pc-cot-120`; only compact reports and plots are checked into Git.

## Reproduction and implementation gates

| Gate | Result |
| --- | --- |
| Abel-7B-001 artifact verification | Passed; all pinned files and shard hashes match |
| CLLM 7B Math artifact verification | Passed; all pinned files and shard hashes match |
| Jacobi dataset verification | Passed; SHA256 `252c568eca46c3e023881f97de7460ada3fb633f72fb8c9b8030ac5ee70ca696` |
| Official GSM8K demo | Passed; answer 500, 384 tokens, 103.15 tokens/s |
| Abel greedy AR vs vanilla Jacobi | Passed; 100/100 endpoints agree |
| Unit/integration tests | Passed; 28 tests |
| Attention backend | SDPA for all current quality/cache runs; FlashAttention unavailable |
| Full official accuracy | Running in remote screen `cllm-acc-full` |
| 500-sample speed profile | Running in remote screen `cllm-speed-500` |

The official full accuracy and speed gates are intentionally marked pending until
their background jobs finish. Speed is only compared between methods using the same
attention backend.

## Hidden-state cache

The first 2k/512 cache exposed a sampling flaw: record-count grouped splitting let
two huge augmented `data_id` groups dominate all 2,000 training blocks. The cache
builder was changed to use deterministic group-level hashing and round-robin sampling
across problems. The invalid cache was deleted and rebuilt.

| Property | Train | Validation |
| --- | ---: | ---: |
| Accepted blocks | 2,000 | 512 |
| Unique `data_id` | 2,000 | 512 |
| Cross-split overlap | 0 | 0 |
| Mean aligned trajectory length | 10.764 | 10.854 |

Additional checks:

- Six misaligned candidates were rejected before writing.
- Frozen LM-head transition alignment is 432,803/432,803 tokens for accepted data.
- States are BF16 `[N,17,16,4096]`.
- Every valid rho grid is monotonic from 0.002 to 5.0.
- Valid trajectory lengths cover 2 through 17 states.

## 64-block overfit gate

All four variants reduced endpoint error. The two configurations extended to 160
steps reached:

| Variant | Endpoint error reduction | Token agreement gain |
| --- | ---: | ---: |
| CDEQ baseline | 18.34% | +15.13 pp |
| CDEQ + CT | 48.51% | +57.67 pp |

At 40 steps, Init-only and Init+CT reduced endpoint error by 23.66% and 22.94%,
respectively. This confirms that the implementation can optimize all requested paths
on a tiny subset.

## 2k/512 baseline and search

The held-out identity baseline is endpoint relative error 1.011137 and endpoint
token agreement 43.1885%.

| Configuration | Endpoint error | Relative error gain | Token agreement | Token gain |
| --- | ---: | ---: | ---: | ---: |
| Default: r512, LR 3e-4, local 0.1 | 0.876771 | 13.29% | 45.0562% | +1.87 pp |
| CE 0.05 | 0.910936 | 9.91% | 46.6187% | +3.43 pp |
| CE 0.10 | 0.929982 | 8.03% | 46.7529% | +3.56 pp |
| CE 0.20 | 0.953601 | 5.69% | 46.7407% | +3.55 pp |
| Tuned: r512, LR 1e-3, local 0.1 | 0.862611 | 14.69% | 46.0205% | +2.83 pp |
| Tuned: r512, LR 1e-3, local 0.2 | **0.860156** | **14.93%** | 45.4224% | +2.23 pp |

Token CE improves token agreement somewhat but consistently harms hidden endpoint
error, so the selected setting keeps `token_ce_weight=0`. Successive halving covered
rank `{256,512,1024}`, LR `{1e-4,3e-4,1e-3}`, and local/global weights
`{0.1/0.9,0.2/0.8}`.

The selected adapter has 5,770,752 trainable parameters, 0.0856% of the frozen
6,738,677,760-parameter backbone.

## Failure diagnosis and stop decision

The best checkpoint also fails the gate on its training cache:

| Split | Identity error | Adapter error | Error gain | Identity token | Adapter token | Token gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 0.998723 | 0.838668 | 16.03% | 43.9875% | 47.1469% | +3.16 pp |
| Validation | 1.011137 | 0.860156 | 14.93% | 43.1885% | 45.4224% | +2.23 pp |

Because the training split itself does not reach the preregistered 20% / 5 pp gate,
the main issue is not conventional held-out overfitting. Data leakage is absent,
LM-head token alignment is exact for accepted trajectories, and trajectory coverage is
broad. The leading diagnosis is insufficient hidden approximation/optimization from
the current per-token bottleneck updater; representation normalization, cross-token
coupling, and updater/projection capacity are the next hypotheses.

Per the preregistered stop rule, this run does not expand to the 10k/1k cache or the
formal four-way held-out ablation. It is retained as a reproducible negative
feasibility result rather than spending more compute after the search table is
exhausted.

## Artifacts

- `feasibility-2k/teacher_trajectory.{json,csv,png}`: teacher error/agreement by step.
- `feasibility-2k/ablation_training_curves.png`: default and tuned baseline curves.
- `feasibility-2k/ablation.{csv,md}`: compact run table.
- `official-reproduction/abel_jacobi_equivalence_nocache.json`: 100-block endpoint check.
- `official-reproduction/official_demo.log` and `demo.jsonl`: official demo evidence.

Remote runtime roots:

- Cache: `/home/ljc/data/cllm/hidden_state_cache/gsm8k_v1`
- Experiments: `/home/ljc/experiments/cllm-cdeq`
- Source: `/home/ljc/Code/Consistency_LLM_CDEQ`
