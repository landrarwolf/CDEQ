# CDEQ Agent Notes

Last updated: 2026-07-02.

## Scope

- Maintained code path: `CDEQ-src/DEQ-Sequence`.
- Main entrypoint: `CDEQ-src/DEQ-Sequence/train_transformer.py`.
- Do not revive old forked entrypoints or shell wrappers. Use flags on `train_transformer.py`.
- Current local branch during this work: `codex/deq-local-run`.
- The old top-level `DEQ-Sequence/` copy and root `README.md` were removed intentionally during cleanup. `IGNN/` and `MDEQ/` are kept for later adaptation work.

## Core CDEQ Algorithm In This Code

1. Start from a pretrained DEQ Transformer and its fixed-point function `f_theta`.
   - Runtime artifacts expected locally/remotely:
     - `pretrained_wt103_deqtrans_v3.pkl`
     - `models/pretrained_deq_func.pth`
     - `data/wikitext-103/`

2. Generate a teacher solver trajectory.
   - Command mode: `--save-trajectory`.
   - Default solver is `picard`.
   - Picard trajectory update:
     - `z_{k+1} = f_theta(z_k, x)`.
   - `anderson` exists only for explicit solver experiments.
   - Broyden is intentionally not part of the maintained CDEQ path.
   - Saved trajectory entries contain:
     - `x_traj`
     - `func_args`
     - `trajectory_solver`
     - `dataset`
     - `f_thres`
     - `max_eval_steps`
   - `--save-trajectory` skips cached trajectory files when `dataset`,
     `trajectory_solver`, `f_thres`, and `max_eval_steps` match.
   - Use `--force-trajectory-regen` to delete and regenerate files under the
     same `--trajectory-prefix`.

3. Train the consistency model from cached trajectories.
   - Command mode: `--train-CM`.
   - `--trajectory-solver` must match the solver recorded in each trajectory file.
   - Mismatch is a hard error, not a warning.
   - `ConsistencyFunction` uses the same solver family as the saved trajectory:
     - Picard: train on `f(z_t)`.
     - Anderson: train on `anderson_step(z_t, z_{t-1}, f(z_t), f(z_{t-1}))`.
   - Current distillation loss is simplified:
     - local consistency: student output at current point matches EMA model at the previous point.
     - global consistency: output matches the final trajectory state.
     - code uses `0.1 * local + 0.9 * global`.

4. Run CM inference.
   - Command mode: `-CM` or `--CM`.
   - Loads `--cm-load`.
   - Sets CM solver from `--trajectory-solver`.
   - Default is one-step CM inference toward the equilibrium approximation.
   - `--cm-compare-teacher` is debug-only: it also runs the DEQ teacher solver and prints relative error, so it should not be used for speed measurements.

## Paper Match

The implementation matches the main CDEQ idea:

- Fix a deterministic solver trajectory toward a DEQ equilibrium.
- Cache teacher trajectory states.
- Distill a consistency function over that trajectory.
- Use global and local consistency with an EMA teacher.
- Replace many DEQ solver iterations with one/few CM inference steps.

Important differences from the paper:

- The paper's main description and experiments emphasize Anderson Acceleration teacher trajectories.
- This project now defaults to Picard by user decision. This is still a valid solver-induced trajectory, but it is not the paper's main AA configuration.
- The current sequence implementation is a simplified engineering path:
  - no full trajectory augmentation from the appendix,
  - no full task-level regularization term,
  - no complete multi-step inference time schedule,
  - fixed loss weights instead of configurable `lambda_1/lambda_2`.

Conclusion: treat this as a CDEQ-compatible implementation of the paper's core idea, not a complete reproduction of every paper detail.

## Standard Commands

Run from:

```sh
cd /home/ljc/Code/DEQ/DEQ-Sequence
```

Baseline evaluation:

```sh
python train_transformer.py --debug --gpu-count 1 --max_eval_steps 1
```

Generate default Picard trajectories:

```sh
python train_transformer.py \
  --debug \
  --gpu-count 1 \
  --save-trajectory \
  --deq-func-load ./models/pretrained_deq_func.pth \
  --trajectory-prefix traj_all \
  --f_thres 40
```

Train CM from trajectories:

```sh
python train_transformer.py \
  --debug \
  --gpu-count 1 \
  --train-CM \
  --trajectory-prefix traj_all \
  --cm-start-file-idx 1 \
  --cm-max-file-idx 9 \
  --cm-save best_CM_model.pth \
  --cm-checkpoint cm_checkpoint/cm_checkpoint.pt
```

Run CM inference:

```sh
python train_transformer.py \
  --debug \
  --gpu-count 1 \
  -CM \
  --cm-load best_CM_model.pth \
  --max_eval_steps 1
```

Tiny smoke test:

```sh
python train_transformer.py --debug --gpu-count 1 --save-trajectory \
  --trajectory-prefix /tmp/cdeq_refactor_smoke \
  --deq-func-load ./models/pretrained_deq_func.pth \
  --f_thres 4 --max_eval_steps 1

python train_transformer.py --debug --gpu-count 1 --train-CM \
  --trajectory-prefix /tmp/cdeq_refactor_smoke \
  --cm-start-file-idx 1 --cm-max-file-idx 1 \
  --cm-max-traj-per-file 1 --cm-num-samples 1 \
  --cm-epochs 1 --cm-batch-size 1 --cm-train-points 4 \
  --cm-save /tmp/cdeq_refactor_cm_smoke.pth \
  --cm-checkpoint /tmp/cdeq_refactor_cm_smoke.pt \
  --max_eval_steps 1

python train_transformer.py --debug --gpu-count 1 -CM \
  --cm-load /tmp/cdeq_refactor_cm_smoke.pth \
  --max_eval_steps 1
```

## Remote Runtime

- SSH alias: `pc-cot-120`.
- Remote project path: `/home/ljc/Code/DEQ/DEQ-Sequence`.
- Conda init path: `/opt/anaconda3/etc/profile.d/conda.sh`.
- Conda env: `IGNN`.
- Use this remote prefix:

```sh
ssh -o UpdateHostKeys=no pc-cot-120 \
  'cd /home/ljc/Code/DEQ/DEQ-Sequence && \
   source /opt/anaconda3/etc/profile.d/conda.sh && \
   conda activate IGNN && \
   PYTORCH_ALLOC_CONF=expandable_segments:True python train_transformer.py ...'
```

Remote smoke checks previously passed in `IGNN`:

- Picard trajectory generation with `--f_thres 4 --max_eval_steps 1`.
- Picard CM train with `1 sample / 1 epoch / batch 1`.
- Picard `-CM` inference from the temporary CM weight.
- Anderson trajectory/train/inference also works when explicitly selected.
- Picard trajectory plus Anderson training correctly fails with `Trajectory solver mismatch`.

## Sync Rules

Only sync code and docs. Do not sync or commit datasets, trajectories, checkpoints, or model weights.

Do not sync:

- `data/`
- `*.pt`
- `*.pth`
- `*.pkl`
- `cm_checkpoint/`
- Wikitext data

Preferred targeted sync from local:

```sh
cd /Users/landrarwolf/Documents/CDEQ/CDEQ-src
rsync -az -e 'ssh -o UpdateHostKeys=no' --relative \
  DEQ-Sequence/train_transformer.py \
  DEQ-Sequence/models/deq_transformer.py \
  DEQ-Sequence/models/deq_transformer_CD.py \
  DEQ-Sequence/README.md \
  pc-cot-120:/home/ljc/Code/DEQ/
```

Avoid broad `rsync --delete` for this project unless the user explicitly asks.

## Git And Safety Notes

- Check `git status --short --branch` before and after edits.
- Do not revert user-made deletions or unrelated dirty files.
- Keep generated trajectory and weight files out of Git.
- Previous useful commits:
  - `63877c8 Refactor DEQ sequence entrypoint`
  - `86e59f6 Tie CDEQ solver choice to trajectories`

## Quick Verification

Local static checks:

```sh
python -m py_compile \
  CDEQ-src/DEQ-Sequence/train_transformer.py \
  CDEQ-src/DEQ-Sequence/models/deq_transformer.py \
  CDEQ-src/DEQ-Sequence/models/deq_transformer_CD.py

git diff --check
rg --files CDEQ-src/DEQ-Sequence | rg '\.sh$'
```

Expected:

- `py_compile` passes.
- `git diff --check` has no whitespace errors.
- No `.sh` files remain under `CDEQ-src/DEQ-Sequence`.
