# DEQ-Sequence

This directory keeps the Wikitext-103 DEQ/CDEQ sequence path. The maintained
entrypoint is:

```sh
python train_transformer.py
```

Old forked scripts were removed. Use command flags instead of editing comments
inside the code.

CDEQ trajectory generation, CM training, and CM inference must use the same
`--trajectory-solver`. The default is `picard`; `anderson` is only for explicit
solver experiments.
The legacy Broyden solver is intentionally not exposed in this maintained path.
Trajectory generation skips an existing cache when `dataset`, solver, `f_thres`,
and `max_eval_steps` match; use `--force-trajectory-regen` to overwrite it.

## Required Local Artifacts

These files are runtime inputs and must stay out of Git:

- `data/wikitext-103/`
- `pretrained_wt103_deqtrans_v3.pkl`
- `models/pretrained_deq_func.pth`
- generated `*.pt` trajectory files
- generated `*.pth` CM weights

## Baseline Evaluation

```sh
python train_transformer.py \
  --debug \
  --gpu-count 1 \
  --max_eval_steps 1
```

## CDEQ Workflow

Generate Picard trajectories from the pretrained DEQ function:

```sh
python train_transformer.py \
  --debug \
  --gpu-count 1 \
  --save-trajectory \
  --deq-func-load ./models/pretrained_deq_func.pth \
  --trajectory-prefix traj_all \
  --f_thres 40
```

Train the consistency model from saved trajectories:

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

Run one-step CM inference:

```sh
python train_transformer.py \
  --debug \
  --gpu-count 1 \
  -CM \
  --cm-load best_CM_model.pth \
  --max_eval_steps 1
```

By default `-CM` runs only the consistency model. Add
`--cm-compare-teacher` when you also want to run the DEQ teacher solver and
print the relative error for debugging.

## Smoke Test

For a fast implementation check, use a temporary prefix and tiny limits:

```sh
python train_transformer.py --debug --gpu-count 1 --save-trajectory \
  --trajectory-prefix /tmp/cdeq_refactor_smoke \
  --deq-func-load ./models/pretrained_deq_func.pth --f_thres 4 --max_eval_steps 1

python train_transformer.py --debug --gpu-count 1 --train-CM \
  --trajectory-prefix /tmp/cdeq_refactor_smoke --cm-start-file-idx 1 \
  --cm-max-file-idx 1 --cm-max-traj-per-file 1 --cm-num-samples 1 \
  --cm-epochs 1 --cm-batch-size 1 --cm-train-points 4 \
  --cm-save /tmp/cdeq_refactor_cm_smoke.pth \
  --cm-checkpoint /tmp/cdeq_refactor_cm_smoke.pt --max_eval_steps 1

python train_transformer.py --debug --gpu-count 1 -CM \
  --cm-load /tmp/cdeq_refactor_cm_smoke.pth --max_eval_steps 1
```
