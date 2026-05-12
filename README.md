# Consistency DEQ

This repository provides the official codebase for **Consistency DEQ**. It includes three tasks across four datasets. The implementation centers on **Consistency Model** based inference as the primary method.



---

## 1. Tasks & Datasets

**Task A — Sequence Modeling**
- Dataset: **wikitext-103**

**Task B — Graph Node Classification**
- Datasets: **ogbn-arxiv**, **ogbn-products**

**Task C — Vision Classification**
- Dataset: **ImageNet**

Total: **3 tasks / 4 datasets**.

---

## 2. Repository Structure

```
code/
  DEQ-Sequence/             # Sequence task (wikitext-103)
  IGNN/                     # Graph tasks (ogbn-arxiv, ogbn-products)
  MDEQ/                     # Vision task (ImageNet)
  cm_plugin/                # CM plugin interface and refiner loader
  cm_checkpoints/           # CM checkpoints per task/dataset
  README.md                 # This document
```

---

## 3. Inference Commands (Four Datasets on Three Tasks)

All commands are assumed to be executed from the repository root.

### 3.1 Sequence — wikitext-103

**DEQ baseline inference**
```
python DEQ-Sequence/train_transformer.py --eval --data ./DEQ-Sequence/data/wikitext-103
```

**CM inference (wikitext-103)**
```
python DEQ-Sequence/train_transformer.py \
  --eval \
  --data ./DEQ-Sequence/data/wikitext-103 \
  --cm_enable \
  --cm_load cm_checkpoints/sequence/wt103/best_cm_model.pth
```

### 3.2 Graph — ogbn-arxiv

**DEQ baseline inference**
```
python IGNN/train_IGNN_ogbn_arxiv.py --inference
```

**CM inference (ogbn-arxiv)**
```
python IGNN/train_IGNN_ogbn_arxiv.py \
  --inference \
  --cm_enable \
  --cm_load cm_checkpoints/graph/ogbn-arxiv/best_cm_model.pth
```

### 3.3 Graph — ogbn-products

**DEQ baseline inference**
```
python IGNN/train_IGNN_ogbn_products.py --inference
```

**CM inference (ogbn-products)**
```
python IGNN/train_IGNN_ogbn_products.py \
  --inference \
  --cm_enable \
  --cm_load cm_checkpoints/graph/ogbn-products/best_cm_model.pth
```

### 3.4 Vision — ImageNet

**DEQ baseline inference**
```
python MDEQ/tools/cls_valid.py --cfg MDEQ/experiments/imagenet/cls_mdeq_SMALL.yaml
```

**CM inference (ImageNet)**
```
python MDEQ/tools/cls_valid.py \
  --cfg MDEQ/experiments/imagenet/cls_mdeq_SMALL.yaml \
  --opts CM.ENABLE True CM.LOAD_PATH cm_checkpoints/vision/imagenet/best_cm_model.pth
```

---

## 4. Data Description

- **wikitext-103**: Large-scale word-level language modeling benchmark.
- **ogbn-arxiv / ogbn-products**: Graph datasets from OGB for node classification.
- **ImageNet**: Image classification benchmark used for vision task.





## 5. Environment Setup

- Python: **3.10**
- PyTorch: **1.13.0**
- NumPy: **1.26.4**
- pandas: **2.2.1**
- SciPy: **1.11.3**
- scikit-learn: **1.3.0**
- torch_scatter: **2.1.1**
- torch_sparse: **0.6.17**
- torch_geometric: **1.6.1**

---

