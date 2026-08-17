# Resource Aware Multi Branch Channel Selection for UAV Wildlife Monitoring

This repository accompanies the paper **"Resource Aware Multi Branch Channel Selection for UAV Wildlife Monitoring"**. The work proposes a lightweight UAV-ready vision framework for wildlife classification and tracking under strict onboard constraints such as memory, latency, and power.

The main idea is to learn an efficient internal network structure during training and then deploy a **static fused model** at inference time. This avoids runtime controller overhead while keeping the model suitable for real-time UAV and edge-device deployment.

---

## Overview

UAV-based wildlife monitoring is useful for ecological surveys, animal activity analysis, species recognition, and animal tracking. However, UAV platforms usually have limited compute, memory, and battery power. Large vision models can perform well, but they are often too heavy for onboard deployment.

This project introduces a compact backbone called **RepBlockMask**, trained with a lightweight reinforcement-learning controller called **Coach**. During training, the Coach selects branch configurations using **Proximal Policy Optimization (PPO)**. After training, the selected configuration is structurally fused into a normal convolutional network for fast and predictable inference.

The key property of RepBlockMask is that its branches own **disjoint channel groups** and their outputs are **concatenated**, not summed. Because of this, masking a branch removes its channels from the block output, so the selected configuration directly changes the deployed width, parameter count, and FLOPs. This is what makes the resource-awareness claim measurable rather than nominal.

---

## Key Features

- **RepBlockMask backbone** with three parallel branches over disjoint channel budgets:
  - `3x3 convolution` for spatial features
  - `1x1 convolution` for channel mixing
  - `identity branch` that forwards a strided channel slice of the block input

- **Concatenative reparameterization**: the block output is the concatenation of the active branches, so a masked branch reduces the deployed output width.

- **Coach-PPO controller** for training-time structure selection.

- **Static fused deployment** with no controller, no PPO rollout, and no dynamic routing at inference.

- Supports multiple UAV wildlife vision tasks:
  - Animal activity classification
  - Wild animal species recognition
  - Single-object tracking
  - Multi-object tracking

- Designed for edge/UAV deployment with predictable latency and reduced model size.

---

## Proposed Pipeline

The framework follows four main stages:

1. **Warm-start supervised training**
   The Student model is first trained using the full branch configuration so that all branches learn useful weights.

2. **Coach-PPO exploration**
   A lightweight Coach controller selects branch masks for each RepBlockMask stage. The reward balances task quality, structural cost, and stability.

3. **Sweet-spot selection and fine-tuning**
   The best configuration is selected based on validation reward and fine-tuned under a fixed structure.

4. **Structural fusion and deployment**
   Active branches are fused into standard convolution layers, producing a static model that is exported to ONNX for measurement and deployment.

---

## Architecture Summary

```text
Input Image / Frame
        |
        v
Stem Convolution
        |
        v
RepBlockMask Stage 1
        |
        v
RepBlockMask Stage 2
        |
        v
RepBlockMask Stage 3
        |
        v
Task-Specific Head
  | Classification Head
  | MOT Detection + ReID Head
  | SOT Siamese Tracking Head
```

### Channel schedule and branch budgets

The backbone uses a compact channel schedule (Stem -> Stage 1 -> Stage 2 -> Stage 3) of `{32, 64, 128, 256}`, with channel counts rounded to small divisors (typically 8) for better memory alignment on CPU/SIMD and edge-NPU kernels.

Within stage `i` of width `C_i`, each branch is given its own channel budget:

```text
n_3x3^(i) + n_1x1^(i) + n_id^(i) = C_i
```

The identity branch forwards a strided channel slice of the block input, sized to `n_id^(i)`, which preserves a gradient pathway without adding parameters. The per-stage budgets are set in the config files under `configs/`.

When a branch is masked off, its channels are dropped from the concatenation, so the stage output width becomes the sum of the budgets of the **active** branches only. The identity budget is clamped so that identity-only presets remain a valid residual path.

### Stage presets

Each RepBlockMask stage can use one of the following presets:

| Preset | Mask `(3x3, 1x1, identity)` | Meaning |
|---|---|---|
| 0 | `(1, 1, 1)` | Full branch mode |
| 1 | `(1, 1, 0)` | 3x3 + 1x1 branches |
| 2 | `(1, 0, 0)` | 3x3 only |
| 3 | `(0, 1, 0)` | 1x1 only |
| 4 | `(0, 0, 1)` | Identity only |

A model configuration is written as a per-stage preset triple. The configuration selected by the Coach for the classification backbone is **(0, 1, 2)**, meaning Stage 1 uses the full branch mode, Stage 2 drops the identity branch, and Stage 3 keeps only the 3x3 branch.

---

## Datasets Used

The experiments use four wildlife/UAV-related datasets:

| Dataset | Task | Description |
|---|---|---|
| Sheep Activity Dataset | Classification | Classifies sheep activities such as grazing, running, and sitting |
| NESTLER Wild Animal Recognition Dataset | Classification | Recognizes wildlife species such as fox, jackal, and vulture |
| VisDrone-SOT2019 | Single-object tracking | UAV-based single target tracking benchmark |
| AnimalTrack | Multi-object tracking | Multi-animal tracking in natural scenes |

> Note: Datasets are not included in this repository. Please download them from their official sources and place them in the expected dataset folders.

### Split protocol

For Sheep Activity and NESTLER, each source video is assigned **in its entirety** to one partition, stratified by class with a fixed random seed, and frames are then sampled at 2 fps. This avoids near-duplicate frames leaking between train and test. The resulting frame counts are:

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| Sheep Activity | 3,171 | 389 | 407 |
| NESTLER | 3,658 | 975 | 1,175 |
| AnimalTrack | 12,481 | 2,223 | 1,929 |
| VisDrone-SOT2019 | 18,685 | 7,068 | 5,572 |

AnimalTrack and VisDrone-SOT2019 follow their official partitions.

### Input resolutions

| Task | Input |
|---|---|
| Classification | `224 x 224`, ImageNet mean/std, random horizontal flip and mild color jitter |
| SOT (VisDrone-SOT2019) | `127 x 127` template, `255 x 255` search, random translation and scale jitter |
| MOT (AnimalTrack) | `320 x 320`, mosaic, random horizontal flip, HSV color jitter |

---

## Main Results

All parameter and FLOP figures below are **measured from the exported ONNX graph at FP16**, not from the training graph. FLOPs are counted as twice the number of multiply-accumulate operations.

### Classification

Both classification datasets share the same backbone and the same selected configuration `(0, 1, 2)`, so the cost figures are reported once and apply to both rows.

| Dataset | Accuracy | Parameters | FLOPs |
|---|---:|---:|---:|
| Sheep Activity | 0.934 | 0.222M | *(fill from ONNX)* |
| NESTLER | 0.924 | 0.222M | *(fill from ONNX)* |

### Single-Object Tracking: VisDrone-SOT2019

| Metric | Value |
|---|---:|
| Success AUC | 42.5 |
| DP@20 | 65.0 |
| NP-AUC | 60.3 |
| Parameters | 3.4M |
| FLOPs | 1.113G |

### Multi-Object Tracking: AnimalTrack

| Metric | Value |
|---|---:|
| MOTA | 42.6 |
| IDF1 | 40.8 |
| Parameters | 6.3M |
| FLOPs | 4.03G |

> MOT cost figures correspond to a `320 x 320` input. Baselines reported at higher input resolutions are not directly comparable on cost.

### Seeds

The proposed model is trained and evaluated with three random seeds (42, 123, 456) on each task. Aggregate metrics are reported as mean ± std across runs; per-class and per-track statistics are reported for the median-performing seed.

### On-device Inference

The fused models were evaluated on an NVIDIA Jetson Orin Nano in 15 W mode. Training and offline evaluation used a single NVIDIA RTX 3050 with 6 GB VRAM under mixed precision.

---

## Repository Structure

```text
.
├── configs/                  # YAML/JSON experiment configs
│   ├── sheep_cls.yaml
│   ├── nestler_cls.yaml
│   ├── visdrone_sot.yaml
│   └── animaltrack_mot.yaml
│
├── datasets/                 # Dataset root folders, not committed to Git
│   ├── sheep_activity/
│   ├── nestler/
│   ├── visdrone_sot2019/
│   └── animaltrack/
│
├── models/
│   ├── repblockmask.py       # RepBlockMask implementation (concatenative)
│   ├── backbone.py           # Student backbone
│   ├── coach.py              # Coach PPO policy network
│   ├── heads.py              # Classification, SOT, and MOT heads
│   └── fusion.py             # Structural reparameterization utilities
│
├── train/
│   ├── train_classification.py
│   ├── train_sot.py
│   ├── train_mot.py
│   └── train_coach_ppo.py
│
├── evaluate/
│   ├── eval_classification.py
│   ├── eval_sot.py
│   └── eval_mot.py
│
├── tools/
│   ├── export_onnx.py
│   ├── fuse_model.py
│   ├── measure_cost.py       # Params/FLOPs from the exported ONNX graph
│   └── benchmark_jetson.py
│
├── checkpoints/              # Trained checkpoints, not committed to Git
├── results/                  # Logs, plots, and evaluation outputs
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment

Using Conda:

```bash
conda create -n uav-wildlife python=3.10 -y
conda activate uav-wildlife
```

Or using `venv`:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Example dependencies:

```text
torch
torchvision
numpy
opencv-python
pandas
scikit-learn
matplotlib
tqdm
pyyaml
onnx
onnxruntime
```

---

## Dataset Preparation

Place datasets in the `datasets/` directory:

```text
datasets/
├── sheep_activity/
├── nestler/
├── visdrone_sot2019/
└── animaltrack/
```

Recommended split format:

```text
dataset_name/
├── train/
├── val/
└── test/
```

For classification datasets, the folder structure can follow the standard ImageFolder format:

```text
sheep_activity/
├── train/
│   ├── grazing/
│   ├── running/
│   └── sitting/
├── val/
└── test/
```

Splits are made at the **video level** before frame extraction, so all frames from a given source video land in the same partition.

For tracking datasets, follow the official benchmark annotation format or convert annotations into the expected loader format used by this repository.

---

## Training

### Classification training

Optimizer AdamW, cosine LR with 5-epoch warm-up, batch 64, 80 epochs.

```bash
python train/train_classification.py \
  --config configs/sheep_cls.yaml \
  --dataset datasets/sheep_activity \
  --output runs/sheep_activity
```

For NESTLER:

```bash
python train/train_classification.py \
  --config configs/nestler_cls.yaml \
  --dataset datasets/nestler \
  --output runs/nestler
```

### Coach-PPO training

For classification, PPO is triggered at 15 update points after the warm-start stage, with 8 PPO optimization epochs per update over rollouts collected across five training epochs (120 PPO update epochs total). For tracking, PPO is triggered at 9 update points (72 PPO update epochs). These updates are offline-only and are not part of the deployed fused model.

```bash
python train/train_coach_ppo.py \
  --config configs/sheep_cls.yaml \
  --checkpoint runs/sheep_activity/warm_start.pth \
  --output runs/sheep_activity/coach_ppo
```

### Single-object tracking training

SGD (momentum 0.9), batch 32, 50 epochs over template/search pairs.

```bash
python train/train_sot.py \
  --config configs/visdrone_sot.yaml \
  --dataset datasets/visdrone_sot2019 \
  --output runs/visdrone_sot
```

### Multi-object tracking training

AdamW, batch 16, 50 epochs, multi-task detection + ReID loss.

```bash
python train/train_mot.py \
  --config configs/animaltrack_mot.yaml \
  --dataset datasets/animaltrack \
  --output runs/animaltrack_mot
```

---

## Reward Configuration

Reward coefficients are set in the task configs. All coefficients are scaled relative to the performance weight (`alpha_acc` / `alpha_iou` = 1.0).

| Coefficient | Classification | MOT | SOT |
|---|---:|---:|---:|
| Failure weight | 0.5 | 0.7 | 1.0 |
| Confidence regularizer | 0.2 | 0.2 | 0.2 |
| Entropy regularizer | 0.1 | 0.1 | 0.1 |
| Structural cost `lambda` | 0.10 | 0.08 | 0.08 |
| Switching weight `mu` | — (0) | 0.05 | 0.10 |
| Jitter weight `kappa` | — | — | 0.10 |

The switching term applies only to the video tasks, where consecutive decision steps are temporally ordered. It is set to zero for classification, whose mini-batches carry no sequential structure.

---

## Structural Fusion for Deployment

After Coach-PPO training, select the best branch configuration and fuse the active branches:

```bash
python tools/fuse_model.py \
  --checkpoint runs/sheep_activity/coach_ppo/best.pth \
  --config configs/sheep_cls.yaml \
  --output runs/sheep_activity/fused_model.pth
```

The fused model is used for deployment. It does not require the Coach controller during inference. Because branches own disjoint channel groups, the fused model's width reflects only the active branches.

---

## Evaluation

### Classification evaluation

```bash
python evaluate/eval_classification.py \
  --config configs/sheep_cls.yaml \
  --checkpoint runs/sheep_activity/fused_model.pth \
  --dataset datasets/sheep_activity/test
```

Reports per-class Precision/Recall/F1, macro-F1, overall Accuracy, and threshold-free ROC-AUC and PR-AP under one-vs-rest aggregation. Confusion matrices are written from the same prediction file used for the metric tables.

### SOT evaluation

```bash
python evaluate/eval_sot.py \
  --config configs/visdrone_sot.yaml \
  --checkpoint runs/visdrone_sot/fused_model.pth \
  --dataset datasets/visdrone_sot2019
```

### MOT evaluation

```bash
python evaluate/eval_mot.py \
  --config configs/animaltrack_mot.yaml \
  --checkpoint runs/animaltrack_mot/fused_model.pth \
  --dataset datasets/animaltrack
```

---

## ONNX Export, Cost Measurement, and Edge Benchmarking

Cost figures reported in the paper are measured from the exported ONNX graph at FP16, so they reflect the deployed artifact rather than the training-time graph.

Export the fused model to ONNX:

```bash
python tools/export_onnx.py \
  --checkpoint runs/sheep_activity/fused_model.pth \
  --config configs/sheep_cls.yaml \
  --precision fp16 \
  --output exports/sheep_activity_fused.onnx
```

Measure parameters and FLOPs from the exported graph:

```bash
python tools/measure_cost.py \
  --onnx exports/sheep_activity_fused.onnx \
  --input-size 224 224
```

Benchmark on Jetson Orin Nano (15 W mode):

```bash
python tools/benchmark_jetson.py \
  --onnx exports/sheep_activity_fused.onnx \
  --input-size 224 224 \
  --batch-size 1
```

For MOT, use `--input-size 320 320`.

---

## Explainability

The paper uses task-aligned explainability outputs:

- **Grad-CAM** for classification
- **Siamese score maps** for single-object tracking
- **Center heatmaps** for multi-object tracking

These visualizations help verify whether the model is focusing on animal regions instead of irrelevant background patterns.

---

## Acknowledgement

This repository is based on research in lightweight deep learning, structural reparameterization, reinforcement-learning-guided architecture selection, and UAV-based ecological monitoring.
