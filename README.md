# Resource-Aware-Multi-Branch-Feature-Fusion-for-UAV-Wildlife-Monitoring

# Resource-Aware Multi-Branch Feature Fusion for UAV Wildlife Monitoring

This repository accompanies the paper **"Resource Aware Multi Branch Feature Fusion for UAV Wildlife Monitoring"**. The work proposes a lightweight UAV-ready vision framework for wildlife classification and tracking under strict onboard constraints such as memory, latency, and power.

The main idea is to learn an efficient internal network structure during training and then deploy a **static fused model** at inference time. This avoids runtime controller overhead while keeping the model suitable for real-time UAV and edge-device deployment.

---

## Overview

UAV-based wildlife monitoring is useful for ecological surveys, animal activity analysis, species recognition, and animal tracking. However, UAV platforms usually have limited compute, memory, and battery power. Large vision models can perform well, but they are often too heavy for onboard deployment.

This project introduces a compact backbone called **RepBlockMask**, trained with a lightweight reinforcement-learning controller called **Coach**. During training, the Coach selects efficient branch configurations using **Proximal Policy Optimization (PPO)**. After training, the selected configuration is structurally fused into a normal convolutional network for fast and predictable inference.

---

## Key Features

- **RepBlockMask backbone** with three parallel branches:
  - `3x3 convolution` for spatial features
  - `1x1 convolution` for channel mixing
  - `identity branch` for lightweight feature preservation

- **Coach-PPO controller** for training-time structure selection.

- **Static fused deployment** using RepVGG-style structural reparameterization.

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
   Active branches are fused into standard convolution layers, producing a static model with no Coach, no PPO rollout, and no dynamic routing at inference.

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

Each RepBlockMask stage can use one of the following presets:

| Preset | Mask `(3x3, 1x1, identity)` | Meaning |
|---|---|---|
| 0 | `(1, 1, 1)` | Full branch mode |
| 1 | `(1, 1, 0)` | 3x3 + 1x1 branches |
| 2 | `(1, 0, 0)` | 3x3 only |
| 3 | `(0, 1, 0)` | 1x1 only |
| 4 | `(0, 0, 1)` | Identity only |

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

---

## Main Results

### Classification

| Dataset | Accuracy | Parameters | FLOPs |
|---|---:|---:|---:|
| Sheep Activity | 0.934 | 0.23M | 1.073G |
| NESTLER | 0.924 | 0.103M | 0.48G |

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

### On-device Inference

The fused models were evaluated on an NVIDIA Jetson Orin Nano in 15 W mode. The proposed fused model achieved fast edge inference across classification and tracking tasks, showing that structural fusion helps convert the training-time adaptive model into a deployment-friendly static graph.

---

## Repository Structure

A suggested structure is shown below. Update file names according to your implementation.

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
│   ├── repblockmask.py       # RepBlockMask implementation
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

For tracking datasets, follow the official benchmark annotation format or convert annotations into the expected loader format used by this repository.

---

## Training

### Classification training

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

```bash
python train/train_coach_ppo.py \
  --config configs/sheep_cls.yaml \
  --checkpoint runs/sheep_activity/warm_start.pth \
  --output runs/sheep_activity/coach_ppo
```

### Single-object tracking training

```bash
python train/train_sot.py \
  --config configs/visdrone_sot.yaml \
  --dataset datasets/visdrone_sot2019 \
  --output runs/visdrone_sot
```

### Multi-object tracking training

```bash
python train/train_mot.py \
  --config configs/animaltrack_mot.yaml \
  --dataset datasets/animaltrack \
  --output runs/animaltrack_mot
```

---

## Structural Fusion for Deployment

After Coach-PPO training, select the best branch configuration and fuse the active branches:

```bash
python tools/fuse_model.py \
  --checkpoint runs/sheep_activity/coach_ppo/best.pth \
  --config configs/sheep_cls.yaml \
  --output runs/sheep_activity/fused_model.pth
```

The fused model is used for deployment. It does not require the Coach controller during inference.

---

## Evaluation

### Classification evaluation

```bash
python evaluate/eval_classification.py \
  --config configs/sheep_cls.yaml \
  --checkpoint runs/sheep_activity/fused_model.pth \
  --dataset datasets/sheep_activity/test
```

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

## ONNX Export and Edge Benchmarking

Export the fused model to ONNX:

```bash
python tools/export_onnx.py \
  --checkpoint runs/sheep_activity/fused_model.pth \
  --config configs/sheep_cls.yaml \
  --output exports/sheep_activity_fused.onnx
```

Benchmark on Jetson Orin Nano:

```bash
python tools/benchmark_jetson.py \
  --onnx exports/sheep_activity_fused.onnx \
  --input-size 224 224 \
  --batch-size 1
```

---

## Explainability

The paper uses task-aligned explainability outputs:

- **Grad-CAM** for classification
- **Siamese score maps** for single-object tracking
- **Center heatmaps** for multi-object tracking

These visualizations help verify whether the model is focusing on animal regions instead of irrelevant background patterns.









## Acknowledgement

This repository is based on research in lightweight deep learning, structural reparameterization, reinforcement-learning-guided architecture selection, and UAV-based ecological monitoring.
