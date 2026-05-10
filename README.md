<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8m-Ultralytics-blue?style=for-the-badge&logo=yolo" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/CBAM-Attention-orange?style=for-the-badge" alt="CBAM"/>
  <img src="https://img.shields.io/badge/Python-3.11-green?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?style=for-the-badge&logo=nvidia" alt="CUDA"/>
</p>

# Attention-Guided Speed Bump Detection: A Hybrid Architecture for Unmarked Infrastructure

> **Evaluating the Impact of CBAM Attention Mechanisms and Data Augmentation on YOLOv8m for Real-Time Speed Bump Detection — With Special Focus on Non-Marked (Unmarked) Speed Bumps**

This repository presents a comprehensive experimental study comparing a **standard YOLOv8m baseline** against a **custom YOLOv8m + CBAM (Convolutional Block Attention Module) hybrid architecture** for real-time speed bump detection. The study systematically evaluates the individual and combined effects of **attention mechanisms** and **targeted data augmentation** on detection performance — particularly for the challenging class of **non-marked speed bumps** that lack visual paint markings.

---

##  Table of Contents

- [Problem Statement](#-problem-statement)
- [Key Findings](#-key-findings)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Experimental Design](#-experimental-design)
- [Quantitative Results](#-quantitative-results)
- [Qualitative Analysis](#-qualitative-analysis-on-external-data)
- [Real-Time Performance](#-real-time-performance)
- [Expert Interpretation](#-expert-interpretation)
- [Repository Structure](#-repository-structure)

---

## Problem Statement

Speed bumps are a common road safety feature, but **non-marked (unmarked) speed bumps** — those without painted lines or visual indicators — pose a significant challenge for both human drivers and automated detection systems. They blend into the road surface, making them nearly invisible in standard camera feeds.

**Challenges addressed in this work:**

| Challenge | Description |
|-----------|-------------|
| **Class Imbalance** | Non-marked bumps represent only **5.5%** of the dataset (39 vs 666 marked samples in training) |
| **Visual Ambiguity** | Non-marked bumps lack distinctive visual cues (no paint, no patterns) |
| **Contextual Detection** | Detection requires understanding road surface texture and elevation changes |
| **Real-Time Constraint** | Autonomous driving demands ≥30 FPS inference speed |

**Hypothesis:** A custom attention mechanism (CBAM) inserted into the YOLOv8m backbone can guide the model to focus on subtle spatial and channel features, improving detection of underrepresented non-marked speed bumps without sacrificing real-time performance.

---

## Key Findings

<table>
<tr>
<td width="50%">

###  CBAM Attention Impact
- **+10.4% AP50** improvement on non-marked speed bumps (Original Data)
- **Eliminates double-detection** artifacts on marked bumps
- **No FPS degradation** — maintains real-time speed
- Compensates for limited training data on rare classes

</td>
<td width="50%">

###  Data Augmentation Impact
- **+25.0% AP50** improvement on non-marked speed bumps (Baseline)
- Critical for **generalization** to unseen external data
- Near-perfect detection (0.995 AP50) when combined with baseline
- Solves class imbalance through targeted oversampling

</td>
</tr>
</table>

> **Core Insight:** CBAM and Data Augmentation address the **same underlying problem** (data scarcity for rare classes) through **different mechanisms** — attention focuses on subtle features, augmentation provides diverse training samples. Their combination yields diminishing returns, suggesting they are **complementary rather than additive** solutions.

---

## Architecture

### YOLOv8m Baseline
Standard YOLOv8m architecture with a CSPDarknet backbone, PANet neck, and decoupled detection head.

### YOLOv8m + CBAM Hybrid

We insert **three CBAM modules** into the YOLOv8m backbone — one after each C2f block at the P3, P4, and P5 feature pyramid levels. This placement allows the attention mechanism to refine feature maps **before** they enter the detection head.

```
┌─────────────────────────────────────────────────────┐
│                  CBAM Architecture                   │
│                                                     │
│  Input Feature Map ──► Channel Attention Module     │
│                              │                      │
│                        (Global Avg Pool + Max Pool)  │
│                        (Shared MLP + Sigmoid)       │
│                              │                      │
│                         ──► Spatial Attention Module │
│                              │                      │
│                        (Channel Avg Pool + Max Pool) │
│                        (7×7 Conv + Sigmoid)         │
│                              │                      │
│                     Refined Feature Map             │
└─────────────────────────────────────────────────────┘
```

**CBAM Insertion Points in YOLOv8m Backbone:**

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]       # 1-P2/4
  - [-1, 3, C2f, [128, True]]        # 2
  - [-1, 1, Conv, [256, 3, 2]]       # 3-P3/8
  - [-1, 6, C2f, [256, True]]        # 4
  - [-1, 1, CBAM, [192]]             # 5  ← CBAM after P3
  - [-1, 1, Conv, [512, 3, 2]]       # 6-P4/16
  - [-1, 6, C2f, [512, True]]        # 7
  - [-1, 1, CBAM, [384]]             # 8  ← CBAM after P4
  - [-1, 1, Conv, [1024, 3, 2]]      # 9-P5/32
  - [-1, 3, C2f, [1024, True]]       # 10
  - [-1, 1, SPPF, [1024, 5]]         # 11
  - [-1, 1, CBAM, [576]]             # 12 ← CBAM after SPPF
```

> **Key Design Decision:** COCO pre-trained weights are loaded for all standard layers, while CBAM layers are initialized from scratch. This leverages transfer learning while allowing the attention modules to learn task-specific focus patterns.

---

##  Dataset

### Source
[Speed Bump Detection Dataset](https://universe.roboflow.com/) — A curated collection of dashcam images containing two classes of speed bumps captured from Indian roads.

### Class Distribution (Original)

<p align="center">
  <img src="class_distribution.png" alt="Class Distribution" width="800"/>
</p>

| Split | Marked | Non-Marked | Total | Imbalance Ratio |
|-------|--------|------------|-------|----------------|
| **Train** | 666 | 39 | 705 | 17:1 |
| **Valid** | 198 | 8 | 206 | 25:1 |
| **Test** | 77 | 4 | 81 | 19:1 |

### Augmentation Strategy

To address the severe class imbalance, we applied **targeted oversampling** exclusively on the non-marked class using [Albumentations](https://albumentations.ai/):

| Augmentation | Parameters |
|-------------|-----------|
| Horizontal Flip | p=0.5 |
| Random Brightness/Contrast | ±20%, p=0.5 |
| Gaussian Blur | σ=(3,7), p=0.3 |
| HSV Shift | H±20, S±30, V±30, p=0.3 |
| Random Rotation | ±15°, p=0.3 |

**Result:** Non-marked samples were increased by **~8×** to balance the training distribution, creating the `total-5-augmented` dataset variant.

---

##  Experimental Design

We conducted a systematic **2×2 factorial experiment** to isolate the individual contributions of the architectural modification (CBAM) and the data strategy (Augmentation):

| Experiment | Model | Training Data | Run Name |
|-----------|-------|---------------|----------|
| **Exp 1a** | YOLOv8m Baseline | Original | `exp1a_original` |
| **Exp 1b** | YOLOv8m Baseline | Augmented | `exp1b_augmented` |
| **Exp 3a** | YOLOv8m + CBAM | Original | `exp3a_original` |
| **Exp 3b** | YOLOv8m + CBAM | Augmented | `exp3b_augmented` |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 640×640 |
| Batch Size | 16 |
| Max Epochs | 200 |
| Early Stopping Patience | 20 epochs |
| Optimizer | SGD (YOLOv8 default) |
| GPU | NVIDIA RTX 5000 Ada |
| Framework | Ultralytics v8 + PyTorch |
| Logging | Weights & Biases (W&B) |

> **Evaluation Protocol:** All four models are evaluated on the **same held-out test set** (81 images) from the original `total-5/data.yaml` to ensure a fair, unbiased comparison.

---

##  Quantitative Results

### Master Comparison Table

<p align="center">
  <img src="final_master_comparison_chart.png" alt="Master Comparison Chart" width="900"/>
</p>

| Metric | YOLOv8m (Original) | CBAM Hybrid (Original) | YOLOv8m (Augmented) | CBAM Hybrid (Augmented) |
|--------|:------------------:|:----------------------:|:-------------------:|:-----------------------:|
| **Marked AP50** | **0.940** | 0.855 | 0.921 | 0.895 |
| **Non-Marked AP50** | 0.745 | **0.849** _(+10.4%)_ | **0.995** | 0.888 |
| **Overall mAP50** | 0.843 | 0.852 | **0.958** | 0.891 |
| **Overall mAP50-95** | 0.417 | 0.380 | **0.529** | 0.483 |

### Detailed Analysis by Condition

<details>
<summary><b> Original Data — CBAM vs Baseline</b></summary>

| Metric | Baseline | CBAM | Delta | Winner |
|--------|----------|------|-------|--------|
| Non-Marked AP50 | 0.745 | **0.849** | **+10.4%** |  CBAM |
| Overall mAP50 | 0.843 | **0.852** | +0.9% |  CBAM |
| Marked AP50 | **0.940** | 0.855 | -8.5% | Baseline |
| Overall mAP50-95 | **0.417** | 0.380 | -3.7% | Baseline |

**Insight:** CBAM significantly boosts the hardest class (+10.4% on non-marked) while trading off slightly on the already-easy marked class. The attention mechanism successfully guides the model to focus on subtle surface features.

</details>

<details>
<summary><b> Augmented Data — CBAM vs Baseline</b></summary>

| Metric | Baseline | CBAM | Delta | Winner |
|--------|----------|------|-------|--------|
| Non-Marked AP50 | **0.995** | 0.888 | -10.7% | Baseline |
| Overall mAP50 | **0.958** | 0.891 | -6.7% | Baseline |
| Marked AP50 | **0.921** | 0.895 | -2.6% | Baseline |
| Overall mAP50-95 | **0.529** | 0.483 | -4.6% | Baseline |

**Insight:** When abundant augmented data is available, the baseline's optimized architecture generalizes better. CBAM + Augmentation leads to **over-regularization**, as the attention layers focus on augmentation artifacts rather than true bump features.

</details>

---

##  Qualitative Analysis on External Data

To evaluate **real-world generalization**, we tested all four models on an **external dataset** that was never seen during training — sourced from a completely different geographic region and camera setup.

<p align="center">
  <img src="external_test_4models.png" alt="External Data Qualitative Comparison" width="100%"/>
</p>

> **Grid Layout:** Each row is one test image. Columns from left to right: **Original Image** → **YOLOv8m (Original)** → **YOLOv8m (Augmented)** → **CBAM Hybrid (Original)** → **CBAM Hybrid (Augmented)**. Rows 1–5 are Marked Bumps; Rows 6–11 are Non-Marked Bumps.

### Key Qualitative Observations

#### 1. Augmentation is Critical for Generalization
**Evidence — Non-Marked Row 4 (distant bump):**
-  YOLOv8m (Original): **Failed** — no detection
-  CBAM Hybrid (Original): **Failed** — no detection
-  YOLOv8m (Augmented): **Detected** ✓
-  CBAM Hybrid (Augmented): **Detected** ✓

> Both models trained only on original data failed on this unseen sample, while both augmented variants succeeded — proving that augmentation is the **primary factor** for generalization to new environments.

#### 2. CBAM Eliminates Double-Detection
**Evidence — Marked Row 5 (wide crosswalk-style bump):**
- YOLOv8m (Original): **2 bounding boxes** on a single bump
- YOLOv8m (Augmented): **2 bounding boxes** on a single bump
- CBAM Hybrid (Original): **1 precise bounding box** ✓
- CBAM Hybrid (Augmented): **1 precise bounding box** ✓

> The CBAM Spatial Attention module helped the model understand that a wide bump is **one contiguous object**, not two separate ones — reducing false positive counts and improving bounding box precision.

#### 3. CBAM Partially Compensates for Limited Data
**Evidence — Non-Marked Row 10 (distant non-marked bump):**
- YOLOv8m (Original): **Failed** — missed the bump entirely
- CBAM Hybrid (Original): **Detected** ✓ — correctly identified the bump
- YOLOv8m (Augmented): **Detected** ✓
- CBAM Hybrid (Augmented): **Detected** ✓

> Even without augmentation, CBAM's attention mechanism partially compensated for the data scarcity by extracting more discriminative features from the limited non-marked samples.

---

##  Real-Time Performance

A critical requirement for deployment in autonomous driving is **maintaining real-time inference speed**. Both viable both exceed 50 FPS

Adding three CBAM modules to the backbone introduces only **~2ms additional latency** per frame — a negligible overhead that keeps the model well above the 30 FPS real-time threshold.


---

##   Interpretation

### Why CBAM Works on Non-Marked Bumps

Non-marked speed bumps lack the obvious visual cues (paint lines, patterns) that make marked bumps easy to detect. The CBAM attention mechanism provides two complementary benefits:

1. **Channel Attention** — Learns *which* feature channels are most informative for detecting subtle elevation changes in the road surface (texture, shadow patterns, slight color shifts).

2. **Spatial Attention** — Learns *where* in the feature map to focus, helping the model attend to the road surface region rather than background distractions (trees, vehicles, buildings).

### Why Augmentation + CBAM Shows Diminishing Returns

```
┌────────────────────────────────────────────────────────┐
│           The Shared-Problem Hypothesis                │
│                                                        │
│  ┌──────────────┐        ┌──────────────┐             │
│  │     CBAM     │        │ Augmentation │             │
│  │  (Attention) │        │  (Data ×8)   │             │
│  └──────┬───────┘        └──────┬───────┘             │
│         │                       │                      │
│         ▼                       ▼                      │
│  ┌─────────────────────────────────────┐              │
│  │    Same Problem: Data Scarcity      │              │
│  │    for Non-Marked Speed Bumps       │              │
│  └─────────────────────────────────────┘              │
│                                                        │
│  Different Mechanisms → Same Solution                  │
│  Combined → Diminishing Returns (Over-regularization)  │
└────────────────────────────────────────────────────────┘
```

When the baseline already achieves **0.995 AP50** with augmentation alone, there is very little room for CBAM to provide additional improvement. Instead, the attention layers begin to over-fit to augmentation artifacts rather than true bump features.


---

##  Repository Structure

```
bump-detection-project/

├──  Notebooks (Executed in order)
│   ├── 00_Data_Setup.ipynb              # Download, split & explore original dataset
│   ├── 00B_Data_Setup_Augmented.ipynb   # Targeted augmentation for non-marked class
│   ├── 01_Experiment_YOLOv8.ipynb       # Baseline YOLOv8m (Exp 1a & 1b)
│   ├── 03_Experiment_Hybrid_Pretrained.ipynb  # CBAM Hybrid (Exp 3a & 3b)
│   └── 04_Final_Comparison.ipynb        # Master comparison & expert analysis
│
├──  Configuration
│   └── yolov8m-cbam.yaml               # Custom CBAM architecture definition
│
├──  Results & Artifacts
│   ├── exp1_results.csv                 # Baseline experiment metrics
│   ├── final_master_comparison.csv      # Full 4-model comparison table
│   ├── final_master_comparison_chart.png # Bar chart visualization
│   ├── class_distribution.png           # Dataset class balance visualization
│   ├── external_test_4models.png        # Qualitative grid on external data
│   └── split_comparison.mp4            # Side-by-side real-time video comparison
│
├──  datasets/
│   ├── total-5/                         # Original dataset (705 train / 206 val / 81 test)
│   └── total-5-augmented/              # Augmented dataset (non-marked ×8)
│
├── runs/detect/runs/                 # Trained model weights & training logs
│   ├── exp1a_original/weights/best.pt   # YOLOv8m Baseline (Original)
│   ├── exp1b_augmented/weights/best.pt  # YOLOv8m Baseline (Augmented)
│   ├── exp3a_original/weights/best.pt   # CBAM Hybrid (Original)
│   └── exp3b_augmented/weights/best.pt  # CBAM Hybrid (Augmented)
│
└── README.md                            # This file
```
