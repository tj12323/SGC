# Measuring 3D Spatial Geometric Consistency in Dynamically Generated Videos

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/<YOUR_LINK>)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black.svg)](https://github.com/tj12323/SGC)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Weijia Dou](https://tj12323.github.io/)<sup>1*</sup>, [Wenzhao Zheng](https://wzzheng.net/)<sup>2,3*,†</sup>, [Weiliang Chen](https://chen-wl20.github.io/)<sup>2</sup>, [Yu Zheng](https://yzheng97.github.io/)<sup>2</sup>, [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)<sup>2</sup>, [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)<sup>2</sup>

<small>(*Equal contribution; †Project leader.)</small>

<sup>1</sup>Tongji University &nbsp;&nbsp; <sup>2</sup>Tsinghua University &nbsp;&nbsp; <sup>3</sup>University of California, Berkeley

</div>

---
**Table of Contents**

* [🧠 Overview](#-overview)
  * [Key Features](#key-features)
* [🛠️ Installation](#️-installation)
* [🚀 Usage](#-usage)
  * [Data Preparation](#data-preparation)
  * [Running the Evaluation](#running-the-evaluation)
* [📊 Evaluation Results](#-evaluation-results)
* [🙏 Acknowledgments](#-acknowledgments)
---

## 🧠 Overview

Recent generative models can produce high-fidelity videos, yet they often exhibit 3D spatial geometric inconsistencies. These failures include geometric warping, incoherent motion, object impermanence, and perspective failures. Existing evaluation methods fail to accurately characterize these inconsistencies: fidelity-centric metrics like FVD are insensitive to geometric distortions, while consistency-focused benchmarks often penalize valid foreground dynamics.

To address this gap, we introduce **SGC**, a metric for evaluating 3D Spatial Geometric Consistency in dynamically generated videos. We quantify geometric consistency by measuring the divergence among multiple camera poses estimated from distinct local regions.

<p align="center">
  <img src="assets/overview.png" alt="Overview of SGC Pipeline" width="800"/>
</p>

### Key Features
1. **Foreground-Background Disentanglement**: Our approach first segments dynamic objects using motion object segmentation (MOS) to isolate the static background. Crucially, all subsequent SGC metrics and methods are applied only *after* this MOS step. As our evaluations show, seemingly improved scores without MOS can be misleading; metrics calculated with MOS more accurately reflect true background geometric inconsistencies.
2. **Depth-Aware Partitioning**: After isolating the static areas, we predict depth for each pixel and partition the remaining static background into spatially coherent sub-regions.
3. **Composite Variance Scoring**: We estimate a local camera pose for each subregion and compute the divergence among these poses. The overall SGC score is computed by aggregating three key evaluations: local inter-segment consistency, global pose consistency, and cross-frame depth consistency error.


---

## 🛠️ Installation

For detailed installation instructions, please refer to the **[Installation Guide](docs/Install.md)**.

Please see `docs/Install.md` for a comprehensive guide on setting up the environment, dependencies, and any required submodules.

---

## 🚀 Usage

### Data Preparation

Our SGC metric can process video files directly or pre-extracted frames. Organize your custom data in a directory structure like this:

```text
your_dataset/
├── video1.mp4                 # Option A: Direct video files
├── experiment_A_video/        # Option B: Pre-extracted frames
│   ├── 00000.jpg 
│   ├── 00001.jpg
│   └── ...
└── ...

```

### Running the Evaluation

Because our metrics are calculated strictly on the static background, you must perform motion object segmentation (MOS) *before* running the SGC calculation.

**Step 1: Extract motion masks to isolate the static background**

```bash
bash scripts/run_seganymo.sh
```

**Step 2: Compute the SGC score on the segmented sub-areas**

```bash
bash scripts/run_sgc.sh
python sgc/calculatescore.py
```

The output will be saved as a JSON file containing the overall SGC score and the breakdown of the three component metrics.


---

## 📊 Evaluation Results


We curate a comprehensive benchmark of 1,296 videos, comprising 996 generated videos and 300 high-motion real videos. Experiments on real and generative videos demonstrate that SGC robustly quantifies geometric inconsistencies, effectively identifying critical failures missed by existing metrics.


| Method              | SGC Score (↓) |
| ------------------- | ------------- |
| Cosmos              | 0.0722        |
| Hotshot             | 0.1172        |
| Latte               | 0.3226        |
| Lavie               | 0.1241        |
| Modelscope          | 0.3129        |
| opensora-i          | 0.1631        |
| opensora-t          | 0.0831        |
| Seine               | 0.2837        |
| Videocrafter        | 0.0973        |
| Zeroscope           | 0.0912        |
| **RT-1 (Real)**     | **0.0639**    |
| **Nuscenes (Real)** | **0.0613**    |
| **OpenVid (Real)**  | **0.0530**    |

(For full quantitative comparisons across all 10 state-of-the-art models, please refer to Table 1 in our paper ). 

---

## 🙏 Acknowledgments

This implementation is made possible by several excellent open-source foundational estimators. We sincerely thank the authors of:

* [VGGT](https://github.com/facebookresearch/vggt)
* [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
* [SegAnyMo](https://github.com/nnanhuang/SegAnyMo)
* [DELTA](https://github.com/snap-research/DELTA_densetrack3d)
