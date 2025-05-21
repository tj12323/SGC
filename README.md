# Measuring 4D Spatial-Temporal Consistency in Generated Videos

---

<p align="center">
  <img src="assets/overview.png" alt="Overview of STC Pipeline" width="700"/>
</p>
<p align="center">
  <em>Overview of the STC computation pipeline. Input RGB frames undergo parallel processing: (i) depth estimation, leading to dense point reconstruction for global camera pose estimation; and (ii) pixel tracking followed by motion segmentation to isolate moving objects. The identified static background is then adaptively segmented. Local camera poses for these static sub-regions are subsequently estimated using information from pixel tracks and depth. Finally, the overall STC score is computed by aggregating four key evaluations: local inter-segment consistency, global pose consistency (comparing local estimates against the global camera motion), reprojection error, and cross-frame depth consistency.</em>
</p>

---

## 📝 Contents

- [Measuring 4D Spatial-Temporal Consistency in Generated Videos](#measuring-4d-spatial-temporal-consistency-in-generated-videos)
  - [📝 Contents](#-contents)
  - [📄 Abstract](#-abstract)
  - [🛠️ Installation](#️-installation)
  - [🚀 Usage](#-usage)
    - [Data Preparation](#data-preparation)
    - [Running STC Metric (TODO: If applicable)](#running-stc-metric-todo-if-applicable)
  - [📊 Evaluation](#-evaluation)
    - [Datasets (TODO: Mention datasets used)](#datasets-todo-mention-datasets-used)
    - [Results (TODO: Link to paper or show key results/comparisons)](#results-todo-link-to-paper-or-show-key-resultscomparisons)

---

## 📄 Abstract

We introduce STC, a metric for evaluating 4D **S**patio-**T**emporal **C**onsistency in generated videos. With the rapid development of video generation models, the visual quality of synthetic videos has improved reached an extent of being almost visually indistinguishable from real videos. Despite performing well on conventional video quality assessment metrics (e.g., FVD), most existing methods still cannot generate videos with perfect spatial-temporal consistency. However, it lacks a quantitative definition and computationally feasible metric to measure the level of spatial-temporal consistency in videos. To address this, we propose to measure spatial-temporal consistency by the variance of camera poses computed by different regions across frames. Specifically, we first establish correspondence between pixels in all the frames and disentangle dynamic and static regions. We then predict depth for each pixel to obtain 3D dense points for each frame. With the point-wise correspondence and dynamics segmentation, we divide the points into subgroups and estimate a sequence of poses for each subgroup. Finally, we compute various metrics to measure the pose variance between different regions and combine them to define the overall STC metric. We compare different metric on both generated and real videos. Experimental results demonstrate that the proposed STC can well differentiate them, showing the practical application of STC for evaluating video quality.

---

## 🛠️ Installation

For detailed installation instructions, please refer to the **[Installation Guide](docs/Install.md)**.

Please see `docs/Install.md` for a comprehensive guide on setting up the environment, dependencies, and any required submodules.

---

## 🚀 Usage

### Data Preparation

Our STC metric can process video files directly or pre-extracted frames. Please organize your data as described below.

**1. For Your Custom Videos/Frames:**

You have two options for providing your own video data:

- **Option A: Video Files**
  If you are using video files directly, place them in a directory:

  ```
  your_dataset/
  ├── video1.mp4
  ├── experiment_A_video.avi
  ├── another_sample.mov
  └── ...
  ```

- **Option B: Pre-extracted Frames**
  If you have already extracted frames from your videos, organize them into sub-directories, where each sub-directory corresponds to a single video:
  ```
  your_dataset_frames/
  ├── video1/               # Corresponds to 'video1.mp4' or a video named 'video1'
  │   ├── 00000.jpg         # Or .png, etc.
  │   ├── 00001.jpg
  │   └── ...
  ├── experiment_A_video/
  │   ├── frame_000.png
  │   ├── frame_001.png
  │   └── ...
  └── ...
  ```

**2. Structure for Datasets Used in Our Experiments:**

The following shows an example of how datasets were organized for the experiments reported in our paper (e.g., K400-val, NuScenes). If you intend to reproduce our results or use any provided evaluation scripts for these specific benchmarks, your data might need to follow a similar structure:

```

evaluation\_datasets/
├── k400-val/
│   ├── cosmos/             \# Sub-category, model outputs, or specific split
│   │   ├── images/
│   │   │   ├── cosmos\_0001/ \# This could be a folder of frames or a single video file
│   │   │   │   ├── 00000.jpg
│   │   │   │   └── ...
│   │   │   ├── cosmos\_0002/
│   │   │   └── ...
│   ├── hotshot/
│   │   ├── images/
|   │   │   ├── hotshot\_0001/
|   │   │   └── ...
│   ├── latte/
│   │   └── ...
│   └── ...                 \# Other categories or models evaluated on k400-val
├── nuscenes/
└── ...                     \# Other benchmark datasets

```

### Running STC Metric (TODO: If applicable)

```bash
# TODO: Provide command-line examples or a Python script snippet to run your STC metric.
# Explain the arguments and expected outputs.

# Example:
# python calculate_stc.py --video_path /path/to/your/video.mp4 --output_dir /path/to/results/
# or
# python calculate_stc_batch.py --video_dir /path/to/your/videos/ --output_file results.csv

# TODO: Specify what the output STC score represents (e.g., higher is better/worse).
```

**Example Python Usage (if it's a library):**

```python
# TODO: If your metric can be used as a library, provide a simple code example.
# from stc_metric import STCalculator
#
# calculator = STCalculator(config_path='path/to/config.yaml') # Or model_weights_path
# video_path = "path/to/video.mp4"
# stc_score = calculator.compute(video_path)
# print(f"STC Score for {video_path}: {stc_score}")
```

---

## 📊 Evaluation

### Datasets (TODO: Mention datasets used)

We evaluated STC on a variety of generated and real videos.

- **Generated Videos:** `[TODO: List or describe sources of generated videos, e.g., specific models evaluated]`
- **Real Videos:** `[TODO: List or describe sources of real videos, e.g., specific datasets like DAVIS, Kinetics, etc.]`

### Results (TODO: Link to paper or show key results/comparisons)

Experimental results demonstrate that the proposed STC can well differentiate generated videos with varying levels of spatio-temporal consistency from real videos.

- `[TODO: Briefly summarize key findings or link to tables/figures in your paper.]`
- `[TODO: Optionally, provide pre-computed STC scores for some benchmark models/datasets if available.]`

```

```
