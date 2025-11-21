# Installation Guide

This guide provides step-by-step instructions to set up the SGC project and its dependencies.

**Tested Environment:**

- **OS:** Ubuntu 22.04
- **Python:** 3.12 (for SGC main environment), 3.10 (for DELTA environment)
- **PyTorch:** 2.4.0 (with CUDA 11.8)
- **GPU:** NVIDIA RTX 3090 (or similar with sufficient VRAM)

While other setups might work, the code has been primarily tested with these specifications. We recommend using `conda` for managing Python environments.

## 1. Clone the Repository

First, clone the SGC repository and navigate into the directory. If the third-party libraries are included as Git submodules, initialize them:

```bash
cd SGC
git submodule update --init --recursive
```

## 2. Main SGC Environment Setup

This environment will be used for running the core SGC metric.

```bash
conda create -n sgc python=3.12.4 -y
conda activate sgc

# Install PyTorch (ensure this matches your CUDA version)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# Install XFormers
pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu118
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu11==25.4.*" "dask-cudf-cu11==25.4.*" "cuml-cu11==25.4.*" \
    "cugraph-cu11==25.4.*" "nx-cugraph-cu11==25.4.*" "cuspatial-cu11==25.4.*" \
    "cuproj-cu11==25.4.*" "cuxfilter-cu11==25.4.*" "cucim-cu11==25.4.*" \
    "pylibraft-cu11==25.4.*" "raft-dask-cu11==25.4.*" "cuvs-cu11==25.4.*" \
    "nx-cugraph-cu11==25.4.*"
```

## 3. Third-Party Dependencies

Our project relies on several third-party libraries for various functionalities (e.g., segmentation, depth estimation, tracking). These are typically located in the `third-party/` directory.

---

### 3.1. SegAnyMo Setup

[SegAnyMo](https://github.com/nnanhuang/SegAnyMo) is used for motion segmentation.

**A. Install SegAnyMo Dependencies:**

```bash
cd third-party/SegAnyMo
pip install -r requirements.txt
```

**B. Install SAM2 (Sub-component of SegAnyMo):**

```bash
cd sam2
pip install -e .

# Download SAM2 checkpoints
mkdir -p checkpoints
cd checkpoints
bash ./download_ckpts.sh
cd ../.. # Back to third-party/SegAnyMo
```

**C. Download Main SegAnyMo Model Checkpoints:**

These checkpoints are for the primary SegAnyMo model.

```bash
mkdir -p checkpoints
cd checkpoints
```

Choose one method to download:

- You can download from [huggingface](https://huggingface.co/Changearthmore/moseg),
- Or you can download the model checkpoints from [google drive](https://drive.google.com/file/d/15VWtEqsROKAxdZbzaXrrmCm4k1D8SJJR/view?usp=drive_link).

- **Configuration:** After downloading, you may need to update the checkpoint path in `configs/example_train.yaml` (relative to the `SegAnyMo` directory). Specifically, set the `resume_path` variable.

```bash
cd ../.. # Back to SGC root directory
```

---

### 3.2. DELTA Setup

[DELTA](https://github.com/snap-research/DELTA_densetrack3d) is used for dense 2D tracking.
**Note:** DELTA requires its own Conda environment.

**A. Create DELTA Environment and Install Dependencies:**

```bash
cd third-party/DenseTrack3D # Ensure you are in SGC/third-party/DenseTrack3D

conda create -n densetrack3d python=3.10 cmake=3.14.0 -y
conda activate densetrack3d

# Downgrade pip for specific dependencies
pip install pip==24.0
pip install -r requirements.txt # Installs PyTorch, etc. as per DELTA's requirements
conda install ffmpeg -c conda-forge -y
pip install -U "ray[default]"
```

**B. Install Unidepth (Sub-component of DELTA):**

```bash
pip install ninja
# Unidepth requires a specific xformers version, different from the main 'sgc' env.
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# Alternative for PyTorch3D if issues:
# cd submodules/UniDepth/unidepth/ops/knn/src && bash compile.sh && cd ../../../../../../
```

**C. Download DELTA Checkpoints:**

```bash
mkdir -p ./checkpoints/
# Download 2D checkpoint for DELTA (example)
gdown --fuzzy https://drive.google.com/file/d/1S_T7DzqBXMtr0voRC_XUGn1VTnPk_7Rm/view?usp=sharing -O ./checkpoints/
```

- **Important:** When you need to run DELTA's processing scripts, make sure to activate its environment: `conda activate densetrack3d`. For SGC, use `conda activate sgc`.

```bash
cd ../.. # Back to SGC root directory
conda activate sgc # Switch back to the main SGC environment
```

---

### 3.3. Video-Depth-Anything Setup

[Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) is used for video depth estimation.

**A. Install Dependencies:**

```bash
cd third-party/Video-Depth-Anything # Ensure you are in SGC/third-party/Video-Depth-Anything
pip install -r requirements.txt
```

**B. Download Pre-trained Models:**

We use the following models from Video-Depth-Anything:

| Model                         | Params |                                                       Checkpoint Download Link                                                        |
| :---------------------------- | -----: | :-----------------------------------------------------------------------------------------------------------------------------------: |
| Video-Depth-Anything-V2-Small |  28.4M | [Download](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true) |
| Video-Depth-Anything-V2-Large | 381.8M | [Download](https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth?download=true) |

You can download these manually and place them in a `checkpoints` directory inside `third-party/Video-Depth-Anything/`, or use the provided script if it handles these specific models:

```bash
mkdir -p checkpoints
# If get_weights.sh downloads the required models:
bash get_weights.sh
```

```bash
cd ../.. # Back to SGC root directory
```

---

### 3.4. VGGT Setup

[VGGT](https://github.com/facebookresearch/vggt) is used for camera parameter estimation.

```bash
cd third-party
git clone https://github.com/facebookresearch/vggt
cd vggt
pip install -e .
```

---

After completing all the above steps, your SGC project and all specified third-party dependencies should be ready for use. Remember to activate the correct Conda environment (e.g., sgc for the main project and most dependencies, or densetrack3d specifically for DELTA processing) depending on which part of the project you are working with.
