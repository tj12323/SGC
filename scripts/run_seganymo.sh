#!/usr/bin/env bash

# Base directory containing video folders
BASE_DIR="data/mygenvideo/k400-val"

# GPUs to use
GPUS=(0 1 2 3)

# Names of directories to skip
SKIP_DIRS=("real_video")

# Loop over each subdirectory in the base directory
for video_dir in "$BASE_DIR"/*/; do
  # Remove trailing slash and extract base name
  video_path="${video_dir%/}"
  dir_name="$(basename "$video_path")"

  # Skip specified directories
  if [[  " ${SKIP_DIRS[@]} " =~ " ${dir_name} " ]]; then
    echo "Skipping directory: $dir_name"
    continue
  fi

  echo "========================================"
  echo "Processing video directory: $video_path"
  echo "========================================"

  # Run inference
  python third-party/SegAnyMo/core/utils/run_inference.py \
    --video_path "$video_path" \
    --gpus ${GPUS[@]} \
    --depths \
    --tracks \
    --dinos \
    --e \
    --track_model delta \
    --depth_model video-depth-anything \
    --step 10
  # Prepare output dirs
  mkdir -p "./third-party/SegAnyMo/result/moseg_delta/$dir_name"
  mkdir -p "./third-party/SegAnyMo/result/sam2/$dir_name"

  python third-party/SegAnyMo/core/utils/run_inference.py \
    --video_path "$video_path" \
    --motin_seg_dir "./third-party/SegAnyMo/result/moseg_delta/$dir_name" \
    --gpus ${GPUS[@]} \
    --motion_seg_infer \
    --e \
    --config_file ./third-party/SegAnyMo/configs/example_train.yaml \
    --track_model delta \
    --depth_model video-depth-anything \
    --step 10
  
  python third-party/SegAnyMo/core/utils/run_inference.py \
    --video_path "$video_path" \
    --sam2dir "./third-party/SegAnyMo/result/sam2/$dir_name" \
    --motin_seg_dir "./third-party/SegAnyMo/result/moseg_delta/$dir_name" \
    --gpus ${GPUS[@]} \
    --sam2 \
    --e \
    --track_model delta \
    --depth_model video-depth-anything \
    --step 10
  echo
done