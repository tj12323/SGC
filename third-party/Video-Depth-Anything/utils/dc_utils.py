# This file is originally from DepthCrafter/depthcrafter/utils.py at main · Tencent/DepthCrafter
# SPDX-License-Identifier: MIT License license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file is released under [ MIT License license], with the full license text available at [https://github.com/Tencent/DepthCrafter?tab=License-1-ov-file].
import numpy as np
import matplotlib.cm as cm
import imageio
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except:
    import cv2
    DECORD_AVAILABLE = False

import os
import glob
import cv2
from typing import Optional, Union
import imageio.v2 as iio
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

def ensure_even(value):
    return value if value % 2 == 0 else value + 1

def read_video_frames(video_path, process_length, target_fps=-1, max_res=-1):
    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = original_height
        width = original_width
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = ensure_even(round(original_height * scale))
            width = ensure_even(round(original_width * scale))

        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

        fps = vid.get_avg_fps() if target_fps == -1 else target_fps
        stride = round(vid.get_avg_fps() / fps)
        stride = max(stride, 1)
        frames_idx = list(range(0, len(vid), stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        frames = vid.get_batch(frames_idx).asnumpy()
    else:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if max_res > 0 and max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)

        fps = original_fps if target_fps < 0 else target_fps

        stride = max(round(original_fps / fps), 1)

        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (process_length > 0 and frame_count >= process_length):
                break
            if frame_count % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                if max_res > 0 and max(original_height, original_width) > max_res:
                    frame = cv2.resize(frame, (width, height))  # Resize frame
                frames.append(frame)
            frame_count += 1
        cap.release()
        frames = np.stack(frames, axis=0)

    return frames, fps


def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])

    writer.close()


def read_image_frames(image_folder_path, process_length, original_fps, target_fps=-1, max_res=-1):
    if not os.path.isdir(image_folder_path):
        raise ValueError(f"Image folder not found: {image_folder_path}")

    supported_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder_path, ext)))


    image_files = sorted(image_files)
    print(f"Found {len(image_files)} image files.")

    num_total_images = len(image_files)
    stride = 1
    calculated_fps = original_fps # Default fps

    if target_fps > 0:
        if original_fps <= 0:
            raise ValueError("original_fps must be positive (> 0) when target_fps is specified.")
        stride = max(round(original_fps / target_fps), 1)
        calculated_fps = target_fps
    elif original_fps <= 0:
         print("Warning: original_fps not provided or invalid (<=0). Cannot calculate stride for target_fps. "
               "Proceeding with stride=1 and returning fps=1.")
         calculated_fps = 1 # Cannot determine a meaningful fps

    selected_indices = list(range(0, num_total_images, stride))
    selected_image_files = [image_files[i] for i in selected_indices]
    # print(f"Selected {len(selected_image_files)} frames based on stride {stride}.")

    if process_length > 0 and len(selected_image_files) > process_length:
        selected_image_files = selected_image_files[:process_length]

    if not selected_image_files:
        print("Warning: No frames selected after applying stride and process_length.")
        return np.empty((0, 0, 0, 3), dtype=np.uint8), calculated_fps

    target_height, target_width = -1, -1
    needs_resizing = False

    first_image_path = selected_image_files[0]
    first_frame = imageio.v2.imread(first_image_path)

    if first_frame.ndim == 2: # Grayscale
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2RGB)
    elif first_frame.ndim == 3 and first_frame.shape[2] == 4: # RGBA
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGBA2RGB) # Use cv2 for reliable conversion
    elif first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(f"Unsupported image format/channels in {os.path.basename(first_image_path)}: shape {first_frame.shape}")

    original_height, original_width = first_frame.shape[:2]

    if max_res > 0 and max(original_height, original_width) > max_res:
        needs_resizing = True
        scale = max_res / max(original_height, original_width)
        target_height = ensure_even(round(original_height * scale))
        target_width = ensure_even(round(original_width * scale))
        # print(f"Resizing needed. Original: ({original_width}x{original_height}), Target: ({target_width}x{target_height})")
    else:
        needs_resizing = False
        target_height, target_width = original_height, original_width # No resizing needed, use original dims
        # print(f"No resizing needed. Using original dimensions: ({target_width}x{target_height})")

    frames_list = []
    skipped_count = 0
    for i, img_path in enumerate(selected_image_files):
        frame = imageio.v2.imread(img_path)

        # --- Ensure RGB format ---
        if frame.ndim == 2: # Grayscale -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4: # RGBA -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.ndim != 3 or frame.shape[2] != 3:
                print(f"Warning: Skipping unsupported image {os.path.basename(img_path)}: shape {frame.shape}")
                skipped_count += 1
                continue # Skip this frame

        current_height, current_width = frame.shape[:2]
        if needs_resizing or current_height != target_height or current_width != target_width:
            # Use INTER_AREA for downsampling, INTER_LINEAR for upsampling or general purpose
            interpolation = cv2.INTER_AREA if (current_width * current_height > target_width * target_height) else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)

        frames_list.append(frame)

    if skipped_count > 0:
        print(f"Skipped {skipped_count} problematic image(s).")

    if not frames_list:
        print("Warning: No frames were successfully loaded or processed.")
        # Return empty array with correct target shape if possible, otherwise (0,0,0,3)
        h, w = (target_height, target_width) if target_height > 0 and target_width > 0 else (0, 0)
        return np.empty((0, h, w, 3), dtype=np.uint8), calculated_fps

    # --- Stack frames into NumPy array ---
    frames_np = np.stack(frames_list, axis=0)
    # print(f"Successfully loaded and processed {frames_np.shape[0]} frames with shape {frames_np.shape}.")
    return frames_np, calculated_fps

def _save_single_depth_frame_worker(
    frame_index: int,
    frame_data: np.ndarray,
    output_folder_path: str,
    file_prefix: str,
    file_format: str,
    target_dtype: type, # np.uint16 in this case
    zfill_width: int
):
    # 1. Per-frame normalization (moved from main loop)
    current_min = np.min(frame_data)
    current_max = np.max(frame_data)
    output_max_value = 65535 # Since target_dtype is uint16

    # Avoid division by zero if this frame is flat
    if current_max - current_min < 1e-6:
            depth_norm_float = np.full_like(frame_data, 0.0, dtype=np.float32)
            current_max = current_min + 1.0 # Assign a range for consistency check below
    else:
        depth_norm_float = (frame_data - current_min) / (current_max - current_min)

    # Clamp to handle potential floating point inaccuracies
    depth_norm_float = np.clip(depth_norm_float, 0.0, 1.0)

    # Scale to target integer range [0, output_max_value]
    output_data = (depth_norm_float * output_max_value).astype(target_dtype)

    # 2. Construct filename
    filename = f"{file_prefix}{str(frame_index).zfill(zfill_width)}.{file_format.lower()}"
    output_path = os.path.join(output_folder_path, filename)

    # 3. Save the image
    iio.imwrite(output_path, output_data)
    return None # Indicate success

# --- Modified save_depth_images function ---
def save_depth_images_parallel(
    frames: np.ndarray,
    output_folder_path: str,
    file_prefix: str = "depth_",
    file_format: str = "png",
    grayscale: bool = False, # Still keep for API consistency
    colormap_name: str = "inferno", # Keep for API consistency
    normalize_globally: bool = True,
    output_dtype: Optional[type] = np.uint8,
    num_workers: Optional[int] = None # Number of parallel processes
):
    if frames.ndim != 3:
        raise ValueError(f"Input 'frames' must be a 3D array (N, H, W), but got shape {frames.shape}")

    if frames.shape[0] == 0:
        print("Warning: Input 'frames' array is empty. No images will be saved.")
        return

    num_frames, height, width = frames.shape

    os.makedirs(output_folder_path, exist_ok=True)

    # --- Pre-calculate common values ---
    zfill_width = 5
    target_dtype = np.uint8
    output_max_value = 255
    if output_dtype is np.uint16:
        target_dtype = np.uint16
        output_max_value = 65535
    elif output_dtype is not None and output_dtype is not np.uint8:
        print(f"Warning: Unsupported output_dtype {output_dtype}. Using np.uint8.")
        output_dtype = np.uint8

    use_parallel = (num_workers is None or num_workers > 0) and \
                   (not normalize_globally or not grayscale) and \
                   num_frames > 1 # No benefit for 1 frame

    if use_parallel:
        workers = num_workers if num_workers is not None else os.cpu_count()
        print(f"Saving {num_frames} frames in parallel using {workers} workers...")

        # Prepare arguments for the worker function
        # Use partial to fix the constant arguments for the worker
        worker_func = partial(
            _save_single_depth_frame_worker, # Need to define this function outside
            output_folder_path=output_folder_path,
            file_prefix=file_prefix,
            file_format=file_format,
            target_dtype=target_dtype,
            zfill_width=zfill_width
        )

        # Create arguments list: [(index, frame_data), (index, frame_data), ...]
        tasks = [(i, frames[i]) for i in range(num_frames)]

        # Create a pool and map tasks
        with mp.Pool(processes=workers) as pool:
            # Use tqdm to show progress with the parallel map
            results = list(pool.starmap(worker_func, tasks))

        # Check for errors returned by workers
        errors = [r for r in results if r is not None]
        if errors:
            print(f"Warning: Encountered {len(errors)} errors during parallel saving.")
            # Optionally print first few errors
            for i, e in enumerate(errors[:5]):
                print(f"  Error {i+1}: {e}")

    else:
        # --- Sequential execution (Original logic simplified) ---
        print(f"Saving {num_frames} frames sequentially...")

        # --- Global Normalization / Colormap Setup (if needed) ---
        d_min, d_max = None, None
        colormap = None
        if normalize_globally:
             d_min = np.min(frames)
             d_max = np.max(frames)
             if d_max - d_min < 1e-6:
                 print("Warning: Global depth range is near zero.")
                 d_max = d_min + 1.0
             print(f"Global normalization range: min={d_min:.4f}, max={d_max:.4f}")

        if not grayscale:
             try:
                 colormap = cm.get_cmap(colormap_name)
             except ValueError: # Handle colormap loading errors
                 raise ValueError(f"Invalid colormap name: {colormap_name}")


        for i in tqdm(range(num_frames)):
            depth = frames[i]
            current_min, current_max = d_min, d_max
            if not normalize_globally: # Per-frame norm if needed
                current_min = np.min(depth)
                current_max = np.max(depth)
                if current_max - current_min < 1e-6:
                     current_max = current_min + 1.0

            # Normalize to [0, 1] float
            if current_max - current_min < 1e-6:
                 depth_norm_float = np.full_like(depth, 0.0, dtype=np.float32)
            else:
                depth_norm_float = (depth - current_min) / (current_max - current_min)
            depth_norm_float = np.clip(depth_norm_float, 0.0, 1.0)

            # Prepare output data
            output_data = None
            if grayscale:
                output_data = (depth_norm_float * output_max_value).astype(target_dtype)
            else:
                # Colormapping logic here (as in original function)
                colored = colormap(depth_norm_float)
                output_data = (colored[:, :, :3] * 255).astype(np.uint8) # Always uint8 for color

            # Save
            filename = f"{file_prefix}{str(i).zfill(zfill_width)}.{file_format.lower()}"
            output_path = os.path.join(output_folder_path, filename)
            iio.imwrite(output_path, output_data)

    print("Finished saving depth images.")