# Combine elements from both scripts
import argparse
import glob
import os
import pickle # Needed if loading DenseTrack ckpt requires it, but likely not for state_dict

import imageio.v2 as imageio
import mediapy as media
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import KDTree # For mapping sparse points to dense grid
import pdb

# Imports from the first script (TAPIR one) - keep necessary ones
# from tapnet_torch import transforms # Keep for coordinate conversion

# Imports from the second script (DenseTrack2D one)
from densetrack3d.datasets.custom_data import read_data # Or use the simple read_video below
from densetrack3d.models.densetrack3d.densetrack2d import DenseTrack2D
from densetrack3d.models.predictor.dense_predictor2d import DensePredictor2D
# from densetrack3d.utils.visualizer import Visualizer # Not needed for NPY generation


def read_video_frames(folder_path):
    """Reads video frames into a NumPy array."""
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {folder_path}")
    video = np.stack([imageio.imread(frame_path) for frame_path in frame_paths])
    # Ensure 3 channels (handle RGBA)
    if video.shape[-1] == 4:
        video = video[..., :3]
    print(f"Read video: {video.shape=} {video.dtype=} {video.min()=} {video.max()=}")
    # Don't convert to media._VideoArray here, keep as numpy
    return video

def preprocess_frames_for_dense(frames_np, device):
    """Preprocess frames for DenseTrack2D input.
    Args:
      frames_np: [num_frames, height, width, 3], [0, 255], np.uint8
    Returns:
      frames_tensor: [1, num_frames, 3, height, width], [0, 255], torch.float32
    """
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(device)
    frames_tensor = frames_tensor.float() # Keep in [0, 255] range, just convert type
    frames_tensor = frames_tensor.unsqueeze(0) # Add batch dimension
    return frames_tensor

from typing import Sequence

import chex
import numpy as np
def convert_grid_coordinates(
    coords: chex.Array,
    input_grid_size: Sequence[int],
    output_grid_size: Sequence[int],
    coordinate_format: str = 'xy',
) -> chex.Array:
  """Convert image coordinates between image grids of different sizes.

  By default, it assumes that the image corners are aligned.  Therefore,
  it adds .5 (since (0,0) is assumed to be the center of the upper-left grid
  cell), multiplies by the size ratio, and then subtracts .5.

  Args:
    coords: The coordinates to be converted.  It is of shape [..., 2] if
      coordinate_format is 'xy' or [..., 3] if coordinate_format is 'tyx'.
    input_grid_size: The size of the image/grid that the coordinates currently
      are with respect to.  This is a 2-tuple of the format [width, height]
      if coordinate_format is 'xy' or a 3-tuple of the format
      [num_frames, height, width] if coordinate_format is 'tyx'.
    output_grid_size: The size of the target image/grid that you want the
      coordinates to be with respect to.  This is a 2-tuple of the format
      [width, height] if coordinate_format is 'xy' or a 3-tuple of the format
      [num_frames, height, width] if coordinate_format is 'tyx'.
    coordinate_format: Which format the coordinates are in.  This can be one
      of 'xy' (the default) or 'tyx', which are the only formats used in this
      project.

  Returns:
    The transformed coordinates, of the same shape as coordinates.

  Raises:
    ValueError: if coordinates don't match the given format.
  """
  if isinstance(input_grid_size, tuple):
    input_grid_size = np.array(input_grid_size)
  if isinstance(output_grid_size, tuple):
    output_grid_size = np.array(output_grid_size)

  if coordinate_format == 'xy':
    if input_grid_size.shape[0] != 2 or output_grid_size.shape[0] != 2:
      raise ValueError(
          'If coordinate_format is xy, the shapes must be length 2.')
  elif coordinate_format == 'tyx':
    if input_grid_size.shape[0] != 3 or output_grid_size.shape[0] != 3:
      raise ValueError(
          'If coordinate_format is tyx, the shapes must be length 3.')
    if input_grid_size[0] != output_grid_size[0]:
      raise ValueError('converting frame count is not supported.')
  else:
    raise ValueError('Recognized coordinate formats are xy and tyx.')

  position_in_grid = coords
  position_in_grid = position_in_grid * output_grid_size / input_grid_size

  return position_in_grid


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Input directory with image frames")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .npy files")
    parser.add_argument("--ckpt", type=str, default="checkpoints/densetrack2d.pth", help="DenseTrack2D checkpoint path")
    # Keep grid_size or add num_points? Let's keep grid_size for consistency
    parser.add_argument("--grid_size", type=int, default=None, help="Grid spacing for *sparse* query points on original frame")
    # DenseTrack2D expects fixed input size or handles resizing internally?
    # The demo implies internal handling but model init takes resolution.
    # Let's keep resizing for consistency with the original script's approach.
    parser.add_argument("--resize_height", type=int, default=192, help="Resize height for model input") # Adjusted default based on DenseTrack2D model_resolution
    parser.add_argument("--resize_width", type=int, default=256, help="Resize width for model input") # Adjusted default
    parser.add_argument("--step", type=int, default=10, help="Spacing between query frames (t)")
    # DenseTrack2D specific params (matching the demo script)
    parser.add_argument("--upsample_factor", type=int, default=4, help="DenseTrack2D model stride/upsample factor")
    parser.add_argument("--window_len", type=int, default=16, help="DenseTrack2D model window length")
    parser.add_argument("--num_virtual_tracks", type=int, default=64, help="DenseTrack2D virtual tracks")
    parser.add_argument("--use_fp16", action="store_true", help="whether to use fp16/bf16 precision")


    args = parser.parse_args()

    # folder_path = os.path.join(args.image_dir,'color')
    folder_path = args.image_dir
    frame_names = [
        os.path.basename(f) for f in sorted(glob.glob(os.path.join(folder_path, "*.[pj]*[np]*[g]*")))  
    ]
    if not frame_names:
        print(f"Error: No image files found in {folder_path}")
        return
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Check if already done ---
    done = True
    q_ts_check = list(range(0, len(frame_names), args.step))
    if not q_ts_check: # Handle case where step > num_frames
        print("No query frames based on step size, exiting.")
        return

    print(f"Checking completion for query frames: {q_ts_check}")
    for t in q_ts_check:
        name_t = os.path.splitext(frame_names[t])[0]
        # Check if *all* target files for this query frame exist
        all_targets_exist = True
        for j in range(len(frame_names)):
            name_j = os.path.splitext(frame_names[j])[0]
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            if not os.path.exists(out_path):
                # print(f"Missing: {out_path}") # Debug print
                all_targets_exist = False
                break # No need to check further for this t
        if not all_targets_exist:
            done = False
            break # No need to check further query frames

    for t in range(0, len(frame_names)):
        name_t = os.path.splitext(frame_names[t])[0]
        # Check if *all* target files for this query frame exist
        all_targets_exist = True
        j = t + 1
        if j < len(frame_names):
            name_j = os.path.splitext(frame_names[j])[0]
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            if not os.path.exists(out_path):
                # print(f"Missing: {out_path}") # Debug print
                all_targets_exist = False
        if not all_targets_exist:
            done = False
            break # No need to check further query frames

    print(f"Overall completion check: {done=}")
    if done:
        print("NPY files already generated for all specified query frames.")
        return
    # --- End Check ---

    # --- Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Create DenseTrack2D model")
    # Use resize dimensions for model_resolution
    # Note: DenseTrack2D stride=4, so resolution should be divisible by 4?
    # The demo used (384//2, 512//2) = (192, 256) which matches our defaults.
    model = DenseTrack2D(
        stride=4, # Often 4 for this model
        window_len=args.window_len,
        add_space_attn=True, # Common setting
        num_virtual_tracks=args.num_virtual_tracks,
        model_resolution=(args.resize_height, args.resize_width),
        upsample_factor=args.upsample_factor
    )

    print(f"Load checkpoint from {args.ckpt}")
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint file not found at {args.ckpt}")
        return
    try:
        with open(args.ckpt, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            # Allow loading even if some keys don't match (e.g., different heads)
            model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    predictor = DensePredictor2D(model=model)
    predictor = predictor.eval().to(device)
    print("Model loaded successfully.")

    # --- Video Loading and Preprocessing ---
    resize_height = args.resize_height
    resize_width = args.resize_width

    video_np_original = read_video_frames(folder_path)
    num_frames, height, width = video_np_original.shape[0:3]

    # Resize video
    print(f"Resizing video to ({resize_height}, {resize_width})...")
    # Important: Use 'area' interpolation for downsampling if possible
    video_np_resized = media.resize_video(video_np_original, (resize_height, resize_width))
    print(f"Resized video shape: {video_np_resized.shape}")

    # Preprocess for DenseTrack2D
    video_tensor = preprocess_frames_for_dense(video_np_resized, device)
    print(f"Preprocessed video tensor shape: {video_tensor.shape}") # Should be [1, T, 3, H_resized, W_resized]

    # --- Define Sparse Query Grid (on original dimensions) ---
    grid_size = args.grid_size
    if grid_size is None:
        max_grid_points = 9000
        grid_size = max(1, int(np.sqrt((height * width) / max_grid_points)))
    y_orig, x_orig = np.mgrid[0:height:grid_size, 0:width:grid_size]
    # Flatten and stack: points_orig shape [N_sparse, 2] (x, y order)
    points_orig = np.stack([x_orig.ravel(), y_orig.ravel()], axis=-1).astype(np.float32)
    num_sparse_points = points_orig.shape[0]
    print(f"Defined {num_sparse_points} sparse query points with grid size {grid_size}.")

    # Scale sparse grid points to resized dimensions for mapping later
    scale_x = (resize_width - 1) / (width - 1) if width > 1 else 1.0
    scale_y = (resize_height - 1) / (height - 1) if height > 1 else 1.0
    points_resized = np.zeros_like(points_orig)
    points_resized[:, 0] = points_orig[:, 0] * scale_x
    points_resized[:, 1] = points_orig[:, 1] * scale_y


    # --- Tracking Loop ---
    q_ts = list(range(0, num_frames, args.step))
    print(f"Processing query frames: {q_ts}")

    for t in tqdm(q_ts, desc="Query Frames"):
        name_t = os.path.splitext(frame_names[t])[0]

        # Check if all outputs for this 't' already exist (redundant check, but safe)
        all_targets_exist_for_t = True
        for j_check in range(num_frames):
             name_j_check = os.path.splitext(frame_names[j_check])[0]
             out_path_check = f"{out_dir}/{name_t}_{name_j_check}.npy"
             if not os.path.exists(out_path_check):
                 all_targets_exist_for_t = False
                 break
        if all_targets_exist_for_t:
             print(f"Skipping query frame t={t} ({name_t}), all outputs exist.")
             continue

        print(f"\nProcessing query frame t={t} ({name_t})")

        # --- Run DenseTrack2D for this query frame ---
        print(f"  Running DenseTrack2D predictor with grid_query_frame={t}...")
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=args.use_fp16):
                # Ensure predictor takes B T C H W
                out_dict = predictor(video_tensor, grid_query_frame=t, backward_tracking=True)

        # Extract dense results (on GPU initially)
        # Shapes: trajs_uv [1, T, N_dense, 2], vis [1, T, N_dense], conf [1, T, N_dense]
        trajs_uv_dense = out_dict["trajs_uv"] # Coords in resized space
        vis_dense = out_dict["vis"]
        conf_dense = out_dict["conf"]
        # dense_reso = out_dict["dense_reso"] # H_dense, W_dense - might be useful for checking N_dense
        num_dense_points = trajs_uv_dense.shape[2]
        print(f"  Dense predictor output: {num_dense_points=}, {trajs_uv_dense.shape=}")


        if num_dense_points == 0:
            print(f"  Warning: Dense predictor returned 0 points for t={t}. Skipping NPY generation for this t.")
            # Create empty files to mark as done? Or just skip? Skipping for now.
            continue # Skip to next query frame t

        # --- Map Sparse Points to Dense Output ---
        k_neighbors = 5
        print(f"  Mapping sparse points to {k_neighbors} nearest dense outputs & selecting by confidence...")

        # Get the coordinates of the dense grid points AT THE QUERY FRAME t
        # These should be the starting points of the dense tracks in resized coords
        # Move to CPU for KDTree
        dense_coords_at_t = trajs_uv_dense[0, t, :, :].detach().cpu().numpy() # Shape [N_dense, 2]

        # Get confidence scores at query frame t, also move to CPU
        # conf_dense shape: [1, T, N_dense] -> detach/cpu -> [T, N_dense] index t -> [N_dense]
        conf_dense_t_cpu = conf_dense[0, t, :].detach().cpu().numpy() # Shape [N_dense]

        # Use KDTree for efficient nearest neighbor lookup
        try:
            tree = KDTree(dense_coords_at_t)
            # Find the indices of the k nearest neighbors in the dense output
            # Note: If N_dense < k, it will return all N_dense neighbors.
            distances, multi_indices = tree.query(points_resized, k=min(k_neighbors, num_dense_points), workers=-1) # Use all cores
            # multi_indices shape: [N_sparse, k] (or [N_sparse] if k=1, or [N_sparse, N_dense] if k > N_dense)

            # Ensure multi_indices is 2D even if k=1
            if multi_indices.ndim == 1:
                multi_indices = multi_indices[:, np.newaxis]

            # Refine selection based on confidence at frame t
            sparse_to_dense_indices = np.zeros(num_sparse_points, dtype=np.int64)
            for i in range(num_sparse_points):
                neighbor_indices = multi_indices[i] # Indices of the k nearest dense points for sparse point i

                # Ensure indices are valid (should be guaranteed by KDTree within N_dense range)
                # Get confidences for these neighbors
                neighbor_confidences = conf_dense_t_cpu[neighbor_indices]

                # Find which of the k neighbors has the highest confidence
                best_neighbor_local_idx = np.argmax(neighbor_confidences) # Index within the 'neighbor_indices' array (0 to k-1)

                # Get the actual index in the full dense array
                sparse_to_dense_indices[i] = neighbor_indices[best_neighbor_local_idx]

            print(f"  Mapped {num_sparse_points} sparse points to highest-confidence neighbor (among k={k_neighbors}).")

        except Exception as e:
             print(f"  Error during KDTree mapping/confidence selection for t={t}: {e}. Skipping NPY generation for this t.")
             traceback.print_exc()
             continue # Skip to next query frame t

        # =========== MODIFIED MAPPING SECTION END ===========

        # --- Select Sparse Tracks from Dense Output ---
        # Gather the full trajectories for the selected indices
        # Move relevant tensors to CPU *after* indexing on GPU (usually faster)
        selected_indices_tensor = torch.from_numpy(sparse_to_dense_indices).long().to(device)

        # Index shapes: [1, T, N_dense, 2/1] -> [1, T, N_sparse, 2/1]
        trajs_sparse_resized = torch.index_select(trajs_uv_dense, 2, selected_indices_tensor)[0] # Remove batch dim -> [T, N_sparse, 2]
        vis_sparse = torch.index_select(vis_dense, 2, selected_indices_tensor)[0]             # -> [T, N_sparse]
        conf_sparse = torch.index_select(conf_dense, 2, selected_indices_tensor)[0]           # -> [T, N_sparse]

        # Move results to CPU and convert to numpy
        trajs_sparse_resized_np = trajs_sparse_resized.detach().cpu().numpy()
        vis_sparse_np = vis_sparse.detach().cpu().numpy()
        conf_sparse_np = conf_sparse.detach().cpu().numpy()
        print(f"  Selected sparse tracks shapes: {trajs_sparse_resized_np.shape=}, {vis_sparse_np.shape=}, {conf_sparse_np.shape=}")

        # --- Post-process Tracks ---
        print("  Post-processing tracks...")
        # 1. Convert coordinates from resized space back to original video space
        # trajs_sparse_original_np = convert_dense_to_original_coords(
        #     trajs_sparse_resized_np,
        #     (resize_height, resize_width),
        #     (height, width)
        # )
        trajs_sparse_original_np = convert_grid_coordinates(
                    trajs_sparse_resized_np, (resize_width - 1, resize_height - 1), (width - 1, height - 1)
        )

        # 2. Combine into the final [N_sparse, T, 4] structure
        # Permute T, N_sparse -> N_sparse, T
        trajs_sparse_original_np = np.transpose(trajs_sparse_original_np, (1, 0, 2)) # [N_sparse, T, 2]
        vis_sparse_np = np.transpose(vis_sparse_np, (1, 0))                         # [N_sparse, T]
        conf_sparse_np = np.transpose(conf_sparse_np, (1, 0))                       # [N_sparse, T]

        outputs = np.concatenate(
            [
                trajs_sparse_original_np, # x, y (original coords)
                vis_sparse_np[..., None],    # visibility/occlusion
                conf_sparse_np[..., None],   # confidence/quality
            ],
            axis=-1,
        ).astype(np.float32) # Shape [N_sparse, T, 4]
        print(f"  Combined output shape: {outputs.shape}")


        # 3. Overwrite coordinates at query frame 't' with the exact grid points
        # Ensure points_orig [N_sparse, 2] matches the order of outputs [N_sparse, T, 4]
        # points_orig has (x, y) order, matching outputs[:,:,0:2]
        if t < num_frames: # Ensure t is a valid index
            outputs[:, t, 0:2] = points_orig
            print(f"  Overwrote coordinates at query frame t={t} with original grid points.")
        else:
            print(f"  Warning: Query frame index t={t} is out of bounds ({num_frames=}). Cannot overwrite coordinates.")


        # --- Save NPY Files ---
        print(f"  Saving NPY files for query frame t={t}...")
        for j in range(num_frames):
            name_j = os.path.splitext(frame_names[j])[0]
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            # Extract data for frame j: shape [N_sparse, 4]
            frame_data = outputs[:, j, :]
            np.save(out_path, frame_data)

        print(f"  Finished saving NPY files for query frame t={t}.")

    for t in tqdm(range(num_frames - 1), desc="Query Frames"):
        # pdb.set_trace()
        name_t = os.path.splitext(frame_names[t])[0]

        # Check if all outputs for this 't' already exist (redundant check, but safe)
        all_targets_exist_for_t = True
        for j_check in range(2):
            name_j_check = os.path.splitext(frame_names[j_check+t])[0]
            out_path_check = f"{out_dir}/{name_t}_{name_j_check}.npy"
            if not os.path.exists(out_path_check):
                all_targets_exist_for_t = False
        if all_targets_exist_for_t:
             print(f"Skipping query frame t={t} ({name_t}), all outputs exist.")
             continue

        print(f"\nProcessing query frame t={t} ({name_t})")

        # --- Run DenseTrack2D for this query frame ---
        print(f"  Running DenseTrack2D predictor with grid_query_frame={t}...")
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=args.use_fp16):
                # Ensure predictor takes B T C H W
                video_pair = video_tensor[:, t : t + 2, ...]
                out_dict = predictor(video_pair, grid_query_frame=0, backward_tracking=True)

        # Extract dense results (on GPU initially)
        # Shapes: trajs_uv [1, T, N_dense, 2], vis [1, T, N_dense], conf [1, T, N_dense]
        trajs_uv_dense = out_dict["trajs_uv"] # Coords in resized space
        vis_dense = out_dict["vis"]
        conf_dense = out_dict["conf"]
        # dense_reso = out_dict["dense_reso"] # H_dense, W_dense - might be useful for checking N_dense
        num_dense_points = trajs_uv_dense.shape[2]
        print(f"  Dense predictor output: {num_dense_points=}, {trajs_uv_dense.shape=}")


        if num_dense_points == 0:
            print(f"  Warning: Dense predictor returned 0 points for t={t}. Skipping NPY generation for this t.")
            # Create empty files to mark as done? Or just skip? Skipping for now.
            continue # Skip to next query frame t

        # --- Map Sparse Points to Dense Output ---
        k_neighbors = 5
        print(f"  Mapping sparse points to {k_neighbors} nearest dense outputs & selecting by confidence...")

        # Get the coordinates of the dense grid points AT THE QUERY FRAME t
        # These should be the starting points of the dense tracks in resized coords
        # Move to CPU for KDTree
        dense_coords_at_t = trajs_uv_dense[0, 0, :, :].detach().cpu().numpy() # Shape [N_dense, 2]

        # Get confidence scores at query frame t, also move to CPU
        # conf_dense shape: [1, T, N_dense] -> detach/cpu -> [T, N_dense] index t -> [N_dense]
        conf_dense_t_cpu = conf_dense[0, 0, :].detach().cpu().numpy() # Shape [N_dense]

        # Use KDTree for efficient nearest neighbor lookup
        try:
            tree = KDTree(dense_coords_at_t)
            # Find the indices of the k nearest neighbors in the dense output
            # Note: If N_dense < k, it will return all N_dense neighbors.
            distances, multi_indices = tree.query(points_resized, k=min(k_neighbors, num_dense_points), workers=-1) # Use all cores
            # multi_indices shape: [N_sparse, k] (or [N_sparse] if k=1, or [N_sparse, N_dense] if k > N_dense)

            # Ensure multi_indices is 2D even if k=1
            if multi_indices.ndim == 1:
                multi_indices = multi_indices[:, np.newaxis]

            # Refine selection based on confidence at frame t
            sparse_to_dense_indices = np.zeros(num_sparse_points, dtype=np.int64)
            for i in range(num_sparse_points):
                neighbor_indices = multi_indices[i] # Indices of the k nearest dense points for sparse point i

                # Ensure indices are valid (should be guaranteed by KDTree within N_dense range)
                # Get confidences for these neighbors
                neighbor_confidences = conf_dense_t_cpu[neighbor_indices]

                # Find which of the k neighbors has the highest confidence
                best_neighbor_local_idx = np.argmax(neighbor_confidences) # Index within the 'neighbor_indices' array (0 to k-1)

                # Get the actual index in the full dense array
                sparse_to_dense_indices[i] = neighbor_indices[best_neighbor_local_idx]

            print(f"  Mapped {num_sparse_points} sparse points to highest-confidence neighbor (among k={k_neighbors}).")

        except Exception as e:
             print(f"  Error during KDTree mapping/confidence selection for t={t}: {e}. Skipping NPY generation for this t.")
             traceback.print_exc()
             continue # Skip to next query frame t

        # =========== MODIFIED MAPPING SECTION END ===========

        # --- Select Sparse Tracks from Dense Output ---
        # Gather the full trajectories for the selected indices
        # Move relevant tensors to CPU *after* indexing on GPU (usually faster)
        selected_indices_tensor = torch.from_numpy(sparse_to_dense_indices).long().to(device)

        # Index shapes: [1, T, N_dense, 2/1] -> [1, T, N_sparse, 2/1]
        trajs_sparse_resized = torch.index_select(trajs_uv_dense, 2, selected_indices_tensor)[0] # Remove batch dim -> [T, N_sparse, 2]
        vis_sparse = torch.index_select(vis_dense, 2, selected_indices_tensor)[0]             # -> [T, N_sparse]
        conf_sparse = torch.index_select(conf_dense, 2, selected_indices_tensor)[0]           # -> [T, N_sparse]

        # Move results to CPU and convert to numpy
        trajs_sparse_resized_np = trajs_sparse_resized.detach().cpu().numpy()
        vis_sparse_np = vis_sparse.detach().cpu().numpy()
        conf_sparse_np = conf_sparse.detach().cpu().numpy()
        print(f"  Selected sparse tracks shapes: {trajs_sparse_resized_np.shape=}, {vis_sparse_np.shape=}, {conf_sparse_np.shape=}")

        # --- Post-process Tracks ---
        print("  Post-processing tracks...")
        # 1. Convert coordinates from resized space back to original video space
        # trajs_sparse_original_np = convert_dense_to_original_coords(
        #     trajs_sparse_resized_np,
        #     (resize_height, resize_width),
        #     (height, width)
        # )
        trajs_sparse_original_np = convert_grid_coordinates(
                    trajs_sparse_resized_np, (resize_width - 1, resize_height - 1), (width - 1, height - 1)
        )

        # 2. Combine into the final [N_sparse, T, 4] structure
        # Permute T, N_sparse -> N_sparse, T
        trajs_sparse_original_np = np.transpose(trajs_sparse_original_np, (1, 0, 2)) # [N_sparse, T, 2]
        vis_sparse_np = np.transpose(vis_sparse_np, (1, 0))                         # [N_sparse, T]
        conf_sparse_np = np.transpose(conf_sparse_np, (1, 0))                       # [N_sparse, T]

        outputs = np.concatenate(
            [
                trajs_sparse_original_np, # x, y (original coords)
                vis_sparse_np[..., None],    # visibility/occlusion
                conf_sparse_np[..., None],   # confidence/quality
            ],
            axis=-1,
        ).astype(np.float32) # Shape [N_sparse, T, 4]
        print(f"  Combined output shape: {outputs.shape}")


        # 3. Overwrite coordinates at query frame 't' with the exact grid points
        # Ensure points_orig [N_sparse, 2] matches the order of outputs [N_sparse, T, 4]
        # points_orig has (x, y) order, matching outputs[:,:,0:2]
        if t < num_frames: # Ensure t is a valid index
            outputs[:, 0, 0:2] = points_orig
            print(f"  Overwrote coordinates at query frame t={t} with original grid points.")
        else:
            print(f"  Warning: Query frame index t={t} is out of bounds ({num_frames=}). Cannot overwrite coordinates.")


        # --- Save NPY Files ---
        print(f"  Saving NPY files for query frame t={t}...")
        for j in range(2):
            print(j)
            name_j = os.path.splitext(frame_names[t+j])[0]
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            # Extract data for frame j: shape [N_sparse, 4]
            frame_data = outputs[:, j, :]
            np.save(out_path, frame_data)

        print(f"  Finished saving NPY files for query frame t={t}.")

    print("\nProcessing finished.")


if __name__ == "__main__":
    main()