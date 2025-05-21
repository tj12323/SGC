import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.segmentation import slic
from skimage.util import img_as_float
import warnings
import logging
import os
import argparse
import glob
import json
from functools import partial
import pdb
import datetime
import math # Added for isnan
import re

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from skimage import color # Added for Lab conversion
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configure logging (same as before) ---
log_dir = "logs_global"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filepath = os.path.join(log_dir, f"{timestamp}.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info(f"Logging initialized. Log file: {log_filepath}")


# --- Helper Functions (Existing ones mostly unchanged, ADDED calculate_variance_vs_reference_pose) ---

def load_frame_data(frame_path, depth_input, mos_path, semantic_path=None, depth_scale=1.0):
    # (Implementation unchanged)
    frame = cv2.imread(frame_path)

    if isinstance(depth_input, np.ndarray):
        depth_raw = depth_input
    elif isinstance(depth_input, str) and depth_input.lower().endswith('.npz'):
        npz = np.load(depth_input, allow_pickle=False)
        if 'depth' in npz:
            depth_raw = npz['depth']
        else:
            depth_raw = npz[npz.files[0]]
    else:
        depth_raw = cv2.imread(depth_input, cv2.IMREAD_UNCHANGED)

    if mos_path is not None:
        mos_mask = cv2.imread(mos_path, cv2.IMREAD_GRAYSCALE)
    else:
        mos_mask = None

    semantic_mask = None
    if semantic_path and os.path.exists(semantic_path):
        semantic_mask = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
    elif semantic_path:
         logging.warning(f"Semantic path specified but not found: {semantic_path}")

    if frame is None: raise IOError(f"Failed to load frame: {frame_path}")
    if depth_raw is None: raise IOError(f"Failed to load depth map")

    depth = depth_raw.astype(np.float32)
    if depth_scale > 0 and depth_scale != 1.0:
         depth /= depth_scale
         # logging.debug(f"Applied depth scale: {depth_scale}")
    elif depth_scale <= 0:
         # logging.info("Using raw integer depth values (depth_scale <= 0).")
         pass # Keep depth as is

    min_d, max_d = (np.min(depth[depth > 0]), np.max(depth)) if np.any(depth > 0) else (0, 0)
    # logging.debug(f"Frame {os.path.basename(frame_path)}: Using depth range [{min_d:.2f}, {max_d:.2f}] (after applying scale={depth_scale})")

    return frame, depth, mos_mask, semantic_mask

def apply_masks(frame, depth, mos_mask, semantic_mask=None, static_semantic_labels=None):
    # (Implementation unchanged)
    if mos_mask is None:
        logging.warning("MOS mask is None, treating entire frame as static.")
        static_mask = np.ones(frame.shape[:2], dtype=bool)
    else:
        static_mask = (mos_mask == 0)

    if semantic_mask is not None and static_semantic_labels:
        if not hasattr(static_semantic_labels, '__iter__'):
            logging.error("static_semantic_labels must be iterable. Treating as empty.")
            static_semantic_labels = []
        try:
            semantic_static_mask = np.isin(semantic_mask, static_semantic_labels)
            static_mask = static_mask & semantic_static_mask
        except Exception as e:
            logging.error(f"Error applying semantic mask filter: {e}")

    static_frame = cv2.bitwise_and(frame, frame, mask=static_mask.astype(np.uint8))
    static_depth = depth.copy()
    static_depth[~static_mask] = 0 # Use 0 for invalid depth in masked areas
    return static_frame, static_depth, static_mask


def get_3d_points(pixels_2d, depth_map, K): # Added K argument
    # (Implementation unchanged)
    # Input validation for K
    if K is None or not isinstance(K, np.ndarray) or K.shape != (3, 3):
        logging.error("get_3d_points received invalid K.")
        return np.empty((0, 3)), np.empty((0, 2)), np.array([], dtype=bool)

    if pixels_2d.shape[0] == 0: return np.empty((0, 3)), np.empty((0, 2)), np.array([], dtype=bool)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = pixels_2d[:, 0]
    v = pixels_2d[:, 1]
    h, w = depth_map.shape
    u_idx, v_idx = np.clip(u.astype(int), 0, w - 1), np.clip(v.astype(int), 0, h - 1)
    Z = depth_map[v_idx, u_idx]
    valid_depth_mask = Z > 1e-5 # Use a small threshold
    if not np.any(valid_depth_mask): return np.empty((0, 3)), np.empty((0, 2)), np.array([], dtype=bool)
    u_valid, v_valid, Z_valid = u[valid_depth_mask], v[valid_depth_mask], Z[valid_depth_mask]
    X_valid = (u_valid - cx) * Z_valid / fx
    Y_valid = (v_valid - cy) * Z_valid / fy
    points_3d = np.vstack((X_valid, Y_valid, Z_valid)).T
    pixels_2d_valid = pixels_2d[valid_depth_mask]
    return points_3d, pixels_2d_valid, valid_depth_mask


def orthogonalize_rotation_matrix(R_in):
    # (Implementation unchanged)
    if R_in is None or not isinstance(R_in, np.ndarray) or R_in.shape != (3, 3):
        logging.warning("Invalid input provided for orthogonalization.")
        return None
    try:
        # Perform SVD
        U, _, Vt = np.linalg.svd(R_in)

        # Calculate the orthogonal matrix UV^T
        R_ortho = U @ Vt

        # Ensure the determinant is +1 (a proper rotation)
        if np.linalg.det(R_ortho) < 0:
            Vt[-1, :] *= -1
            R_ortho = U @ Vt

        final_det = np.linalg.det(R_ortho)
        if abs(final_det - 1.0) > 1e-6:
             logging.warning(f"Orthogonalization resulted in determinant {final_det:.7f}. Check SVD results.")

        return R_ortho

    except np.linalg.LinAlgError:
        logging.warning("SVD did not converge during matrix orthogonalization. Returning original matrix.")
        return R_in
    except Exception as e:
        logging.error(f"Unexpected error during orthogonalization: {e}", exc_info=True)
        return None

def angular_distance(R1, R2):
    # (Implementation unchanged)
    if R1 is None or R2 is None or not isinstance(R1, np.ndarray) or not isinstance(R2, np.ndarray) or R1.shape != (3,3) or R2.shape != (3,3): return np.inf

    tolerance = 1e-2
    det1_ok = abs(np.linalg.det(R1) - 1.0) <= tolerance
    det2_ok = abs(np.linalg.det(R2) - 1.0) <= tolerance

    if not det1_ok or not det2_ok:
        # Be less verbose if it's a known issue during variance calculation
        # logging.debug(f"Attempted angular distance with non-rotation matrix (det1_ok={det1_ok}, det2_ok={det2_ok}).")
        return np.inf
    try:
        R_diff = R1.T @ R2
        trace = np.trace(R_diff)
        # Clamp the value to avoid domain errors with arccos due to floating point inaccuracies
        angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        return np.degrees(angle_rad)
    except ValueError as e:
        logging.error(f"Error in angular_distance: {e}")
        return np.inf


def normalize_translation(t):
    # (Implementation unchanged)
    if t is None or not isinstance(t, np.ndarray): return None
    t_flat = t.flatten()
    norm = np.linalg.norm(t_flat)
    if norm < 1e-9: return np.zeros_like(t_flat) # Return zero vector if norm is too small
    return t_flat / norm

# --- Segmentation function (using Depth-Aware SLIC) ---
# def segment_static_background(static_frame, static_depth, static_mask,
#                               n_segments=100, compactness=10.0,
#                               depth_weight=5.0, min_size_filter=100,
#                               sigma=1.0):
#     # (Implementation unchanged - using the provided Depth-Aware SLIC version)
#     h, w = static_frame.shape[:2]
#     sub_area_segments = np.zeros((h, w), dtype=np.int32)
#     num_segments = 0

#     if not np.any(static_mask):
#         logging.warning("Static mask is empty, no segments generated.")
#         return sub_area_segments, num_segments

#     logging.info(f"Segmenting using Depth-Aware SLIC: n_segments={n_segments}, "
#                  f"compactness={compactness}, depth_weight={depth_weight}, min_size={min_size_filter}")

#     try:
#         if static_frame.dtype != np.uint8:
#              if np.max(static_frame) <= 1.0:
#                  frame_uint8 = (static_frame * 255).astype(np.uint8)
#              else:
#                  frame_uint8 = static_frame.astype(np.uint8)
#         else:
#              frame_uint8 = static_frame

#         lab_image = color.rgb2lab(cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB))

#         depth_normalized = np.zeros_like(static_depth, dtype=np.float32)
#         valid_depth_mask = static_mask & (static_depth > 1e-5)

#         if np.any(valid_depth_mask):
#             valid_depths = static_depth[valid_depth_mask]
#             min_d = np.min(valid_depths)
#             max_d = np.max(valid_depths)
#             depth_range = max_d - min_d

#             if depth_range > 1e-9:
#                 depth_normalized[valid_depth_mask] = (static_depth[valid_depth_mask] - min_d) / depth_range
#             else:
#                 depth_normalized[valid_depth_mask] = 0.5
#             logging.debug(f"Normalized depth range: [{min_d:.2f}, {max_d:.2f}] -> [0, 1]")
#         else:
#             logging.warning("No valid depth values found within the static mask. Depth channel will be zero.")

#         weighted_depth = depth_normalized * depth_weight
#         weighted_depth_ch = weighted_depth[..., np.newaxis]
#         feature_image = np.concatenate((img_as_float(lab_image), weighted_depth_ch), axis=2)
#         logging.debug(f"Feature image shape for SLIC: {feature_image.shape}")

#         slic_labels = slic(
#             feature_image,
#             n_segments=n_segments,
#             compactness=compactness,
#             sigma=sigma,
#             mask=static_mask.astype(bool),
#             start_label=1,
#             enforce_connectivity=True,
#             channel_axis=-1
#         )

#         unique_labels = np.unique(slic_labels)
#         current_id = 1
#         for label in unique_labels:
#             if label == 0: continue
#             segment_mask = (slic_labels == label)
#             pixel_count_in_mask = np.sum(segment_mask & static_mask)
#             if pixel_count_in_mask >= min_size_filter:
#                 sub_area_segments[segment_mask] = current_id
#                 current_id += 1
#         num_segments = current_id - 1
#         if num_segments == 0:
#             logging.warning("Depth-Aware SLIC generated 0 valid segments after size filtering.")

#     except ImportError:
#          logging.error("Scikit-image library not found. Please install it (`pip install scikit-image`)")
#          return np.zeros((h, w), dtype=np.int32), 0
#     except Exception as e:
#         logging.error(f"Depth-Aware SLIC segmentation failed: {e}", exc_info=True)
#         return np.zeros((h, w), dtype=np.int32), 0

#     logging.info(f"Generated {num_segments} depth-aware sub-area segments.")
#     return sub_area_segments, num_segments

# Segmentation function (segment_static_background) remains the same
def segment_static_background(static_frame, static_depth, static_mask, method='slic', **kwargs):
    """Segments the static background into sub-areas."""
    h, w = static_frame.shape[:2]
    sub_area_segments = np.zeros((h, w), dtype=np.int32)
    num_segments = 0
    min_size_filter = kwargs.get('min_size_filter', 100)

    if not np.any(static_mask):
        logging.warning("Static mask is empty, no segments generated.")
        return sub_area_segments, num_segments

    # —— 原有 semantic 分支 —— #
    if method == 'semantic' and 'semantic_mask' in kwargs and kwargs['semantic_mask'] is not None and 'static_labels' in kwargs:
        # ... (semantic branch unchanged) ...
        logging.info("Segmenting using semantic labels.")
        semantic_mask = kwargs['semantic_mask']
        static_labels = kwargs['static_labels']
        current_id = 1
        if not hasattr(static_labels, '__iter__'):
            logging.error("static_labels must be iterable.")
            static_labels = []

        for label in static_labels:
            label_mask = (semantic_mask == label) & static_mask
            if not np.any(label_mask): continue
            num_labels_cc, labels_im_cc = cv2.connectedComponents(label_mask.astype(np.uint8)) # Renamed to avoid conflict
            for i_cc in range(1, num_labels_cc): # Renamed to avoid conflict
                component_mask = (labels_im_cc == i_cc)
                if np.sum(component_mask) >= min_size_filter:
                    sub_area_segments[component_mask] = current_id
                    current_id += 1
        num_segments = current_id - 1
        if num_segments == 0:
            logging.warning("Semantic segmentation yielded no valid segments.")

    # —— 原有 depth 分支 (MODIFIED) —— #
    elif method == 'depth':
        logging.info("Segmenting using GPU-accelerated depth clustering (cuML).")
        n_clusters = kwargs.get('n_depth_clusters', 10)
        valid_depth_pixels = static_depth[static_mask]
        pixel_coords = np.argwhere(static_mask) # y, x coordinates

        if valid_depth_pixels.shape[0] < n_clusters * 2:
            logging.warning(f"Not enough valid depth pixels ({valid_depth_pixels.shape[0]}) for {n_clusters} clusters. Falling back to SLIC.")
            method = 'slic' # Fallback to SLIC
        else:
            try:
                import cupy as cp
                from cuml.cluster import KMeans as cuKMeans # Or MiniBatchKMeans if strictly needed and available

                # 1. Move data to GPU
                gpu_valid_depth_pixels = cp.asarray(valid_depth_pixels.reshape(-1, 1), dtype=cp.float32)

                # Heuristic for n_init with cuML KMeans, usually okay with default or a modest number
                # cuML's KMeans is often very fast.
                kmeans_model_gpu = cuKMeans(
                    n_clusters=n_clusters,
                    random_state=0, # For reproducibility
                    n_init=10,      # Default for scikit-learn, adjust as needed for cuML
                    # max_iter=300 # Default for scikit-learn
                )
                logging.info(f"Using cuML KMeans with n_clusters={n_clusters}")

                # 2. Fit and predict on GPU
                gpu_labels = kmeans_model_gpu.fit_predict(gpu_valid_depth_pixels)

                # 3. Move labels back to CPU (NumPy)
                labels = cp.asnumpy(gpu_labels)

                temp_segments = np.zeros((h, w), dtype=np.int32)
                # Assign labels back to their original 2D pixel locations
                temp_segments[pixel_coords[:, 0], pixel_coords[:, 1]] = labels + 1 # labels are 0-indexed

                current_id = 1
                unique_labels = np.unique(temp_segments)
                for unique_label_val in unique_labels: # Renamed to avoid conflict
                    if unique_label_val == 0: continue # Skip background/unlabeled
                    segment_mask = (temp_segments == unique_label_val)
                    if np.sum(segment_mask) >= min_size_filter:
                        sub_area_segments[segment_mask] = current_id
                        current_id += 1
                num_segments = current_id - 1
            except ImportError:
                logging.error("cuML or CuPy not found. Cannot use GPU for K-Means. Falling back to SLIC or scikit-learn MiniBatchKMeans.")
                # Optionally, implement a fallback to scikit-learn's MiniBatchKMeans here
                # For now, falling back to SLIC as per original logic for other failures
                method = 'slic'
                num_segments = 0
            except Exception as e:
                logging.error(f"GPU Depth clustering with cuML KMeans failed: {e}. Falling back to SLIC.", exc_info=True)
                method = 'slic' # Fallback to SLIC on error
                num_segments = 0 # Ensure num_segments is reset for fallback logic

    # —— 新增 grid 网格分割分支 —— #
    elif method == 'grid':
        # ... (grid branch unchanged) ...
        logging.info("Segmenting using uniform grid.")
        grid_rows, grid_cols = kwargs.get('grid_size', (4, 4))
        cell_h = h // grid_rows
        cell_w = w // grid_cols
        current_id = 1

        for i_grid in range(grid_rows): # Renamed to avoid conflict
            y0 = i_grid * cell_h
            y1 = (i_grid + 1) * cell_h if i_grid < grid_rows - 1 else h
            for j_grid in range(grid_cols): # Renamed to avoid conflict
                x0 = j_grid * cell_w
                x1 = (j_grid + 1) * cell_w if j_grid < grid_cols - 1 else w
                cell_mask = static_mask[y0:y1, x0:x1]
                if np.sum(cell_mask) >= min_size_filter:
                    sub_area_segments[y0:y1, x0:x1][cell_mask] = current_id
                    current_id += 1
        num_segments = current_id - 1
        logging.info(f"Generated {num_segments} grid-based segments.")
        return sub_area_segments, num_segments # Return early for grid

    # —— 原有 SLIC 分支 (and fallback) —— #
    # This block will be executed if method was 'slic' initially,
    # or if semantic/depth clustering failed or produced no segments.
    if method == 'slic' or num_segments == 0: # Check num_segments here for fallback
        if num_segments == 0 and method != 'slic': # Log if falling back
            logging.info(f"Previous segmentation method '{method}' yielded 0 segments. Falling back to SLIC segmentation.")
        else: # Original SLIC call
            logging.info("Segmenting using SLIC superpixels.")

        n_slic_segments = kwargs.get('n_slic_segments', 100)
        compactness = kwargs.get('compactness', 10)
        # sigma = kwargs.get('slic_sigma', 1.0) # This was in the commented out depth-aware SLIC
        # Ensure static_frame is suitable for img_as_float
        if static_frame.dtype == np.uint8:
            float_frame = img_as_float(static_frame)
        elif np.max(static_frame) > 1.0 and static_frame.dtype != np.float32 and static_frame.dtype != np.float64: # e.g. uint16
            logging.warning(f"static_frame dtype is {static_frame.dtype} with max > 1. Normalizing for SLIC.")
            # Basic normalization, might need adjustment based on actual range
            frame_max = np.max(static_frame)
            if frame_max > 0:
                 float_frame = static_frame.astype(np.float32) / frame_max
            else:
                 float_frame = static_frame.astype(np.float32)
        else: # Assumed to be float in [0,1] or int that img_as_float handles
            float_frame = img_as_float(static_frame)

        try:
            # Note: The original commented Depth-Aware SLIC used a feature_image
            # including LAB colors and normalized depth. This SLIC uses only RGB (or grayscale if frame is so).
            # Ensure static_mask is boolean for slic mask parameter
            slic_labels = slic(
                float_frame,
                n_segments=n_slic_segments,
                compactness=compactness,
                sigma=kwargs.get('slic_sigma', 0), # skimage slic uses sigma for pre-smoothing if > 0
                start_label=1, # So that 0 can be 'no segment'
                mask=static_mask.astype(bool) if static_mask is not None else None,
                enforce_connectivity=True,
                channel_axis=-1 if float_frame.ndim == 3 else None # For color images
            )
            unique_labels_slic = np.unique(slic_labels) # Renamed to avoid conflict
            current_id = 1
            # Reset sub_area_segments if we are falling back
            if num_segments == 0 and method != 'slic': # If we fell back, clear previous attempts
                sub_area_segments.fill(0)

            for label_slic in unique_labels_slic: # Renamed to avoid conflict
                if label_slic == 0: continue # SLIC might produce 0 for masked out areas or if start_label is 0
                segment_mask = (slic_labels == label_slic)
                # Ensure filtering by static_mask in case SLIC expanded beyond it slightly or mask was None
                if np.sum(segment_mask & static_mask) >= min_size_filter:
                    sub_area_segments[segment_mask] = current_id
                    current_id += 1
            num_segments = current_id - 1
        except ImportError:
            logging.error("Scikit-image library not found for SLIC. Please install it (`pip install scikit-image`)")
            num_segments = 0 # Ensure it's 0 if SLIC cannot run
        except Exception as e:
            logging.error(f"SLIC segmentation failed: {e}", exc_info=True)
            num_segments = 0 # Ensure it's 0 on error

    final_method_log = method if not (method != 'slic' and num_segments == 0) else 'slic (fallback)'
    logging.info(f"Generated {num_segments} sub-area segments using method '{final_method_log}'.")
    return sub_area_segments, num_segments



# --- Pose Estimation and Local Consistency (Unchanged) ---
def estimate_pose_per_subarea(frame_idx, sub_area_segments, num_segments,
                              tracks_for_pair, depth_prev, K_prev, K_curr, # Added K_prev, K_curr
                              distCoeffs=None, pnp_reprojection_error=8.0):
    # (Implementation unchanged)
    sub_area_poses = {}
    if tracks_for_pair is None or 'points_prev' not in tracks_for_pair or 'points_curr' not in tracks_for_pair:
        logging.warning(f"Frame {frame_idx}: Invalid or missing track data for PnP.")
        return sub_area_poses

    points_prev = tracks_for_pair['points_prev']
    points_curr = tracks_for_pair['points_curr']

    if points_prev is None or points_curr is None or points_prev.shape[0] == 0:
        logging.warning(f"Frame {frame_idx}: PnP tracking data is empty.")
        return sub_area_poses

    points_3d_prev, pixels_2d_prev_valid, valid_mask_3d = get_3d_points(points_prev, depth_prev, K_prev)
    if points_3d_prev.shape[0] == 0: # Check early if no 3D points generated
         logging.warning(f"Frame {frame_idx}: No valid 3D points generated from depth_prev for PnP.")
         return sub_area_poses

    points_2d_curr_matched = points_curr[valid_mask_3d]

    if points_3d_prev.shape[0] < 4:
        logging.warning(f"Frame {frame_idx}: Insufficient valid 3D points ({points_3d_prev.shape[0]}) after depth check for PnP.")
        return sub_area_poses

    pixels_2d_curr_idx = points_2d_curr_matched.astype(int)
    h, w = sub_area_segments.shape
    pixels_2d_curr_idx[:, 0] = np.clip(pixels_2d_curr_idx[:, 0], 0, w - 1)
    pixels_2d_curr_idx[:, 1] = np.clip(pixels_2d_curr_idx[:, 1], 0, h - 1)
    try:
        point_segment_ids = sub_area_segments[pixels_2d_curr_idx[:, 1], pixels_2d_curr_idx[:, 0]]
    except IndexError as e:
         logging.error(f"IndexError accessing sub_area_segments in PnP for frame {frame_idx}. Seg shape: {sub_area_segments.shape}, Max Index: ({np.max(pixels_2d_curr_idx[:, 1])}, {np.max(pixels_2d_curr_idx[:, 0])}). Error: {e}")
         return sub_area_poses

    successful_pnp_count = 0
    for seg_id in range(1, num_segments + 1):
        mask_seg = (point_segment_ids == seg_id)
        num_points_in_seg = np.sum(mask_seg)
        if num_points_in_seg < 4: continue

        obj_points_3d = points_3d_prev[mask_seg]
        img_points_2d = points_2d_curr_matched[mask_seg]

        try:
            obj_points_3d = obj_points_3d.astype(np.float32).reshape(-1, 1, 3)
            img_points_2d = img_points_2d.astype(np.float32).reshape(-1, 1, 2)

            # Using solvePnPRansac for robustness
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_points_3d, img_points_2d, K_curr, distCoeffs=distCoeffs,
                iterationsCount=100, reprojectionError=pnp_reprojection_error, confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE # Or SOLVEPNP_EPNP if faster/preferred
            )
            # Define minimum inliers based on points available in segment
            min_inliers = max(4, int(num_points_in_seg * 0.3)) # Require at least 30% inliers or 4 points

            if success and inliers is not None and len(inliers) >= min_inliers:
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                # Check determinant more strictly after Rodrigues
                if abs(np.linalg.det(rotation_matrix) - 1.0) < 1e-3:
                    sub_area_poses[seg_id] = (rotation_matrix, tvec.flatten())
                    successful_pnp_count += 1
                else:
                    logging.debug(f"Frame {frame_idx}, Seg {seg_id}: PnP RANSAC resulted in invalid rotation matrix (det={np.linalg.det(rotation_matrix):.3f}). Inliers: {len(inliers)}/{num_points_in_seg}")
            # else: # Log failure reasons if needed
            #     if not success: logging.debug(f"Frame {frame_idx}, Seg {seg_id}: PnP RANSAC failed.")
            #     elif inliers is None: logging.debug(f"Frame {frame_idx}, Seg {seg_id}: PnP RANSAC returned None inliers.")
            #     else: logging.debug(f"Frame {frame_idx}, Seg {seg_id}: PnP RANSAC insufficient inliers ({len(inliers)}/{min_inliers}).")

        except cv2.error as e:
             if "points" in str(e).lower() or "size" in str(e).lower():
                 logging.warning(f"Frame {frame_idx}, Seg {seg_id}: OpenCV PnP input error: {e}. Points: {num_points_in_seg}, K_curr: {K_curr is not None}")
             else:
                 logging.error(f"Frame {frame_idx}, Seg {seg_id}: OpenCV Error during PnP: {e}")
        except Exception as e:
             logging.error(f"Frame {frame_idx}, Seg {seg_id}: Unexpected Error during PnP: {e}", exc_info=True)

    # logging.info(f"Frame {frame_idx}: Estimated poses for {successful_pnp_count}/{num_segments} sub-areas.")
    return sub_area_poses

def trim_outliers_iqr(data, cov=97.5):
    """
    利用 IQR 法剔除异常值：
      - 计算 25% 和 75% 分位数 q1, q3
      - IQR = q3 - q1
      - 保留 [q1 - k*IQR, q3 + k*IQR] 区间内的数据
    """
    a = np.array(data)
    if a.size == 0:
        return a
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    q_upper = np.percentile(a, cov)
    # 则 k 应该满足 q3 + k*iqr = q_upper
    k = (q_upper - q3) / iqr
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return a[(a >= lower) & (a <= upper)]

def assess_local_consistency(sub_area_poses):
    if not sub_area_poses:
        return np.nan, np.nan, None, None

    # 过滤掉无效 pose
    valid = {k: v for k, v in sub_area_poses.items()
             if v is not None and v[0] is not None and v[1] is not None}
    if len(valid) < 2:
        return 0.0, 0.0, None, None

    rotations = [pose[0] for pose in valid.values()]
    translations = [pose[1] for pose in valid.values()]

    rot_variance = np.nan
    trans_variance = np.nan
    mean_rotation = None
    mean_translation_orig = None

    #—— 1）旋转方差 ——#
    try:
        # 转到旋转向量空间
        rvecs = [R.from_matrix(m).as_rotvec() * 180.0/np.pi for m in rotations]
        mean_rvec = np.mean(rvecs, axis=0)
        mean_rotation = R.from_rotvec(mean_rvec * np.pi/180.0).as_matrix()

        # 逐个计算 (angular distance)^2
        rot_d2 = [angular_distance(m, mean_rotation)**2 for m in rotations]
        # 只保留有限值
        rot_d2 = [d for d in rot_d2 if np.isfinite(d)]

        # 剔除 IQR 外的异常点
        trimmed = trim_outliers_iqr(rot_d2, cov=98.0)
        if trimmed.size > 0:
            rot_variance = float(trimmed.mean())
        else:
            rot_variance = float(np.mean(rot_d2))  # 若剔完了也回退到全量均值
        rot_variance = float(np.mean(rot_d2))

    except Exception as e:
        logging.error(f"Error calculating rotation variance: {e}")

    #—— 2）平移方差 ——#
    try:
        # norm_ts = [normalize_translation(t) for t in translations]
        norm_ts = [t for t in translations]
        norm_ts = [t for t in norm_ts if t is not None]

        if len(norm_ts) >= 2:
            mean_t = np.mean(norm_ts, axis=0)
            # mean_t = normalize_translation(mean_t)  # 再归一化

            d2 = [np.linalg.norm(t - mean_t)**2 for t in norm_ts]
            # 剔除异常
            # trimmed = trim_outliers_iqr(d2, k=2)
            # if trimmed.size > 0:
            #     trans_variance = float(trimmed.mean())
            # else:
            #     trans_variance = float(np.mean(d2))
            trans_variance = float(np.mean(d2))

        else:
            trans_variance = 0.0

        # 原始平移均值（不归一化）
        mean_translation_orig = np.mean(translations, axis=0)

    except Exception as e:
        logging.error(f"Error calculating translation variance: {e}")

    return rot_variance, trans_variance, mean_rotation, mean_translation_orig


def calculate_reprojection_error(points_3d_prev, points_2d_curr_matched, pose, K_curr, # Added K_curr
                                 sub_area_segments, seg_id, distCoeffs=None):
    # (Implementation unchanged)
    if pose is None or pose[0] is None or pose[1] is None: return np.nan
    R_sub, t_sub = pose
    if points_3d_prev is None or points_2d_curr_matched is None or points_3d_prev.shape[0] == 0:
        return np.nan

    pixels_2d_curr_idx = points_2d_curr_matched.astype(int)
    h, w = sub_area_segments.shape
    pixels_2d_curr_idx[:, 0] = np.clip(pixels_2d_curr_idx[:, 0], 0, w - 1)
    pixels_2d_curr_idx[:, 1] = np.clip(pixels_2d_curr_idx[:, 1], 0, h - 1)
    try: point_segment_ids = sub_area_segments[pixels_2d_curr_idx[:, 1], pixels_2d_curr_idx[:, 0]]
    except IndexError: return np.nan

    mask_seg = (point_segment_ids == seg_id)
    num_points = np.sum(mask_seg)
    if num_points == 0: return 0.0 # Return 0 error if no points in segment

    obj_points = points_3d_prev[mask_seg]
    img_points_actual = points_2d_curr_matched[mask_seg]

    try:
        rvec, _ = cv2.Rodrigues(R_sub)
        obj_points_cv = obj_points.reshape(-1, 1, 3).astype(np.float32)
        img_points_projected, _ = cv2.projectPoints(obj_points_cv, rvec, t_sub, K_curr, distCoeffs=distCoeffs)

        if img_points_projected is None: return np.nan
        img_points_projected = img_points_projected.squeeze(axis=1)
        # Handle case where only one point is projected
        if num_points == 1 and img_points_projected.ndim == 1:
            img_points_projected = img_points_projected.reshape(1, 2)
        if img_points_actual.shape != img_points_projected.shape:
            logging.warning(f"Shape mismatch in reprojection: actual {img_points_actual.shape}, projected {img_points_projected.shape}")
            return np.nan

        errors = np.linalg.norm(img_points_actual - img_points_projected, axis=1)
        mean_error = np.mean(errors)
        return mean_error if np.isfinite(mean_error) else np.nan
    except cv2.error as e:
        logging.error(f"OpenCV error during reprojection for seg_id {seg_id}: {e}")
        return np.nan
    except Exception as e:
        logging.error(f"Unexpected error during reprojection for seg_id {seg_id}: {e}", exc_info=True)
        return np.nan

# --- Helper Function for Relative Pose Calculation (Unchanged) ---
def calculate_relative_pose(pose_wc_prev, pose_wc_curr):
    """
    Calculates the relative pose T_curr_prev, which transforms points
    from the PREVIOUS camera frame to the CURRENT camera frame.
    Inputs pose_wc_prev and pose_wc_curr are World-to-Camera poses (T_wc).

    Formula: T_curr_prev = T_wc_curr @ T_wc_prev_inv
             R_curr_prev = R_wc_curr @ R_wc_prev.T
             t_curr_prev = t_wc_curr - R_curr_prev @ t_wc_prev

    Args:
        pose_wc_prev (tuple): (R_wc_prev, t_wc_prev) World-to-Camera pose for the previous frame.
        pose_wc_curr (tuple): (R_wc_curr, t_wc_curr) World-to-Camera pose for the current frame.

    Returns:
        tuple: (R_rel_ortho, t_rel) or (None, None) if calculation fails.
               R_rel_ortho is the orthogonalized relative rotation (R_curr_prev).
               t_rel is the relative translation (t_curr_prev).
    """
    if pose_wc_prev is None or pose_wc_curr is None:
        logging.warning("Cannot calculate relative pose: Missing input pose(s).")
        return None, None
    if not isinstance(pose_wc_prev, tuple) or len(pose_wc_prev) != 2 or \
       not isinstance(pose_wc_curr, tuple) or len(pose_wc_curr) != 2:
        logging.error("Invalid pose format for relative pose calculation. Expected (R_wc, t_wc).")
        return None, None

    try:
        R_wc_prev, t_wc_prev = pose_wc_prev
        R_wc_curr, t_wc_curr = pose_wc_curr

        # Input validation (Rotation matrices)
        if not isinstance(R_wc_prev, np.ndarray) or R_wc_prev.shape != (3,3) or \
           not isinstance(R_wc_curr, np.ndarray) or R_wc_curr.shape != (3,3):
             logging.error("Invalid Rotation matrix in relative pose calculation.")
             return None, None
        # Basic check for rotation properties (optional but good)
        # if abs(np.linalg.det(R_wc_prev) - 1.0) > 1e-3 or abs(np.linalg.det(R_wc_curr) - 1.0) > 1e-3:
        #     logging.warning("Input rotation matrices might not be proper rotations (det != 1).")

        # Input validation (Translation vectors)
        if not isinstance(t_wc_prev, np.ndarray) or not isinstance(t_wc_curr, np.ndarray):
             logging.error("Invalid Translation vector in relative pose calculation.")
             return None, None

        # Ensure translations are column vectors (3, 1) for calculation
        t_wc_prev = t_wc_prev.reshape(3, 1)
        t_wc_curr = t_wc_curr.reshape(3, 1)

        # --- CORRECTED FORMULA for T_curr_prev = T_wc_curr @ T_wc_prev_inv ---
        # R_curr_prev = R_wc_curr @ R_wc_prev.T
        R_rel = R_wc_curr @ R_wc_prev.T

        # t_curr_prev = t_wc_curr - R_curr_prev @ t_wc_prev
        # Use the calculated R_rel (which is R_curr_prev) here
        t_rel = t_wc_curr - (R_rel @ t_wc_prev)
        # --- End of Corrected Formula ---

        # Orthogonalize the resulting relative rotation for stability
        R_rel_ortho = orthogonalize_rotation_matrix(R_rel)
        if R_rel_ortho is None:
            logging.warning("Orthogonalization failed for calculated relative rotation. Returning non-orthogonalized.")
            # Consider if you want to return the potentially non-orthogonal R_rel or None
            # Returning None for consistency with previous error handling:
            return None, None
            # Alternatively, return the non-orthogonalized version:
            # R_rel_ortho = R_rel # Fallback

        # Return the orthogonalized rotation and the calculated translation (flattened)
        return R_rel_ortho, t_rel.flatten()

    except np.linalg.LinAlgError as e:
         logging.error(f"Linear algebra error during relative pose calculation: {e}", exc_info=True)
         return None, None
    except Exception as e:
        logging.error(f"Unexpected error calculating relative pose: {e}", exc_info=True)
        return None, None


# --- *** NEW HELPER FUNCTION *** ---
def calculate_variance_vs_reference_pose(sub_area_poses, reference_pose):
    """
    Calculates the variance of sub-area poses relative to a given reference pose.

    Args:
        sub_area_poses (dict): Dictionary {seg_id: (R_sub, t_sub)} of locally estimated poses.
        reference_pose (tuple): The reference pose (R_ref, t_ref) to compare against.
                                R_ref should be orthogonalized.

    Returns:
        tuple: (rot_variance_vs_ref, trans_variance_vs_ref)
               - rot_variance_vs_ref: Mean squared angular distance (deg^2) between
                                      sub-area rotations and R_ref.
               - trans_variance_vs_ref: Mean squared Euclidean distance between
                                        normalized sub-area translations and normalized t_ref.
               Returns (np.nan, np.nan) if calculation is not possible.
    """
    if not sub_area_poses:
        logging.debug("No sub-area poses provided for variance calculation vs reference.")
        return np.nan, np.nan
    if reference_pose is None or reference_pose[0] is None or reference_pose[1] is None:
        logging.warning("Invalid reference pose provided for variance calculation.")
        return np.nan, np.nan

    R_ref, t_ref = reference_pose
    # norm_t_ref = normalize_translation(t_ref)
    norm_t_ref = t_ref

    if norm_t_ref is None:
        logging.warning("Reference translation vector normalization failed.")
        # We might still be able to calculate rotation variance
        # return np.nan, np.nan # Option: fail both if t_ref is bad
        pass # Continue to calculate rotation variance if possible

    valid_poses = {k: v for k, v in sub_area_poses.items() if v is not None and v[0] is not None and v[1] is not None}
    if not valid_poses:
        logging.debug("No valid sub-area poses found for variance calculation vs reference.")
        return np.nan, np.nan # Return NaN if no valid poses to compare

    rot_distances_sq = []
    trans_distances_sq = []

    for seg_id, (R_sub, t_sub) in valid_poses.items():
        # Rotation variance calculation
        try:
            # R_ref is assumed to be pre-orthogonalized
            # Optionally orthogonalize R_sub too for robustness?
            # R_sub_ortho = orthogonalize_rotation_matrix(R_sub)
            # if R_sub_ortho is None: continue # Skip if sub-area R is bad

            # Use original R_sub unless it causes issues
            dist_rad = angular_distance(R_sub, R_ref)
            if np.isfinite(dist_rad):
                rot_distances_sq.append(dist_rad**2)
            # else: # Log if a specific comparison fails
            #     logging.debug(f"Angular distance failed for seg {seg_id} vs reference.")
        except Exception as e:
            logging.error(f"Error calculating angular distance for seg {seg_id} vs reference: {e}")

        # Translation variance calculation (if ref translation is valid)
        if norm_t_ref is not None:
            try:
                # norm_t_sub = normalize_translation(t_sub)
                norm_t_sub = t_sub
                if norm_t_sub is not None:
                    dist_sq = np.linalg.norm(norm_t_sub - norm_t_ref)**2
                    trans_distances_sq.append(dist_sq)
                # else: # Log if sub-area normalization fails
                #    logging.debug(f"Normalization failed for sub-area translation {seg_id}.")
            except Exception as e:
                logging.error(f"Error calculating translation distance for seg {seg_id} vs reference: {e}")

    # Calculate mean squared distances (variances)
    # 剔除 IQR 外的异常点
    trimmed = trim_outliers_iqr(rot_distances_sq, cov=98.0)
    rot_variance_vs_ref = np.mean(trimmed)
    # trimmed = trim_outliers_iqr(trans_distances_sq, k=2)
    trans_variance_vs_ref = np.mean(trans_distances_sq)

    # Handle case where reference translation was invalid from the start
    if norm_t_ref is None and not trans_distances_sq:
        trans_variance_vs_ref = np.nan

    return rot_variance_vs_ref, trans_variance_vs_ref
# --- End of NEW HELPER FUNCTION ---


# --- Helper Function for Depth Warping (Unchanged) ---
def warp_depth(depth_prev, K_prev, K_curr, pose_curr_from_prev, depth_curr_shape, distCoeffs=None): # Added K_prev, K_curr
    # (Implementation unchanged)
    h_prev, w_prev = depth_prev.shape
    h_curr, w_curr = depth_curr_shape

    if K_prev is None or not isinstance(K_prev, np.ndarray) or K_prev.shape != (3, 3):
        logging.error("warp_depth received invalid K_prev.")
        return np.zeros(depth_curr_shape, dtype=np.float32)
    if K_curr is None or not isinstance(K_curr, np.ndarray) or K_curr.shape != (3, 3):
        logging.error("warp_depth received invalid K_curr.")
        return np.zeros(depth_curr_shape, dtype=np.float32)

    fx_prev, fy_prev, cx_prev, cy_prev = K_prev[0, 0], K_prev[1, 1], K_prev[0, 2], K_prev[1, 2]
    fx_curr, fy_curr, cx_curr, cy_curr = K_curr[0, 0], K_curr[1, 1], K_curr[0, 2], K_curr[1, 2]

    if pose_curr_from_prev is None or pose_curr_from_prev[0] is None or pose_curr_from_prev[1] is None:
        logging.warning("Invalid pose provided for depth warping.")
        return np.zeros(depth_curr_shape, dtype=np.float32)

    R_curr_prev, t_curr_prev = pose_curr_from_prev
    t_curr_prev = t_curr_prev.reshape(3, 1)

    depth_warped = np.zeros(depth_curr_shape, dtype=np.float32)
    z_buffer = np.full(depth_curr_shape, np.inf, dtype=np.float32)

    # Optimize by generating pixel coordinates once
    vs_prev, us_prev = np.meshgrid(np.arange(h_prev), np.arange(w_prev), indexing='ij')
    pixels_prev = np.stack((us_prev.flatten(), vs_prev.flatten()), axis=-1)
    depth_values_prev = depth_prev.flatten()

    valid_depth_mask = depth_values_prev > 1e-5
    pixels_prev_valid = pixels_prev[valid_depth_mask]
    depth_values_prev_valid = depth_values_prev[valid_depth_mask]
    us_prev_valid = pixels_prev_valid[:, 0]
    vs_prev_valid = pixels_prev_valid[:, 1]

    # Unproject valid points using K_prev
    X_prev_valid = (us_prev_valid - cx_prev) * depth_values_prev_valid / fx_prev
    Y_prev_valid = (vs_prev_valid - cy_prev) * depth_values_prev_valid / fy_prev
    P_prev_cam_valid = np.stack((X_prev_valid, Y_prev_valid, depth_values_prev_valid), axis=-1) # Shape (N, 3)

    # Transform to 3D points in camera (i) frame using relative pose
    # P_curr = R @ P_prev + t -> P_curr.T = P_prev.T @ R.T + t.T
    P_curr_cam_valid = (R_curr_prev @ P_prev_cam_valid.T + t_curr_prev).T # Shape (N, 3)
    X_curr_valid = P_curr_cam_valid[:, 0]
    Y_curr_valid = P_curr_cam_valid[:, 1]
    Z_curr_valid = P_curr_cam_valid[:, 2]

    # Filter points with valid depth in current frame
    valid_z_curr_mask = Z_curr_valid > 1e-5
    X_curr_proj = X_curr_valid[valid_z_curr_mask]
    Y_curr_proj = Y_curr_valid[valid_z_curr_mask]
    Z_curr_proj = Z_curr_valid[valid_z_curr_mask]

    if len(Z_curr_proj) == 0:
        logging.debug("No points remained after warping and Z>0 check.")
        return depth_warped # Return empty map

    # Project to 2D pixels in camera (i) frame using K_curr
    us_curr_proj = (X_curr_proj * fx_curr / Z_curr_proj) + cx_curr
    vs_curr_proj = (Y_curr_proj * fy_curr / Z_curr_proj) + cy_curr

    # Round and clip coordinates
    us_idx = np.round(us_curr_proj).astype(int)
    vs_idx = np.round(vs_curr_proj).astype(int)

    # Create mask for points within current image bounds
    valid_bounds_mask = (us_idx >= 0) & (us_idx < w_curr) & (vs_idx >= 0) & (vs_idx < h_curr)

    us_final = us_idx[valid_bounds_mask]
    vs_final = vs_idx[valid_bounds_mask]
    Z_final = Z_curr_proj[valid_bounds_mask]

    if len(Z_final) == 0:
        logging.debug("No points remained after boundary check.")
        return depth_warped # Return empty map

    # Z-buffering: Sort points by depth (ascending) to draw farther points first
    # Then iterate in reverse (descending) to draw closer points last
    sort_indices = np.argsort(Z_final)
    us_final_sorted = us_final[sort_indices]
    vs_final_sorted = vs_final[sort_indices]
    Z_final_sorted = Z_final[sort_indices]

    # Efficiently update depth_warped using sorted indices
    # This overwrites pixels, effectively keeping the smallest Z (closest point)
    depth_warped[vs_final_sorted, us_final_sorted] = Z_final_sorted

    # Optional: Use the z_buffer approach if preferred, but direct assignment is often faster
    # for i in range(len(us_final)):
    #     u_idx, v_idx, Z_curr = us_final[i], vs_final[i], Z_final[i]
    #     if Z_curr < z_buffer[v_idx, u_idx]:
    #         depth_warped[v_idx, u_idx] = Z_curr
    #         z_buffer[v_idx, u_idx] = Z_curr

    return depth_warped

# 在文件顶部引入这两个常量（与 compute_video_score 保持一致）
METRIC_KEYS = [
    'average_local_rotation_variance',
    'average_local_translation_variance',
    'average_reprojection_error',
    'average_global_rotation_variance',
    'average_global_translation_variance',
    'average_depth_consistency_error',
]

def parse_summary_file(summary_path):
    """
    与前面相同：从 summary txt 文件中提取 normalization parameters 和 PCA weights
    """
    with open(summary_path, 'r') as f:
        text = f.read()
    norm_match = re.search(
        r"--- Normalization Parameters Used .*?---\n\s*(\{.*?\})\n\n",
        text, flags=re.S
    )
    weight_match = re.search(
        r"--- PCA-Derived Weights .*?Calculated PCA Weights:\s*(\{.*?\})\n",
        text, flags=re.S
    )
    if not norm_match or not weight_match:
        raise ValueError("无法在 summary 文件中找到所需的 JSON 块，请检查格式。")
    norm_params = json.loads(norm_match.group(1))
    pca_weights = json.loads(weight_match.group(1))
    return norm_params, pca_weights
# --- Main Processing Function (MODIFIED) ---

def evaluate_3d_consistency(args):
    # --- Setup Debug Dirs (Unchanged) ---
    debug_root = "check"
    static_debug_dir = os.path.join(debug_root, "static")
    segments_debug_dir = os.path.join(debug_root, "segments")
    calib_debug_dir = os.path.join(debug_root, "calib_points")
    warp_debug_dir = os.path.join(debug_root, "warped_depth")
    plot_debug_dir = os.path.join(debug_root, "frames_plots", args.video_name)
    for d in (static_debug_dir, segments_debug_dir, calib_debug_dir, warp_debug_dir, plot_debug_dir):
        os.makedirs(d, exist_ok=True)

    # --- Get list of data files (Unchanged) ---
    try:
        frame_files = sorted([f for f in os.listdir(args.frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        # Load depths
        depth_files = sorted([f for f in os.listdir(args.depth_dir) if f.lower().endswith('.npz')])
        if not depth_files: raise ValueError("No .npz depth files found.")
        npz_path = os.path.join(args.depth_dir, depth_files[0])
        logging.info(f"Loading depths from: {npz_path}")
        npz = np.load(npz_path, allow_pickle=False)
        if 'depths' not in npz: raise ValueError("'depths' key not found in NPZ file.")
        depths_all = npz['depths']
        if args.mos_dir is not None:
            mos_files = sorted([f for f in os.listdir(args.mos_dir) if f.lower().endswith('.png')])
        else:
            mos_files = None

        num_frames = len(frame_files)
        if mos_files is not None:
            if not (num_frames == depths_all.shape[0] == len(mos_files)):
                raise ValueError(f"Mismatch in file counts: "
                                f"{len(frame_files)} frames ({args.frames_dir}), "
                                f"{depths_all.shape[0]} depths ({args.depth_dir}), "
                                f"{len(mos_files)} MOS ({args.mos_dir}).")
        if num_frames < 2:
             raise ValueError("Need at least two frames to evaluate consistency.")
        logging.info(f"Found {num_frames} frames, depth maps, and MOS masks.")

        # Semantic files validation (unchanged)
        semantic_files = []
        if args.semantics_dir and os.path.isdir(args.semantics_dir):
             semantic_files = sorted([f for f in os.listdir(args.semantics_dir) if f.lower().endswith('.png')])
             if len(semantic_files) != num_frames:
                 logging.warning(f"Semantic mask count ({len(semantic_files)}) differs from frame count ({num_frames}). Matching may be incorrect.")
             else:
                  logging.info(f"Found {len(semantic_files)} semantic masks.")
        elif args.semantics_dir:
             logging.warning(f"Semantic directory specified but not found: {args.semantics_dir}")

    except FileNotFoundError as e:
        logging.error(f"Input directory not found: {e}. Please check paths.")
        return None
    except ValueError as e:
        logging.error(f"Input error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error during initial file loading: {e}", exc_info=True)
        return None

    # --- Tracking Data Directory (Unchanged) ---
    if not os.path.isdir(args.tracks_dir):
        logging.error(f"Tracking directory not found: {args.tracks_dir}")
        return None
    logging.info(f"Using track directory: {args.tracks_dir}")

    # --- VGGT Initialization and Inference (Unchanged) ---
    logging.info("--- Initializing VGGT Model ---")
    frame_files_basenames = sorted([f for f in os.listdir(args.frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    frame_files_fullpaths = [os.path.join(args.frames_dir, f) for f in frame_files_basenames]
    vggt_poses = {} # Stores { frame_idx: (R_wc, t_wc) }
    vggt_intrinsics = {} # Stores { frame_idx: K }
    num_frames = len(frame_files_fullpaths)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_capability = (0, 0)
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if device == "cuda" and compute_capability[0] >= 8 else torch.float16
        logging.info(f"Using device: {device}, dtype: {dtype}")

        model = VGGT.from_pretrained(args.vggt_model_name).to(device).eval()

        logging.info(f"Loading and preprocessing {num_frames} images for VGGT...")
        images_tensor_4d = load_and_preprocess_images(frame_files_fullpaths).to(device)
        images_tensor_5d = images_tensor_4d.unsqueeze(0) # Add batch dim: [1, N, C, H, W]
        logging.debug(f"Reshaped images tensor for aggregator to: {images_tensor_5d.shape}")

        logging.info("--- Running VGGT Inference (Aggregator + Camera Head) ---")
        pose_encodings = None
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device=="cuda"), dtype=dtype):
                logging.debug(f"Running VGGT aggregator with input shape: {images_tensor_5d.shape}")
                aggregated_tokens_list, ps_idx = model.aggregator(images_tensor_5d)
                logging.debug(f"Aggregator output token list length: {len(aggregated_tokens_list)}")
                logging.debug("Running VGGT camera_head...")
                pose_encodings = model.camera_head(aggregated_tokens_list)[-1]
                logging.debug(f"Camera_head output shape: {pose_encodings.shape}")

        logging.info("--- Processing VGGT Outputs ---")
        if pose_encodings is None:
             raise RuntimeError("VGGT camera head did not produce pose encodings.")
        # Expected shape [1, N, 9] after latest pose_enc updates
        expected_enc_shape = (1, num_frames, 9)
        if pose_encodings.shape != expected_enc_shape:
             raise RuntimeError(f"VGGT pose encoding has unexpected shape {pose_encodings.shape}. Expected {expected_enc_shape}.")

        img_shape_hw = (images_tensor_5d.shape[-2], images_tensor_5d.shape[-1])

        try:
            extrinsic_matrices, intrinsic_matrices = pose_encoding_to_extri_intri(
                pose_encodings, # Pass the full [1, num_frames, 9] tensor
                img_shape_hw
            )
            actual_ext_shape = extrinsic_matrices.shape
            actual_int_shape = intrinsic_matrices.shape
            logging.info(f"Vectorized call results: ACTUAL Extrinsic shape: {actual_ext_shape}, ACTUAL Intrinsic shape: {actual_int_shape}")

            expected_ext_shape = (1, num_frames, 3, 4)
            expected_int_shape = (1, num_frames, 3, 3)
            if actual_ext_shape != expected_ext_shape:
                raise ValueError(f"Vectorized extrinsic matrix shape {actual_ext_shape} does not match expected {expected_ext_shape}.")
            if actual_int_shape != expected_int_shape:
                 raise ValueError(f"Vectorized intrinsic matrix shape {actual_int_shape} does not match expected {expected_int_shape}.")

            R_wc_batch = extrinsic_matrices[0, :, :3, :3]  # Shape: [num_frames, 3, 3]
            t_wc_batch = extrinsic_matrices[0, :, :3, 3]   # Shape: [num_frames, 3]
            K_batch = intrinsic_matrices[0]                # Shape: [num_frames, 3, 3]

            valid_k_mask = (K_batch[:, 0, 0] > 0) & (K_batch[:, 1, 1] > 0) & \
                           (K_batch[:, 0, 2] > 0) & (K_batch[:, 1, 2] > 0)
            invalid_k_indices = torch.where(~valid_k_mask)[0]
            if len(invalid_k_indices) > 0:
                logging.warning(f"Found potentially invalid intrinsics (via vectorized check) for frames: {invalid_k_indices.tolist()}")

            R_np = R_wc_batch.detach().cpu().float().numpy()
            t_np = t_wc_batch.detach().cpu().float().numpy()
            K_np = K_batch.detach().cpu().float().numpy()

            logging.info("Populating final dictionaries from vectorized results...")
            for i in range(num_frames):
                 # Orthogonalize rotation here before storing
                 R_ortho = orthogonalize_rotation_matrix(R_np[i])
                 if R_ortho is None:
                     logging.warning(f"Failed to orthogonalize VGGT R_wc for frame {i}. Storing original.")
                     R_ortho = R_np[i] # Store original if failed
                 vggt_poses[i] = (R_ortho, t_np[i]) # Store orthogonalized R
                 vggt_intrinsics[i] = K_np[i]

                 if i < 3: # Log first few final values
                     logging.debug(f"VGGT Frame {i} (Stored): K=\n{vggt_intrinsics[i]}")
                     logging.debug(f"VGGT Frame {i} (Stored): R_wc (ortho'd)=\n{vggt_poses[i][0]}")
                     logging.debug(f"VGGT Frame {i} (Stored): t_wc={vggt_poses[i][1]}")

            logging.info(f"Successfully extracted and stored vectorized VGGT poses/intrinsics for {len(vggt_poses)} frames.")
            if len(vggt_poses) != num_frames:
                 logging.warning(f"Mismatch in final dictionary size! ({len(vggt_poses)}/{num_frames})")

        except ValueError as e_val:
             logging.error(f"ValueError during vectorized processing: {e_val}", exc_info=True)
             raise
        except Exception as e_vec:
             logging.error(f"An unexpected error occurred during vectorized processing: {e_vec}", exc_info=True)
             raise

        # Clean up GPU memory
        del model, images_tensor_4d, images_tensor_5d, aggregated_tokens_list, ps_idx, pose_encodings
        del extrinsic_matrices, intrinsic_matrices, R_wc_batch, t_wc_batch, K_batch
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        logging.info(f"Successfully extracted VGGT poses/intrinsics for {len(vggt_poses)} frames.")
        if len(vggt_poses) != num_frames:
            logging.warning(f"VGGT did not provide outputs for all input frames! ({len(vggt_poses)}/{num_frames})")

    except ImportError:
         logging.error("VGGT or its dependencies (torch, etc.) not found. Please install them.")
         return None
    except FileNotFoundError:
         logging.error(f"VGGT model '{args.vggt_model_name}' not found. Check model name or download.")
         return None
    except RuntimeError as e:
         logging.error(f"Runtime error during VGGT processing: {e}", exc_info=True)
         return None
    except Exception as e:
         logging.error(f"An unexpected error occurred during VGGT processing: {e}", exc_info=True)
         return None
    # --- End of VGGT Section ---


    # --- Initialize Results Storage (MODIFIED Keys) ---
    all_results = {
        'frame_local_consistency_rot_var': [],  # List of float (local rot variance)
        'frame_local_consistency_trans_var': [],# List of float (local trans variance)
        'frame_reprojection_errors': [],        # List of float (avg reproj error per pair)
        'frame_global_consistency_rot_var': [], # List of float (rot variance vs VGGT) <-- CHANGED
        'frame_global_consistency_trans_var':[],# List of float (trans variance vs VGGT) <-- CHANGED
        'frame_depth_consistency_error': [],    # List of float (avg depth diff)
        'mean_local_relative_poses': [],        # List of (R, t) or None (mean local pose)
        'vggt_relative_poses': [],              # List of (R, t) or None (relative VGGT pose)
        'num_valid_tracks_per_pair': [],        # List of int
        'num_pnp_poses_per_pair': [],           # List of int
        'skipped_pairs': 0,
    }

    # --- Process Frame by Frame ---
    prev_data = {}
    processed_pairs_count = 0
    for i in range(num_frames):
        frame_basename_no_ext = os.path.splitext(frame_files[i])[0]
        frame_filename = frame_files[i]
        logging.info(f"--- Processing Frame {i}/{num_frames-1} ({frame_filename}) ---")

        # Construct paths (unchanged)
        frame_path = os.path.join(args.frames_dir, frame_filename)
        depth_slice = depths_all[i]
        if args.mos_dir is not None:
            mos_path = os.path.join(args.mos_dir, frame_basename_no_ext + ".png")
            if not os.path.exists(mos_path): logging.warning(f"MOS file not found: {mos_path}"); mos_path = None # Allow processing without MOS? Maybe error instead.
        else:
            mos_path = None
        
        semantic_path = None
        if args.semantics_dir and i < len(semantic_files):
            # Robust semantic file matching (unchanged)
            expected_semantic_file = frame_basename_no_ext + ".png"
            temp_semantic_path = os.path.join(args.semantics_dir, expected_semantic_file)
            if os.path.exists(temp_semantic_path): semantic_path = temp_semantic_path
            else:
                semantic_basename_idx = os.path.splitext(semantic_files[i])[0]
                if semantic_basename_idx == frame_basename_no_ext:
                    temp_semantic_path_idx = os.path.join(args.semantics_dir, semantic_files[i])
                    if os.path.exists(temp_semantic_path_idx):
                         semantic_path = temp_semantic_path_idx
                         logging.debug(f"Used index-based match for semantic file: {semantic_files[i]}")
                    else: logging.warning(f"Semantic file listed but not found: {temp_semantic_path_idx}")
                else: logging.warning(f"Frame {i}: No semantic mask found for {expected_semantic_file}")


        # Check for VGGT intrinsics FIRST (unchanged)
        if i not in vggt_intrinsics:
            logging.error(f"Skipping frame {i} ({frame_files_basenames[i]}): Missing VGGT intrinsics.")
            prev_data = {}
            if i > 0: # Append NaN placeholders if skipping affects a pair
                all_results['skipped_pairs'] += 1
                all_results['frame_local_consistency_rot_var'].append(np.nan)
                all_results['frame_local_consistency_trans_var'].append(np.nan)
                all_results['frame_reprojection_errors'].append(np.nan)
                all_results['frame_global_consistency_rot_var'].append(np.nan) # Changed key
                all_results['frame_global_consistency_trans_var'].append(np.nan)# Changed key
                all_results['frame_depth_consistency_error'].append(np.nan)
                all_results['mean_local_relative_poses'].append(None)
                all_results['vggt_relative_poses'].append(None)
                all_results['num_valid_tracks_per_pair'].append(0)
                all_results['num_pnp_poses_per_pair'].append(0)
            continue

        K_curr = vggt_intrinsics[i]

        # --- Load Frame Data (unchanged) ---
        try:
            if not frame_path:
                 raise IOError(f"Missing required files for frame {i} (frame/mos).")

            frame, depth, mos_mask, semantic_mask = load_frame_data(
                frame_path, depth_slice, mos_path, semantic_path, args.depth_scale
            )
        except IOError as e:
            logging.error(f"Skipping frame {i} due to loading error: {e}")
            prev_data = {}
            if i > 0: # Append NaN placeholders
                all_results['skipped_pairs'] += 1
                all_results['frame_local_consistency_rot_var'].append(np.nan)
                all_results['frame_local_consistency_trans_var'].append(np.nan)
                all_results['frame_reprojection_errors'].append(np.nan)
                all_results['frame_global_consistency_rot_var'].append(np.nan) # Changed key
                all_results['frame_global_consistency_trans_var'].append(np.nan)# Changed key
                all_results['frame_depth_consistency_error'].append(np.nan)
                all_results['mean_local_relative_poses'].append(None)
                all_results['vggt_relative_poses'].append(None)
                all_results['num_valid_tracks_per_pair'].append(0)
                all_results['num_pnp_poses_per_pair'].append(0)
            continue
        except Exception as e:
             logging.error(f"Unexpected error loading data for frame {i}: {e}", exc_info=True)
             prev_data = {}
             if i > 0: # Append NaN placeholders
                all_results['skipped_pairs'] += 1
                all_results['frame_local_consistency_rot_var'].append(np.nan)
                all_results['frame_local_consistency_trans_var'].append(np.nan)
                all_results['frame_reprojection_errors'].append(np.nan)
                all_results['frame_global_consistency_rot_var'].append(np.nan) # Changed key
                all_results['frame_global_consistency_trans_var'].append(np.nan)# Changed key
                all_results['frame_depth_consistency_error'].append(np.nan)
                all_results['mean_local_relative_poses'].append(None)
                all_results['vggt_relative_poses'].append(None)
                all_results['num_valid_tracks_per_pair'].append(0)
                all_results['num_pnp_poses_per_pair'].append(0)
             continue

        # 1. Preprocess: Isolate Static Scene (unchanged)
        static_frame, static_depth, static_mask = apply_masks(
            frame, depth, mos_mask, semantic_mask, args.static_semantic_labels
        )

        # DEBUG Visualizations (Static, Segments - Unchanged)
        if args.debug_vis_interval > 0 and i % args.debug_vis_interval == 0:
             cv2.imwrite(os.path.join(static_debug_dir, f"{frame_basename_no_ext}_static_frame.png"), static_frame)
             depth_vis = cv2.normalize(static_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
             cv2.imwrite(os.path.join(static_debug_dir, f"{frame_basename_no_ext}_static_depth.png"), depth_vis)

        # 2. Segment Static Background (Current Frame i - using Depth-Aware SLIC)
        # sub_area_segments, num_segments = segment_static_background(
        #     static_frame, static_depth, static_mask,
        #     n_segments=args.n_slic_segments,
        #     compactness=args.slic_compactness,
        #     depth_weight=args.depth_aware_slic_weight,
        #     min_size_filter=args.min_segment_size,
        #     sigma=args.slic_sigma
        # )
        sub_area_segments, num_segments = segment_static_background(
            static_frame, static_depth, static_mask,
            method=args.segmentation_method,
            semantic_mask=semantic_mask, static_labels=args.static_semantic_labels,
            n_depth_clusters=args.n_depth_clusters,
            n_slic_segments=args.n_slic_segments, compactness=args.slic_compactness,
            min_size_filter=args.min_segment_size
        )

        # DEBUG Visualization (Segments - Unchanged)
        if args.debug_vis_interval > 0 and num_segments > 0 and i % args.debug_vis_interval == 0:
             seg_norm = ((sub_area_segments.astype(np.float32) / num_segments) * 255).astype(np.uint8)
             seg_colormap = cv2.applyColorMap(seg_norm, cv2.COLORMAP_JET)
             seg_colormap[sub_area_segments == 0] = [0, 0, 0] # Make non-segmented areas black
             cv2.imwrite(os.path.join(segments_debug_dir, f"{frame_basename_no_ext}_segments_colormap.png"), seg_colormap)
             overlay = cv2.addWeighted(static_frame, 0.7, seg_colormap, 0.3, 0)
             cv2.imwrite(os.path.join(segments_debug_dir, f"{frame_basename_no_ext}_segments_overlay.png"), overlay)

        # Store data for current frame (unchanged)
        current_data = {
            'depth': depth, 'sub_area_segments': sub_area_segments, 'num_segments': num_segments,
            'frame_basename': frame_basename_no_ext,
            'K': K_curr,
            'static_mask': static_mask,
            'frame_index': i
        }

        # --- Analysis requiring frame pair (i > 0) ---
        if i == 0:
            prev_data = current_data
            continue # Skip to next frame

        # --- Load Tracks for Pair (i-1, i) (Logic Unchanged, added valid track count) ---
        tracks_for_pnp = None
        num_valid_pair_tracks = 0
        skip_pair = False
        if not prev_data or 'frame_index' not in prev_data:
            logging.warning(f"Skipping pair ({i-1}, {i}) because previous frame data is missing.")
            skip_pair = True
        else:
            prev_frame_idx = prev_data['frame_index']
            # Check VGGT data for the PAIR
            if prev_frame_idx not in vggt_poses or i not in vggt_poses or \
               prev_frame_idx not in vggt_intrinsics: # K_curr already checked
                logging.warning(f"Skipping pair ({i-1}, {i}): Missing VGGT pose/intrinsics for frame {prev_frame_idx} or {i}.")
                skip_pair = True

        if not skip_pair:
            # Track loading logic (unchanged)
            try:
                prev_basename = prev_data['frame_basename']
                curr_basename = current_data['frame_basename']
                track_file_prev_coords = os.path.join(args.tracks_dir, f"{prev_basename}_{prev_basename}.npy")
                track_file_curr_state = os.path.join(args.tracks_dir, f"{prev_basename}_{curr_basename}.npy")

                if not os.path.exists(track_file_curr_state) or not os.path.exists(track_file_prev_coords):
                    logging.warning(f"Track file(s) missing for PnP: Pair ({prev_basename} -> {curr_basename})")
                    skip_pair = True
                else:
                    data_prev_coords = np.load(track_file_prev_coords)
                    data_curr = np.load(track_file_curr_state)

                    if data_prev_coords.shape[0] != data_curr.shape[0] or data_prev_coords.ndim != 2 or data_curr.ndim != 2 or data_prev_coords.shape[1] < 4 or data_curr.shape[1] < 4:
                         logging.warning(f"Track data shape mismatch or format error for pair ({prev_basename} -> {curr_basename}). Prev: {data_prev_coords.shape}, Curr: {data_curr.shape}")
                         skip_pair = True
                    else:
                        points_prev = data_prev_coords[:, :2]
                        vis_prev = data_prev_coords[:, 2]
                        points_curr = data_curr[:, :2]
                        vis_curr = data_curr[:, 2]
                        conf_curr = data_curr[:, 3]

                        # Filter based on visibility in *current* frame and confidence
                        visible_curr = (vis_curr > 0)
                        confident_enough = conf_curr >= args.min_track_confidence
                        valid_mask = visible_curr & confident_enough
                        num_valid_pair_tracks = np.sum(valid_mask)

                        if num_valid_pair_tracks >= 4: # Need at least 4 for PnP
                            points_prev_filtered = points_prev[valid_mask]
                            points_curr_filtered = points_curr[valid_mask]
                            tracks_for_pnp = {'points_prev': points_prev_filtered.astype(np.float32),
                                              'points_curr': points_curr_filtered.astype(np.float32)}
                            logging.info(f"Pair ({i-1}, {i}): Found {num_valid_pair_tracks} valid tracks for PnP.")
                        else:
                             logging.warning(f"Pair ({i-1}, {i}): Insufficient valid tracks ({num_valid_pair_tracks}). Need >= 4.")
                             skip_pair = True

            except FileNotFoundError as e:
                 logging.warning(f"Error loading track files for pair ({i-1}, {i}): {e}")
                 skip_pair = True
            except Exception as e:
                 logging.error(f"Unexpected error processing tracks for pair ({i-1}, {i}): {e}", exc_info=True)
                 skip_pair = True

        # Store track count regardless of skip status (if calculated)
        all_results['num_valid_tracks_per_pair'].append(num_valid_pair_tracks)

        # Handle skipped pair (append NaNs/Nones)
        if skip_pair:
            all_results['skipped_pairs'] += 1
            all_results['frame_local_consistency_rot_var'].append(np.nan)
            all_results['frame_local_consistency_trans_var'].append(np.nan)
            all_results['frame_reprojection_errors'].append(np.nan)
            all_results['frame_global_consistency_rot_var'].append(np.nan) # Changed key
            all_results['frame_global_consistency_trans_var'].append(np.nan)# Changed key
            all_results['frame_depth_consistency_error'].append(np.nan)
            all_results['mean_local_relative_poses'].append(None)
            all_results['vggt_relative_poses'].append(None)
            all_results['num_pnp_poses_per_pair'].append(0)
            # Ensure track count list is updated if skipping happened before calc
            if len(all_results['num_valid_tracks_per_pair']) == processed_pairs_count:
                all_results['num_valid_tracks_per_pair'].append(0) # Append 0 if track loading failed early
            # Update prev_data logic
            if 'frame_index' in current_data: prev_data = current_data
            else: prev_data = {}
            continue # Move to the next frame

        # --- Pair Processing ---
        processed_pairs_count += 1
        K_prev = prev_data['K']

        # 3. Estimate Camera Poses per Sub-Area (Local PnP - Unchanged)
        sub_area_poses = estimate_pose_per_subarea(
            i, current_data['sub_area_segments'], current_data['num_segments'],
            tracks_for_pnp, prev_data['depth'],
            K_prev, K_curr, # Pass both intrinsics
            None, # distCoeffs
            args.pnp_reprojection_error
        )
        num_estimated_poses = len(sub_area_poses)
        all_results['num_pnp_poses_per_pair'].append(num_estimated_poses)
        logging.info(f"Pair ({i-1}, {i}): Estimated local poses for {num_estimated_poses}/{current_data['num_segments']} sub-areas.")

        # 4. Assess Local Consistency (Variance around local mean - Unchanged)
        local_rot_var, local_trans_var, mean_R_local, mean_t_local = assess_local_consistency(sub_area_poses)
        all_results['frame_local_consistency_rot_var'].append(local_rot_var)
        all_results['frame_local_consistency_trans_var'].append(local_trans_var)
        all_results['mean_local_relative_poses'].append((mean_R_local, mean_t_local) if mean_R_local is not None else None) # Store the mean local pose
        log_level = logging.INFO if np.isfinite(local_rot_var) and np.isfinite(local_trans_var) else logging.WARNING
        logging.log(log_level, f"Pair ({i-1}, {i}): Local Consistency: R Var={local_rot_var:.4f} (deg^2), T Var={local_trans_var:.4f}")


        # 5. *** MODIFIED: Calculate Global Consistency (Variance vs VGGT Pose) ***
        prev_frame_idx = prev_data['frame_index']
        curr_frame_idx = current_data['frame_index']
        global_rot_var, global_trans_var = np.nan, np.nan # Initialize variances
        R_vggt_rel, t_vggt_rel = None, None # Initialize relative VGGT pose

        # Retrieve VGGT world-to-camera poses (already checked they exist for the pair)
        pose_wc_prev_vggt = vggt_poses[prev_frame_idx]
        pose_wc_curr_vggt = vggt_poses[curr_frame_idx]

        # Calculate relative pose from VGGT poses (R is already orthogonalized by calculate_relative_pose)
        R_vggt_rel, t_vggt_rel = calculate_relative_pose(pose_wc_prev_vggt, pose_wc_curr_vggt)
        all_results['vggt_relative_poses'].append((R_vggt_rel, t_vggt_rel) if R_vggt_rel is not None else None) # Store relative VGGT pose

        if R_vggt_rel is not None and t_vggt_rel is not None:
            logging.debug(f"Pair ({i-1}, {i}): Calculated relative VGGT pose successfully.")
            reference_pose_vggt = (R_vggt_rel, t_vggt_rel)

            # --- Call the NEW helper function ---
            global_rot_var, global_trans_var = calculate_variance_vs_reference_pose(
                sub_area_poses, reference_pose_vggt
            )

            # --- 新增: 找到全局一致性中与 reference_pose_vggt 差距最大的区域 ---
            # outlier_global_seg = None
            # max_global_diff = -1.0
            # if reference_pose_vggt is not None:
            #     R_ref, t_ref = reference_pose_vggt
            #     for seg_id, (R_i, t_i) in sub_area_poses.items():
            #         angle = angular_distance(R_ref, R_i)
            #         t_dist = np.linalg.norm(t_i - t_ref)
            #         diff = angle + t_dist
            #         if diff > max_global_diff:
            #             max_global_diff = diff
            #             outlier_global_seg = seg_id
            # vis_dir = os.path.join("vis", args.video_name)
            # os.makedirs(vis_dir, exist_ok=True)
            # # 读取原始帧图像
            # orig_img = cv2.imread(os.path.join(args.frames_dir, frame_files[i]))
            # orig_img = static_frame
            # overlay = orig_img.copy()
            # # 本地异常区域轮廓: 红色
            # if outlier_local_seg is not None:
            #     mask_local = (current_data['sub_area_segments'] == outlier_local_seg).astype(np.uint8)
            #     contours, _ = cv2.findContours(mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     cv2.drawContours(overlay, contours, -1, (0,0,255), 2)
            # # 全局异常区域轮廓: 绿色
            # # if outlier_global_seg is not None:
            # #     mask_global = (current_data['sub_area_segments'] == outlier_global_seg).astype(np.uint8)
            # #     contours, _ = cv2.findContours(mask_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # #     cv2.drawContours(overlay, contours, -1, (0,255,0), 2)
            # # 保存可视化结果
            # out_path = os.path.join(vis_dir, f"pair_{i-1}_{i}_outliers.png")
            # cv2.imwrite(out_path, overlay)
            # logging.info(f"Saved outlier visualization: {out_path}")
            
            # --- MODIFIED VISUALIZATION: Poses with Segments Overlay ---
            # 1. Create base image with segment overlay (similar to _segments_overlay.png)
            vis_img_with_overlay = static_frame.copy() # Start with the static frame of current frame 'i'
            num_curr_segments = current_data.get('num_segments', 0)
            curr_sub_area_segments = current_data.get('sub_area_segments', None)

            if num_curr_segments > 0 and curr_sub_area_segments is not None:
                # Normalize segment IDs for colormap application
                # Using the same normalization as in the earlier segment debug visualization
                seg_norm = ((curr_sub_area_segments.astype(np.float32) / num_curr_segments) * 255).astype(np.uint8)
                seg_colormap = cv2.applyColorMap(seg_norm, cv2.COLORMAP_JET)
                
                # Make non-segmented areas (ID 0) black in the colormap
                seg_colormap[curr_sub_area_segments == 0] = [0, 0, 0]
                
                # Blend static frame with segment colormap
                alpha_segments = 0.3 # Transparency for segment colors (same as debug)
                vis_img_with_overlay = cv2.addWeighted(
                    static_frame,             # Source image 1 (current static frame)
                    1.0,                      # Weight for source image 1
                    seg_colormap,             # Source image 2 (segment colors)
                    alpha_segments,           # Weight for source image 2 (making it semi-transparent)
                    0                         # Gamma value
                )
            # else: vis_img_with_overlay remains a copy of static_frame if no segments

            # # 2. Prepare for drawing arrows on this new base image
            # vis_img = vis_img_with_overlay # All arrows will be drawn on this image
            # h, w = vis_img.shape[:2]
            # K_curr = current_data['K'] # Intrinsics for the current frame 'i'

            # # 3. Draw Global Reference Vector (Green Arrow) - Logic Unchanged
            # # (Uses reference_pose_vggt: R_ref, t_ref)
            # R_ref_viz, t_ref_viz = reference_pose_vggt
            # # axis3d_viz should be defined, e.g.:
            # axis3d_viz = np.array([[0,0,0], [1,1,1]/np.sqrt(3)], dtype=np.float32) 
            # rvec_ref_viz, _ = cv2.Rodrigues(R_ref_viz)
            # tvec_ref_viz = t_ref_viz.reshape(3,1)

            # MAX_DISPLAY_LENGTH_PX = 75.0
            # PERCENTILE_FOR_SCALING = 80  # 使用第90个百分位数
            # MIN_ARROW_COUNT_FOR_PERCENTILE = 10 # 计算百分位数的最小样本数
            # epsilon = 1e-6

            # arrow_data_for_drawing = []
            # all_original_lengths = [] # 存储所有有效的原始箭头长度

            # # --- PASS 1: 计算所有原始投影向量并收集有效长度 ---

            # # 1a. 全局参考位姿
            # anchor_global_x = w // 2
            # anchor_global_y = h // 2
            # rvec_ref_viz, _ = cv2.Rodrigues(R_ref_viz)
            # tvec_ref_viz_reshaped = t_ref_viz.reshape(3,1)

            # pts2d_ref_obj, _ = cv2.projectPoints(axis3d_viz, rvec_ref_viz, tvec_ref_viz_reshaped, K_curr, None)
            # p0_ref_proj = pts2d_ref_obj[0].ravel()
            # p1_ref_proj = pts2d_ref_obj[1].ravel()
            # v_ref_proj_x_orig = p1_ref_proj[0] - p0_ref_proj[0]
            # v_ref_proj_y_orig = p1_ref_proj[1] - p0_ref_proj[1]
            
            # current_length = math.sqrt(v_ref_proj_x_orig**2 + v_ref_proj_y_orig**2)
            # if current_length > epsilon: # 只收集有效长度
            #     all_original_lengths.append(current_length)
            
            # arrow_data_for_drawing.append({
            #     'vx_orig': v_ref_proj_x_orig, 'vy_orig': v_ref_proj_y_orig,
            #     'anchor_x': anchor_global_x, 'anchor_y': anchor_global_y,
            #     'color': (237,34,37), 'thickness': 4, 'tipLength': 0.1
            # })

            # # 1b. 局部子区域位姿
            # seg_map_for_centroids = current_data['sub_area_segments']
            # for seg_id, (R_i_viz, t_i_viz) in sub_area_poses.items():
            #     if seg_id == 0: continue
            #     mask_viz = (seg_map_for_centroids == seg_id).astype(np.uint8)
            #     M_viz = cv2.moments(mask_viz)
            #     if M_viz['m00'] < epsilon: continue
                
            #     cx_viz = int(M_viz['m10'] / M_viz['m00'])
            #     cy_viz = int(M_viz['m01'] / M_viz['m00'])
                
            #     rvec_i_viz, _ = cv2.Rodrigues(R_i_viz)
            #     tvec_i_viz_reshaped = t_i_viz.reshape(3,1)
                
            #     pts2d_i_obj, _ = cv2.projectPoints(axis3d_viz, rvec_i_viz, tvec_i_viz_reshaped, K_curr, None)
            #     p0_proj = pts2d_i_obj[0].ravel()
            #     p1_proj = pts2d_i_obj[1].ravel()
            #     v_proj_x_orig = p1_proj[0] - p0_proj[0]
            #     v_proj_y_orig = p1_proj[1] - p0_proj[1]
                
            #     current_length = math.sqrt(v_proj_x_orig**2 + v_proj_y_orig**2)
            #     if current_length > epsilon: # 只收集有效长度
            #         all_original_lengths.append(current_length)
                
            #     arrow_data_for_drawing.append({
            #         'vx_orig': v_proj_x_orig, 'vy_orig': v_proj_y_orig,
            #         'anchor_x': cx_viz, 'anchor_y': cy_viz,
            #         'color': (147,200,192), 'thickness': 4, 'tipLength': 0.1
            #     })

            # # --- 计算稳健的参考长度和通用缩放因子 ---
            # common_scale = 0.0 
            # reference_length_for_scaling = 0.0

            # if len(all_original_lengths) > 0:
            #     # 注意：np.percentile 需要输入是numpy array
            #     np_lengths = np.array(all_original_lengths)

            #     if len(np_lengths) >= MIN_ARROW_COUNT_FOR_PERCENTILE:
            #         reference_length_for_scaling = np.percentile(np_lengths, PERCENTILE_FOR_SCALING)
            #     else:
            #         # 如果箭头数量太少，使用最大值作为回退方案
            #         reference_length_for_scaling = np.max(np_lengths)
                
            #     if reference_length_for_scaling > epsilon:
            #         common_scale = MAX_DISPLAY_LENGTH_PX / reference_length_for_scaling
            #     # 如果 reference_length_for_scaling 依然接近0 (例如，所有长度都非常小)
            #     # common_scale 会非常大或保持为0。若要避免过大，可加条件：
            #     # common_scale = min(common_scale, SOME_MAX_ACCEPTABLE_SCALE_UP_FACTOR)
            #     # 但通常，如果P90都很小，说明整体pose变化不大，按比例放大是合理的。
            
            # # 调试打印 (可以按需保留或移除)
            # # num_total_arrows = len(arrow_data_for_drawing)
            # # print(f"Pair ({i-1}-{i}): Total arrows: {num_total_arrows}, Valid lengths collected: {len(all_original_lengths)}")
            # # if len(all_original_lengths) > 0:
            # #     print(f"  Lengths stats: Min={np.min(all_original_lengths):.2f}, Max={np.max(all_original_lengths):.2f}, P{PERCENTILE_FOR_SCALING}={np.percentile(np.array(all_original_lengths), PERCENTILE_FOR_SCALING):.2f if len(all_original_lengths) >= MIN_ARROW_COUNT_FOR_PERCENTILE else 'N/A'}")
            # # print(f"  Reference length for scaling: {reference_length_for_scaling:.2f}, Common_scale: {common_scale:.4f}")


            # # --- PASS 2: 使用 common_scale 绘制所有箭头 ---
            # for arrow_data in arrow_data_for_drawing:
            #     vx_scaled = arrow_data['vx_orig'] * common_scale
            #     vy_scaled = arrow_data['vy_orig'] * common_scale
                
            #     anchor_x = arrow_data['anchor_x']
            #     anchor_y = arrow_data['anchor_y']
                
            #     # 计算箭头的起点和终点，使其中心位于锚点
            #     arrow_start_x = int(round(anchor_x - vx_scaled / 2.0))
            #     arrow_start_y = int(round(anchor_y - vy_scaled / 2.0))
            #     arrow_end_x   = int(round(anchor_x + vx_scaled / 2.0))
            #     arrow_end_y   = int(round(anchor_y + vy_scaled / 2.0))
                
            #     # 只有当缩放后的箭头有可见长度时才绘制
            #     # （或者如果原始向量非零但缩放后变得很小，画一个点）
            #     if abs(vx_scaled) > epsilon or abs(vy_scaled) > epsilon : # 检查缩放后的向量是否几乎为零
            #             cv2.arrowedLine(vis_img,
            #                         (arrow_start_x, arrow_start_y),
            #                         (arrow_end_x, arrow_end_y),
            #                         arrow_data['color'], 
            #                         arrow_data['thickness'], 
            #                         tipLength=arrow_data['tipLength'])
            #     # 可选：如果原始向量非零，但缩放后长度为0，可以在锚点画个小点
            #     elif math.sqrt(arrow_data['vx_orig']**2 + arrow_data['vy_orig']**2) > epsilon :
            #         cv2.circle(vis_img, (anchor_x, anchor_y), arrow_data['thickness'], arrow_data['color'], -1)           


            # # 5. Save the combined visualization
            # vis_dir = os.path.join("vis", args.video_name)
            # os.makedirs(vis_dir, exist_ok=True)
            # # Changed filename to reflect the new content
            # out_path = os.path.join(vis_dir, f"pair_{i-1}_{i}_poses_with_segments_visualization.png")
            # cv2.imwrite(out_path, vis_img)
            # logging.info(f"Saved poses with segments visualization: {out_path}")
            # # --- End of MODIFIED VISUALIZATION ---

        else:
            logging.warning(f"Pair ({i-1}, {i}): Failed to calculate relative pose from VGGT. Cannot calculate global variance.")

        # Store the results (variances)
        all_results['frame_global_consistency_rot_var'].append(global_rot_var) # Changed key
        all_results['frame_global_consistency_trans_var'].append(global_trans_var) # Changed key
        log_level_global = logging.INFO if np.isfinite(global_rot_var) and np.isfinite(global_trans_var) else logging.WARNING
        logging.log(log_level_global, f"Pair ({i-1}, {i}): Global Consistency vs VGGT: R Var={global_rot_var:.4f} (deg^2), T Var={global_trans_var:.4f}")


        # --- START: 6. Quantify AND Visualize Reprojection Error ---
        reproj_vis_output_dir = 'vis'
        # Uses local poses (sub_area_poses)

        # points_prev_for_reproj are from tracks_for_pnp['points_prev']
        # points_curr_for_reproj are from tracks_for_pnp['points_curr']
        # These are already filtered by track confidence and visibility in current frame.

        # K_prev is prev_data['K'], K_curr is current_data['K']
        
        # Get 3D points in frame f_i-1 using its depth map for *all valid tracks in the pair*
        # The `tracks_for_pnp` already contains filtered points_prev and points_curr
        points_3d_prev_all_tracks, _, valid_mask_3d_prev = get_3d_points(
            tracks_for_pnp['points_prev'], prev_data['depth'], K_prev
        )
        # points_3d_prev_all_tracks corresponds to tracks_for_pnp['points_prev'][valid_mask_3d_prev]
        # The 2D points in current frame for these are tracks_for_pnp['points_curr'][valid_mask_3d_prev]

        avg_reproj_error_pair_val = np.nan # Renamed to avoid conflict with a list
        all_segment_reproj_errors_dict = {} # To store avg error per segment for color-coding and pair average

        # Base image for this pair's reprojection visualization: current static frame f_i
        # Create a clean copy for each pair's visualization if args.debug_vis_interval matches
        vis_image_reproj_pair = None
        vis_image_reproj_pair = vis_img_with_overlay.copy()
        h_img_reproj, w_img_reproj = vis_image_reproj_pair.shape[:2]

        # Visualization parameters
        actual_pt_color = (0, 255, 0)   # Green circles for actual points p_i
        reproj_pt_color = (0, 0, 255)    # Red 'x' for reprojected points Pi()
        error_line_color = (255, 255, 0)   # Cyan lines for error
        line_thickness_reproj = 1
        marker_size_circle_reproj = 3
        marker_size_cross_reproj = 4 # For cv2.drawMarker, size is more like radius

        # Data for overall pair visualization (collecting all points and lines for THIS pair)
        # These will be drawn if visualization is active for this pair
        actual_points_to_draw_on_pair_img = []
        reprojected_points_to_draw_on_pair_img = []
        error_lines_to_draw_on_pair_img = []
        
        # For calculating the average reprojection error for the pair (from valid segment errors)
        valid_segment_avg_errors_for_pair = []

        if points_3d_prev_all_tracks.shape[0] > 0 and num_estimated_poses > 0:
            # These are the 2D points in frame f_i that correspond to the valid 3D points from f_i-1
            points_2d_curr_actual_for_3d_prev = tracks_for_pnp['points_curr'][valid_mask_3d_prev]

            for seg_id, pose_local_curr_prev in sub_area_poses.items():
                if seg_id == 0: # Skip background or invalid segment ID
                    all_segment_reproj_errors_dict[seg_id] = np.nan
                    continue

                R_local_reproj, t_local_reproj = pose_local_curr_prev # This is P_curr_prev_local_j

                # Identify tracks (among points_2d_curr_actual_for_3d_prev) that fall into this seg_id
                # The segmentation is current_data['sub_area_segments']
                # The points are current_data points: points_2d_curr_actual_for_3d_prev
                
                indices_in_segment = []
                for idx_pt, pt_curr_track in enumerate(points_2d_curr_actual_for_3d_prev):
                    x_coord, y_coord = int(round(pt_curr_track[0])), int(round(pt_curr_track[1]))
                    # Check bounds before accessing segment map
                    if 0 <= y_coord < current_data['sub_area_segments'].shape[0] and \
                       0 <= x_coord < current_data['sub_area_segments'].shape[1]:
                        if current_data['sub_area_segments'][y_coord, x_coord] == seg_id:
                            indices_in_segment.append(idx_pt)
                
                if not indices_in_segment:
                    all_segment_reproj_errors_dict[seg_id] = np.nan
                    continue

                # Actual 2D points in f_i for this segment
                p_i_m_segment = points_2d_curr_actual_for_3d_prev[indices_in_segment]
                # Corresponding 3D points from f_i-1 for this segment
                P_i_minus_1_m_segment = points_3d_prev_all_tracks[indices_in_segment]

                if P_i_minus_1_m_segment.shape[0] == 0:
                    all_segment_reproj_errors_dict[seg_id] = np.nan
                    continue
                
                # Project P_i_minus_1_m_segment into f_i using K_curr and pose_local_curr_prev
                if R_local_reproj is not None and t_local_reproj is not None and K_curr is not None:
                    rvec_local_reproj, _ = cv2.Rodrigues(R_local_reproj)
                    
                    # Ensure P_i_minus_1_m_segment is float32 for projectPoints
                    P_i_minus_1_m_segment_f32 = P_i_minus_1_m_segment.astype(np.float32)

                    projected_Pi_curr_segment, _ = cv2.projectPoints(
                        P_i_minus_1_m_segment_f32, # Object points (3D from f_i-1)
                        rvec_local_reproj,         # Rotation vector of local pose
                        t_local_reproj,            # Translation vector of local pose
                        K_curr,                    # Camera matrix of f_i
                        None                       # Distortion coefficients (None)
                    )
                    # projected_Pi_curr_segment shape is (N, 1, 2), reshape to (N, 2)
                    projected_Pi_curr_segment = projected_Pi_curr_segment.reshape(-1, 2)

                    # Calculate reprojection error for these points (L2 norm)
                    # errors_px_segment = || p_i_m_segment - projected_Pi_curr_segment ||
                    errors_px_segment = np.linalg.norm(p_i_m_segment - projected_Pi_curr_segment, axis=1)
                    
                    avg_err_current_segment = np.mean(errors_px_segment) if errors_px_segment.size > 0 else np.nan
                    all_segment_reproj_errors_dict[seg_id] = avg_err_current_segment
                    
                    if not np.isnan(avg_err_current_segment):
                        valid_segment_avg_errors_for_pair.append(avg_err_current_segment)

                    # If visualization is active for this pair, collect points and lines for drawing
                    if vis_image_reproj_pair is not None:
                        for pt_actual, pt_reprojected in zip(p_i_m_segment, projected_Pi_curr_segment):
                            actual_points_to_draw_on_pair_img.append(pt_actual)
                            reprojected_points_to_draw_on_pair_img.append(pt_reprojected)
                            error_lines_to_draw_on_pair_img.append({'p1': pt_actual, 'p2': pt_reprojected})
                else: # R_local_reproj or t_local_reproj or K_curr is None
                    all_segment_reproj_errors_dict[seg_id] = np.nan
            # End of loop for seg_id in sub_area_poses

            # --- Visualization drawing for the current PAIR (if active) ---
            if vis_image_reproj_pair is not None:
                # 1. (Optional) Color-code segments by their average reprojection error
                overlay_reproj_color = vis_image_reproj_pair.copy()
                if all_segment_reproj_errors_dict:
                    valid_errors_for_colormap = [e for e in all_segment_reproj_errors_dict.values() if np.isfinite(e)]
                    if valid_errors_for_colormap:
                        min_err_val_cm = min(valid_errors_for_colormap)
                        max_err_val_cm = max(valid_errors_for_colormap)
                        if max_err_val_cm == min_err_val_cm : max_err_val_cm = min_err_val_cm + 0.1 # Avoid div by zero

                        for seg_id_vis, avg_err_s_vis in all_segment_reproj_errors_dict.items():
                            if np.isfinite(avg_err_s_vis) and seg_id_vis > 0:
                                norm_err_cm = (avg_err_s_vis - min_err_val_cm) / (max_err_val_cm - min_err_val_cm + 1e-6)
                                norm_err_cm = np.clip(norm_err_cm, 0, 1)
                                
                                # Color: Green (low error) to Red (high error)
                                seg_color_reproj = (int(255 * norm_err_cm * 0.5), int(255 * (1 - norm_err_cm)), 0) # BGR - Reddish for high, greenish for low
                                
                                mask_s_vis = (current_data['sub_area_segments'] == seg_id_vis)
                                overlay_reproj_color[mask_s_vis] = seg_color_reproj
                
                alpha_reproj_overlay = 0.35 # Transparency for segment color overlay
                vis_image_reproj_pair = cv2.addWeighted(overlay_reproj_color, alpha_reproj_overlay, vis_image_reproj_pair, 1 - alpha_reproj_overlay, 0)

                # 2. Draw error lines
                for line_data in error_lines_to_draw_on_pair_img:
                    p1 = tuple(np.round(line_data['p1']).astype(int))
                    p2 = tuple(np.round(line_data['p2']).astype(int))
                    cv2.line(vis_image_reproj_pair, p1, p2, error_line_color, line_thickness_reproj)

                # 3. Draw actual 2D feature points (p_i) - circles
                for pt_a in actual_points_to_draw_on_pair_img:
                    center_a = tuple(np.round(pt_a).astype(int))
                    cv2.circle(vis_image_reproj_pair, center_a, marker_size_circle_reproj, actual_pt_color, -1)

                # 4. Draw reprojected points (Pi) - 'x's
                for pt_r in reprojected_points_to_draw_on_pair_img:
                    x_r, y_r = np.round(pt_r).astype(int)
                    cv2.drawMarker(vis_image_reproj_pair, (x_r, y_r), reproj_pt_color, 
                                   markerType=cv2.MARKER_CROSS, markerSize=marker_size_cross_reproj*2, thickness=line_thickness_reproj)
                
                # 6. Save the visualization for the pair
                prev_basename_vis = prev_data['frame_basename']
                curr_basename_vis = current_data['frame_basename']
                reproj_vis_filename = f"reproj_err_pair_{prev_basename_vis}_to_{curr_basename_vis}.png"
                reproj_vis_path = os.path.join(reproj_vis_output_dir,args.video_name,reproj_vis_filename) # Use new dir
                try:
                    cv2.imwrite(reproj_vis_path, vis_image_reproj_pair)
                    logging.info(f"Saved reprojection error visualization: {reproj_vis_path}")
                except Exception as e_write_reproj:
                    logging.error(f"Failed to write reprojection visualization {reproj_vis_path}: {e_write_reproj}")
            # End of visualization drawing for the pair
        else:
            if points_3d_prev_all_tracks.shape[0] == 0: logging.debug("No valid 3D points for reprojection.")
            if num_estimated_poses == 0: logging.debug("No estimated poses for reprojection.")

        all_results['frame_reprojection_errors'].append(avg_reproj_error_pair_val) # Store the calculated pair average
        log_level_reproj = logging.INFO if np.isfinite(avg_reproj_error_pair_val) else logging.WARNING
        logging.log(log_level_reproj, f"Pair ({i-1}, {i}): Avg Local Reprojection Error: {avg_reproj_error_pair_val:.4f} pixels")
        # --- END: 6. Quantify AND Visualize Reprojection Error ---

        # 7. Quantify: Geometric Depth Consistency (Unchanged Logic, uses relative VGGT pose for warping)
        depth_error = np.nan
        pose_for_warp = (R_vggt_rel, t_vggt_rel) if R_vggt_rel is not None else None

        if pose_for_warp:
            logging.debug(f"Pair ({i-1}, {i}): Warping depth using relative VGGT pose.")
            depth_warped = warp_depth(prev_data['depth'], K_prev, K_curr, pose_for_warp, current_data['depth'].shape, None)

            static_mask_curr = current_data['static_mask']
            valid_comparison_mask = (depth_warped > 1e-5) & (current_data['depth'] > 1e-5) & static_mask_curr

            if np.any(valid_comparison_mask):
                depth_diff = np.abs(depth_warped[valid_comparison_mask] - current_data['depth'][valid_comparison_mask])
                depth_error = np.mean(depth_diff)
                logging.debug(f"Pair ({i-1}, {i}): Compared {np.sum(valid_comparison_mask)} pixels for depth consistency.")
            else:
                logging.warning(f"Pair ({i-1}, {i}): No valid overlapping pixels found for depth consistency check.")
                depth_error = np.nan

            # DEBUG Visualization: Warped Depth (Unchanged)
            if args.debug_vis_interval > 0 and i % args.debug_vis_interval == 0 and np.any(valid_comparison_mask):
                depth_warped_vis = cv2.normalize(depth_warped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_curr_vis = cv2.normalize(current_data['depth'], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                diff_vis = cv2.normalize(np.abs(depth_warped - current_data['depth']), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                diff_vis[~valid_comparison_mask] = 0
                diff_colormap = cv2.applyColorMap(diff_vis, cv2.COLORMAP_MAGMA)
                diff_colormap[~valid_comparison_mask] = [0, 0, 0]
                cv2.imwrite(os.path.join(warp_debug_dir, f"{current_data['frame_basename']}_0_depth_curr.png"), depth_curr_vis)
                cv2.imwrite(os.path.join(warp_debug_dir, f"{current_data['frame_basename']}_1_depth_warped_from_{prev_data['frame_basename']}.png"), depth_warped_vis)
                cv2.imwrite(os.path.join(warp_debug_dir, f"{current_data['frame_basename']}_2_depth_diff.png"), diff_colormap)
                cv2.imwrite(os.path.join(warp_debug_dir, f"{current_data['frame_basename']}_3_comparison_mask.png"), valid_comparison_mask.astype(np.uint8)*255)
        else:
             logging.warning(f"Pair ({i-1}, {i}): No valid pose (VGGT Relative) available for depth warping. Skipping depth consistency.")


        all_results['frame_depth_consistency_error'].append(depth_error)
        log_level_depth = logging.INFO if np.isfinite(depth_error) else logging.WARNING
        logging.log(log_level_depth, f"Pair ({i-1}, {i}): Geometric Depth Consistency Error: {depth_error:.4f}")


        # Update previous data state
        prev_data = current_data

    # --- Final Score Calculation (MODIFIED based on new variance metrics) ---
    logging.info(f"Finished processing. Total pairs processed successfully: {processed_pairs_count}/{max(0, num_frames - 1)}")
    if processed_pairs_count == 0:
         logging.error("No frame pairs were successfully processed. Cannot calculate final scores.")
         return None

    # Extract results using the new keys
    local_rot_vars   = all_results['frame_local_consistency_rot_var']
    local_trans_vars = all_results['frame_local_consistency_trans_var']
    reproj           = all_results['frame_reprojection_errors']
    global_rot_vars  = all_results['frame_global_consistency_rot_var'] # Changed key
    global_trans_vars= all_results['frame_global_consistency_trans_var']# Changed key
    depth_err        = all_results['frame_depth_consistency_error']

    metrics = {
        'local_rot_var':   local_rot_vars,
        'local_trans_var': local_trans_vars,
        'reproj_error':      reproj,
        'global_rot_var':    global_rot_vars, # Changed key
        'global_trans_var':  global_trans_vars,# Changed key
        'depth_error':       depth_err,
    }

    # Plotting (Unchanged logic, uses new metric names)
    sns.set_theme(style="whitegrid")
    for name, values in metrics.items():
        # Filter out potential NaNs for plotting if desired, or let seaborn handle them
        valid_indices = [j for j, v in enumerate(values) if np.isfinite(v)]
        if not valid_indices:
            logging.warning(f"No valid data points to plot for metric: {name}")
            continue
        valid_values = [values[j] for j in valid_indices]

        plt.figure(figsize=(10, 4)) # Wider figure
        sns.lineplot(x=valid_indices, y=valid_values, marker="o", legend=False) # Use valid indices for x-axis
        # sns.scatterplot(x=valid_indices, y=valid_values, color="red", s=50) # Add scatter points
        plt.title(f'{name} over Processed Pairs')
        plt.xlabel('Processed Pair Index (Frame i-1 -> i)') # Clarify x-axis
        plt.ylabel(name)
        plt.xlim(left=-1, right=len(values)) # Ensure x-axis shows full range of pairs processed/skipped
        plt.tight_layout() # Adjust layout

        out_path = os.path.join(plot_debug_dir, f'{name}_over_pairs.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved plot: {out_path}")
    
    # 首先解析 summary.txt（路径可从 args 或硬编码）
    summary_txt = args.summary_txt_path  # 你需要在 args 中添加这个字段，指向 composite_3d_consistency_summary_pca_weights.txt
    norm_params, pca_weights = parse_summary_file(summary_txt)

    frame_pair_scores = []
    # 对每一个处理过的“帧对索引”i （0 对应帧 1 vs 0，…）
    num_pairs = len(local_rot_vars)
    for idx in range(num_pairs):
        # 原始值字典
        raw = {
            'average_local_rotation_variance':   local_rot_vars[idx],
            'average_local_translation_variance':local_trans_vars[idx],
            'average_reprojection_error':        reproj[idx],
            'average_global_rotation_variance':  global_rot_vars[idx],
            'average_global_translation_variance': global_trans_vars[idx],
            'average_depth_consistency_error':   depth_err[idx],
        }

        # 计算 m'' 和 S_3DC
        mpp = {}
        num, denom = 0.0, 0.0
        for key in METRIC_KEYS:
            val = raw.get(key, None)
            params = norm_params.get(key, {})
            μ = params.get('mu_raw', None)
            σ = params.get('sigma_raw', None)
            zmin = params.get('z_score_min', None)
            zmax = params.get('z_score_max', None)
            w = pca_weights.get(key, 0.0)

            if val is None or μ is None or σ is None or zmin is None or zmax is None:
                mpp[key] = np.nan
                continue

            # Z-score
            z = 0.0 if σ == 0 else (val - μ) / σ
            # Min-Max 归一化到 [0,1]
            span = zmax - zmin
            m_val = 0.0 if span == 0 else (z - zmin) / span
            mpp[key] = m_val

            if not np.isnan(m_val) and w > 0:
                num   += w * m_val
                denom += w

        S3DC = num / denom if denom > 0 else np.nan
        frame_pair_scores.append(S3DC)

    # 存入 all_results，或打印／写文件都可以
    all_results['frame_pair_S3DC_scores'] = frame_pair_scores
    logging.info("每个帧对的 S_3DC 分数计算完毕。")
    return all_results


# --- Argument Parsing and Main Execution (MODIFIED Args) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D consistency using local variance, global variance vs VGGT, reprojection, and depth.")

    # Input Data Paths (Unchanged)
    parser.add_argument("--frames_dir", type=str, required=True, help="Base directory containing video frame images (subdirs per video).")
    parser.add_argument("--depth_dir", type=str, required=True, help="Base directory containing depth maps (NPZ with 'depths' key, subdirs per video).")
    parser.add_argument("--mos_dir", type=str, required=True, help="Base directory containing Moving Object Segmentation masks (subdirs per video).")
    parser.add_argument("--tracks_dir", type=str, required=True, help="Base directory containing pairwise tracking files (subdirs per video).")
    parser.add_argument("--semantics_dir", type=str, default=None, help="(Optional) Base directory containing semantic segmentation masks (subdirs per video).")
    parser.add_argument("--output_json", type=str, default="consistency_results_globalvar.json", help="Path to save the final results JSON file (will be prefixed with video name).")
    parser.add_argument("--video_name", type=str, required=True, help="Name of the video subdirectory to process.")

    # # Configuration Parameters (Unchanged, added SLIC args)
    # parser.add_argument("--depth_scale", type=float, default=1000.0, help="Factor to divide raw integer depth values by. Use 1.0 or 0 if depth is already metric or relative float.")
    # # parser.add_argument("--segmentation_method", type=str, default="depth_aware_slic", choices=["depth_aware_slic"], help="Segmentation method (only depth-aware SLIC currently implemented).") # Simplified choice
    # parser.add_argument("--static_semantic_labels", type=int, nargs='+', default=[1, 2, 5], help="List of semantic labels considered static (used if semantics_dir provided).")
    # parser.add_argument("--n_slic_segments", type=int, default=20, help="Approximate number of SLIC superpixels for Depth-Aware SLIC.") # Increased default
    # parser.add_argument("--slic_compactness", type=float, default=5.0, help="Compactness parameter for SLIC.")
    # parser.add_argument("--depth_aware_slic_weight", type=float, default=5.0, help="Weight for depth channel in depth-aware SLIC.")
    # parser.add_argument("--slic_sigma", type=float, default=1.0, help="Sigma for Gaussian smoothing before SLIC (0 to disable).")
    # parser.add_argument("--min_segment_size", type=int, default=200, help="Minimum pixel count for a segment to be considered.") # Reduced default
    # parser.add_argument("--pnp_reprojection_error", type=float, default=5.0, help="RANSAC reprojection error threshold for PnP (pixels).") # Reduced default
    # parser.add_argument("--min_track_confidence", type=float, default=0.5, help="Minimum confidence score for a track point to be used.")

    # Configuration Parameters
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Factor to divide raw depth values by (e.g., 1000.0 for mm to meters). Set <= 0 to use raw values.")
    parser.add_argument("--segmentation_method", type=str, default="depth", choices=["slic", "depth", "semantic","grid"], help="Method for segmenting static background.")
    parser.add_argument("--static_semantic_labels", type=int, nargs='+', default=[1, 2, 5], help="List of semantic labels considered static (used if segmentation_method is 'semantic').")
    parser.add_argument("--n_slic_segments", type=int, default=20, help="Approximate number of SLIC superpixels.")
    parser.add_argument("--slic_compactness", type=float, default=10.0, help="Compactness parameter for SLIC.")
    parser.add_argument("--n_depth_clusters", type=int, default=16, help="Number of clusters for depth-based segmentation.")
    parser.add_argument("--min_segment_size", type=int, default=200, help="Minimum pixel count for a segment to be considered.")
    parser.add_argument("--pnp_reprojection_error", type=float, default=8.0, help="RANSAC reprojection error threshold for PnP (pixels).")
    parser.add_argument("--min_track_confidence", type=float, default=0.5, help="Minimum confidence score (from tracks_file[:,:,3]) for a track point to be used (applied to current frame). Default 0.0 uses all visible.")


    # Normalization and Scoring Parameters (MODIFIED global params)
    parser.add_argument("--norm_max_local_rot_var", type=float, default=10.0, help="Max expected LOCAL rotation variance (deg^2) for normalization.")
    parser.add_argument("--norm_max_local_trans_var", type=float, default=0.5, help="Max expected LOCAL normalized translation variance for normalization.")
    parser.add_argument("--norm_max_reproj_err", type=float, default=5.0, help="Max expected reprojection error (pixels) for normalization.")
    parser.add_argument("--norm_max_global_rot_var", type=float, default=25.0, help="Max expected GLOBAL rotation variance vs VGGT (deg^2) for normalization.") # CHANGED NAME & default
    parser.add_argument("--norm_max_global_trans_var", type=float, default=1.0, help="Max expected GLOBAL normalized translation variance vs VGGT for normalization.") # CHANGED NAME & default
    parser.add_argument("--norm_max_depth_err", type=float, default=0.5, help="Max expected depth consistency error (in depth units after scaling) for normalization.")
    parser.add_argument("--final_score_weights", type=float, nargs=4, default=[0.1, 0.1, 0.4, 0.4], metavar=('W_LCON', 'W_REPR', 'W_GCON', 'W_DEPT'), help="Weights for [LocalCons, Reproj, GlobalCons, DepthCons] in final score.") # Updated help text

    # VGGT Configuration (Unchanged)
    parser.add_argument("--vggt_model_name", type=str, default="facebook/VGGT-1B", help="Name of the VGGT model to load from Hugging Face.")

    # Debugging (Unchanged)
    parser.add_argument("--debug_vis_interval", type=int, default=0, help="Interval (in frames) for saving debug visualizations. 0 to disable.")
    
    parser.add_argument("--summary_txt_path", type=str, default="3d_consistency/result/fast_new/composite_3d_consistency_summary_pca_weights.txt", help="Path to the summary file containing normalization parameters and PCA weights.")

    args = parser.parse_args()

    # Validate weights length and sum (Unchanged)
    if len(args.final_score_weights) != 4:
        raise ValueError("final_score_weights must have exactly 4 values.")
    if sum(args.final_score_weights) <= 1e-6:
        logging.warning("Final score weights sum to zero or less. Using equal weights [0.25]*4 instead.")
        args.final_score_weights = [0.25, 0.25, 0.25, 0.25]

    # --- Execute for the specified video ---
    video_name = args.video_name
    logging.info(f"===== Evaluating Video: {video_name} =====")

    # Construct full paths based on video name (Unchanged)
    current_frames_dir = os.path.join(args.frames_dir, video_name)
    current_depth_dir = os.path.join(args.depth_dir, video_name)
    current_mos_dir = os.path.join(args.mos_dir, video_name)
    current_tracks_dir = os.path.join(args.tracks_dir, video_name)
    current_semantics_dir = os.path.join(args.semantics_dir, video_name) if args.semantics_dir else None

    # Check if essential directories exist (Unchanged)
    if not os.path.isdir(current_frames_dir): logging.error(f"Frames directory not found: {current_frames_dir}"); exit()
    if not os.path.isdir(current_depth_dir): logging.error(f"Depth directory not found: {current_depth_dir}"); exit()
    if not os.path.isdir(current_mos_dir): current_mos_dir = None
    if not os.path.isdir(current_tracks_dir): logging.error(f"Tracks directory not found: {current_tracks_dir}"); exit()
    if args.semantics_dir and current_semantics_dir and not os.path.isdir(current_semantics_dir): logging.warning(f"Semantics directory specified but not found: {current_semantics_dir}")


    # Create a copy of args to modify paths for the function call (Unchanged)
    video_args = argparse.Namespace(**vars(args))
    video_args.frames_dir = current_frames_dir
    video_args.depth_dir = current_depth_dir
    video_args.mos_dir = current_mos_dir
    video_args.tracks_dir = current_tracks_dir
    video_args.semantics_dir = current_semantics_dir

    try:
        results = evaluate_3d_consistency(video_args) # Pass the modified args
        video_name_top = video_name.split('_')[0] if '_' in video_name else video_name
        os.makedirs(os.path.join('vis','score',video_name_top), exist_ok=True)
        args.output_json = os.path.join('vis','score',video_name_top,f"{video_name}.txt")
        if results:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                f.write("Pair_Index\tS_3DC_score\n")
                for idx, s in enumerate(results['frame_pair_S3DC_scores']):
                    score_str = f"{s:.6f}" if not np.isnan(s) else "N/A"
                    f.write(f"{idx}\t{score_str}\n")
                f.write(f"Final_S3DC_score\t{np.nanmean(results['frame_pair_S3DC_scores']):.6f}\n")
            logging.info(f"Saved per-pair S_3DC scores → {args.output_json}")


    except Exception as e:
        logging.exception(f"An critical error occurred during the evaluation pipeline for video {video_name}:")

    logging.info(f"===== Finished Evaluating Video: {video_name} =====")