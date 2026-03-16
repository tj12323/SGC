import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.segmentation import slic
from skimage.util import img_as_float
import logging
import os
import argparse
import json
import datetime
import math 

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from skimage import color 
import seaborn as sns
import matplotlib.pyplot as plt


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

def load_frame_data(frame_path, depth_input, mos_path, semantic_path=None, depth_scale=1.0):
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
    elif depth_scale <= 0:
         pass 

    min_d, max_d = (np.min(depth[depth > 0]), np.max(depth)) if np.any(depth > 0) else (0, 0)

    img_h, img_w = frame.shape[:2]
    img_diagonal = np.sqrt(img_h**2 + img_w**2)
    
    return frame, depth, mos_mask, semantic_mask, img_diagonal  

def apply_masks(frame, depth, mos_mask, semantic_mask=None, static_semantic_labels=None):
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
    static_depth[~static_mask] = 0 
    return static_frame, static_depth, static_mask

def get_3d_points(pixels_2d, depth_map, K): 
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
    valid_depth_mask = Z > 1e-5 
    if not np.any(valid_depth_mask): return np.empty((0, 3)), np.empty((0, 2)), np.array([], dtype=bool)
    u_valid, v_valid, Z_valid = u[valid_depth_mask], v[valid_depth_mask], Z[valid_depth_mask]
    X_valid = (u_valid - cx) * Z_valid / fx
    Y_valid = (v_valid - cy) * Z_valid / fy
    points_3d = np.vstack((X_valid, Y_valid, Z_valid)).T
    pixels_2d_valid = pixels_2d[valid_depth_mask]
    return points_3d, pixels_2d_valid, valid_depth_mask

def orthogonalize_rotation_matrix(R_in):
    if R_in is None or not isinstance(R_in, np.ndarray) or R_in.shape != (3, 3):
        logging.warning("Invalid input provided for orthogonalization.")
        return None
    try:
        U, _, Vt = np.linalg.svd(R_in)
        R_ortho = U @ Vt

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
    if R1 is None or R2 is None or not isinstance(R1, np.ndarray) or not isinstance(R2, np.ndarray) or R1.shape != (3,3) or R2.shape != (3,3): return np.inf

    tolerance = 1e-2
    det1_ok = abs(np.linalg.det(R1) - 1.0) <= tolerance
    det2_ok = abs(np.linalg.det(R2) - 1.0) <= tolerance

    if not det1_ok or not det2_ok:
        return np.inf
    try:
        R_diff = R1.T @ R2
        trace = np.trace(R_diff)
        angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        return np.degrees(angle_rad)
    except ValueError as e:
        logging.error(f"Error in angular_distance: {e}")
        return np.inf

def normalize_translation(t):
    if t is None or not isinstance(t, np.ndarray): return None
    t_flat = t.flatten()
    norm = np.linalg.norm(t_flat)
    if norm < 1e-9: return np.zeros_like(t_flat) 
    return t_flat / norm

def segment_static_background(static_frame, static_depth, static_mask,**kwargs):
    """Segments the static background using depth clustering only."""
    h, w = static_frame.shape[:2]
    sub_area_segments = np.zeros((h, w), dtype=np.int32)
    num_segments = 0
    min_size_filter = kwargs.get('min_size_filter', 100)

    if not np.any(static_mask):
        logging.warning("Static mask is empty, no segments generated.")
        return sub_area_segments, num_segments

    logging.info("Segmenting using GPU-accelerated depth clustering (cuML).")
    n_clusters = kwargs.get('n_depth_clusters', 10)
    valid_depth_pixels = static_depth[static_mask]
    pixel_coords = np.argwhere(static_mask)

    if valid_depth_pixels.shape[0] < n_clusters * 2:
        logging.warning(
            f"Not enough valid depth pixels ({valid_depth_pixels.shape[0]}) for {n_clusters} clusters."
        )
        return sub_area_segments, num_segments

    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans

    gpu_valid_depth_pixels = cp.asarray(
        valid_depth_pixels.reshape(-1, 1),
        dtype=cp.float32
    )

    kmeans_model_gpu = cuKMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init=10,
    )
    logging.info(f"Using cuML KMeans with n_clusters={n_clusters}")

    gpu_labels = kmeans_model_gpu.fit_predict(gpu_valid_depth_pixels)
    labels = cp.asnumpy(gpu_labels)

    temp_segments = np.zeros((h, w), dtype=np.int32)
    temp_segments[pixel_coords[:, 0], pixel_coords[:, 1]] = labels + 1

    current_id = 1
    unique_labels = np.unique(temp_segments)

    for unique_label_val in unique_labels:
        if unique_label_val == 0:
            continue

        segment_mask = (temp_segments == unique_label_val)
        if np.sum(segment_mask) >= min_size_filter:
            sub_area_segments[segment_mask] = current_id
            current_id += 1

    num_segments = current_id - 1
    logging.info(f"Generated {num_segments} sub-area segments using method 'depth'.")

    return sub_area_segments, num_segments

def estimate_pose_per_subarea(frame_idx, sub_area_segments, num_segments,
                              tracks_for_pair, depth_prev, K_prev, K_curr, 
                              distCoeffs=None, pnp_reprojection_error=8.0):

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
    if points_3d_prev.shape[0] == 0: 
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

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_points_3d, img_points_2d, K_curr, distCoeffs=distCoeffs,
                iterationsCount=100, reprojectionError=pnp_reprojection_error, confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE 
            )

            min_inliers = max(4, int(num_points_in_seg * 0.3)) 

            if success and inliers is not None and len(inliers) >= min_inliers:
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                if abs(np.linalg.det(rotation_matrix) - 1.0) < 1e-3:
                    sub_area_poses[seg_id] = (rotation_matrix, tvec.flatten())
                    successful_pnp_count += 1
                else:
                    logging.debug(f"Frame {frame_idx}, Seg {seg_id}: PnP RANSAC resulted in invalid rotation matrix (det={np.linalg.det(rotation_matrix):.3f}). Inliers: {len(inliers)}/{num_points_in_seg}")
        except cv2.error as e:
             if "points" in str(e).lower() or "size" in str(e).lower():
                 logging.warning(f"Frame {frame_idx}, Seg {seg_id}: OpenCV PnP input error: {e}. Points: {num_points_in_seg}, K_curr: {K_curr is not None}")
             else:
                 logging.error(f"Frame {frame_idx}, Seg {seg_id}: OpenCV Error during PnP: {e}")
        except Exception as e:
             logging.error(f"Frame {frame_idx}, Seg {seg_id}: Unexpected Error during PnP: {e}", exc_info=True)

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

    k = (q_upper - q3) / iqr
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return a[(a >= lower) & (a <= upper)]

def assess_local_consistency(sub_area_poses):
    if not sub_area_poses:
        return np.nan, np.nan, None, None

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

    try:
        rvecs = [R.from_matrix(m).as_rotvec() * 180.0/np.pi for m in rotations]
        mean_rvec = np.mean(rvecs, axis=0)
        mean_rotation = R.from_rotvec(mean_rvec * np.pi/180.0).as_matrix()

        rot_d2 = [angular_distance(m, mean_rotation)**2 for m in rotations]
        rot_d2 = [d for d in rot_d2 if np.isfinite(d)]

        trimmed = trim_outliers_iqr(rot_d2, cov=98.0)
        if trimmed.size > 0:
            rot_variance = float(trimmed.mean())
        else:
            rot_variance = float(np.mean(rot_d2))  
        rot_variance = float(np.mean(rot_d2))

    except Exception as e:
        logging.error(f"Error calculating rotation variance: {e}")

    try:

        norm_ts = [t for t in translations]
        norm_ts = [t for t in norm_ts if t is not None]

        if len(norm_ts) >= 2:
            mean_t = np.mean(norm_ts, axis=0)
            d2 = [np.linalg.norm(t - mean_t)**2 for t in norm_ts]
            trans_variance = float(np.mean(d2))
        else:
            trans_variance = 0.0

        mean_translation_orig = np.mean(translations, axis=0)

    except Exception as e:
        logging.error(f"Error calculating translation variance: {e}")

    return rot_variance, trans_variance, mean_rotation, mean_translation_orig

def calculate_reprojection_error(points_3d_prev, points_2d_curr_matched, pose, K_curr, 
                                 sub_area_segments, seg_id, distCoeffs=None):

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
    if num_points == 0: return 0.0 

    obj_points = points_3d_prev[mask_seg]
    img_points_actual = points_2d_curr_matched[mask_seg]

    try:
        rvec, _ = cv2.Rodrigues(R_sub)
        obj_points_cv = obj_points.reshape(-1, 1, 3).astype(np.float32)
        img_points_projected, _ = cv2.projectPoints(obj_points_cv, rvec, t_sub, K_curr, distCoeffs=distCoeffs)

        if img_points_projected is None: return np.nan
        img_points_projected = img_points_projected.squeeze(axis=1)

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

        if not isinstance(R_wc_prev, np.ndarray) or R_wc_prev.shape != (3,3) or \
           not isinstance(R_wc_curr, np.ndarray) or R_wc_curr.shape != (3,3):
             logging.error("Invalid Rotation matrix in relative pose calculation.")
             return None, None

        if not isinstance(t_wc_prev, np.ndarray) or not isinstance(t_wc_curr, np.ndarray):
             logging.error("Invalid Translation vector in relative pose calculation.")
             return None, None

        t_wc_prev = t_wc_prev.reshape(3, 1)
        t_wc_curr = t_wc_curr.reshape(3, 1)

        R_rel = R_wc_curr @ R_wc_prev.T
        t_rel = t_wc_curr - (R_rel @ t_wc_prev)

        R_rel_ortho = orthogonalize_rotation_matrix(R_rel)
        if R_rel_ortho is None:
            logging.warning("Orthogonalization failed for calculated relative rotation. Returning non-orthogonalized.")
            return None, None
        return R_rel_ortho, t_rel.flatten()

    except np.linalg.LinAlgError as e:
         logging.error(f"Linear algebra error during relative pose calculation: {e}", exc_info=True)
         return None, None
    except Exception as e:
        logging.error(f"Unexpected error calculating relative pose: {e}", exc_info=True)
        return None, None

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
    norm_t_ref = t_ref

    if norm_t_ref is None:
        logging.warning("Reference translation vector normalization failed.")
        pass 

    valid_poses = {k: v for k, v in sub_area_poses.items() if v is not None and v[0] is not None and v[1] is not None}
    if not valid_poses:
        logging.debug("No valid sub-area poses found for variance calculation vs reference.")
        return np.nan, np.nan 

    rot_distances_sq = []
    trans_distances_sq = []

    for seg_id, (R_sub, t_sub) in valid_poses.items():
        try:
            dist_rad = angular_distance(R_sub, R_ref)
            if np.isfinite(dist_rad):
                rot_distances_sq.append(dist_rad**2)
        except Exception as e:
            logging.error(f"Error calculating angular distance for seg {seg_id} vs reference: {e}")


        if norm_t_ref is not None:
            try:
                norm_t_sub = t_sub
                if norm_t_sub is not None:
                    dist_sq = np.linalg.norm(norm_t_sub - norm_t_ref)**2
                    trans_distances_sq.append(dist_sq)
            except Exception as e:
                logging.error(f"Error calculating translation distance for seg {seg_id} vs reference: {e}")

    trimmed = trim_outliers_iqr(rot_distances_sq, cov=98.0)
    rot_variance_vs_ref = np.mean(trimmed)

    trans_variance_vs_ref = np.mean(trans_distances_sq)
    if norm_t_ref is None and not trans_distances_sq:
        trans_variance_vs_ref = np.nan

    return rot_variance_vs_ref, trans_variance_vs_ref

def warp_depth(depth_prev, K_prev, K_curr, pose_curr_from_prev, depth_curr_shape): 
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

    vs_prev, us_prev = np.meshgrid(np.arange(h_prev), np.arange(w_prev), indexing='ij')
    pixels_prev = np.stack((us_prev.flatten(), vs_prev.flatten()), axis=-1)
    depth_values_prev = depth_prev.flatten()

    valid_depth_mask = depth_values_prev > 1e-5
    pixels_prev_valid = pixels_prev[valid_depth_mask]
    depth_values_prev_valid = depth_values_prev[valid_depth_mask]
    us_prev_valid = pixels_prev_valid[:, 0]
    vs_prev_valid = pixels_prev_valid[:, 1]

    X_prev_valid = (us_prev_valid - cx_prev) * depth_values_prev_valid / fx_prev
    Y_prev_valid = (vs_prev_valid - cy_prev) * depth_values_prev_valid / fy_prev
    P_prev_cam_valid = np.stack((X_prev_valid, Y_prev_valid, depth_values_prev_valid), axis=-1) 

    P_curr_cam_valid = (R_curr_prev @ P_prev_cam_valid.T + t_curr_prev).T 
    X_curr_valid = P_curr_cam_valid[:, 0]
    Y_curr_valid = P_curr_cam_valid[:, 1]
    Z_curr_valid = P_curr_cam_valid[:, 2]

    valid_z_curr_mask = Z_curr_valid > 1e-5
    X_curr_proj = X_curr_valid[valid_z_curr_mask]
    Y_curr_proj = Y_curr_valid[valid_z_curr_mask]
    Z_curr_proj = Z_curr_valid[valid_z_curr_mask]

    if len(Z_curr_proj) == 0:
        logging.debug("No points remained after warping and Z>0 check.")
        return depth_warped 

    us_curr_proj = (X_curr_proj * fx_curr / Z_curr_proj) + cx_curr
    vs_curr_proj = (Y_curr_proj * fy_curr / Z_curr_proj) + cy_curr

    us_idx = np.round(us_curr_proj).astype(int)
    vs_idx = np.round(vs_curr_proj).astype(int)

    valid_bounds_mask = (us_idx >= 0) & (us_idx < w_curr) & (vs_idx >= 0) & (vs_idx < h_curr)

    us_final = us_idx[valid_bounds_mask]
    vs_final = vs_idx[valid_bounds_mask]
    Z_final = Z_curr_proj[valid_bounds_mask]

    if len(Z_final) == 0:
        logging.debug("No points remained after boundary check.")
        return depth_warped 

    sort_indices = np.argsort(Z_final)
    us_final_sorted = us_final[sort_indices]
    vs_final_sorted = vs_final[sort_indices]
    Z_final_sorted = Z_final[sort_indices]
    depth_warped[vs_final_sorted, us_final_sorted] = Z_final_sorted
    return depth_warped

def evaluate_3d_consistency(args):

    debug_root = "check"
    static_debug_dir = os.path.join(debug_root, "static")
    segments_debug_dir = os.path.join(debug_root, "segments")
    calib_debug_dir = os.path.join(debug_root, "calib_points")
    warp_debug_dir = os.path.join(debug_root, "warped_depth")
    plot_debug_dir = os.path.join(debug_root, "frames_plots", args.video_name)
    for d in (static_debug_dir, segments_debug_dir, calib_debug_dir, warp_debug_dir, plot_debug_dir):
        os.makedirs(d, exist_ok=True)

    try:
        frame_files = sorted([f for f in os.listdir(args.frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

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

    if not os.path.isdir(args.tracks_dir):
        logging.error(f"Tracking directory not found: {args.tracks_dir}")
        return None
    logging.info(f"Using track directory: {args.tracks_dir}")

    logging.info("--- Initializing VGGT Model ---")
    frame_files_basenames = sorted([f for f in os.listdir(args.frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    frame_files_fullpaths = [os.path.join(args.frames_dir, f) for f in frame_files_basenames]
    vggt_poses = {} 
    vggt_intrinsics = {} 
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
        images_tensor_5d = images_tensor_4d.unsqueeze(0) 
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

        expected_enc_shape = (1, num_frames, 9)
        if pose_encodings.shape != expected_enc_shape:
             raise RuntimeError(f"VGGT pose encoding has unexpected shape {pose_encodings.shape}. Expected {expected_enc_shape}.")

        img_shape_hw = (images_tensor_5d.shape[-2], images_tensor_5d.shape[-1])

        try:
            extrinsic_matrices, intrinsic_matrices = pose_encoding_to_extri_intri(
                pose_encodings, 
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

            R_wc_batch = extrinsic_matrices[0, :, :3, :3]  
            t_wc_batch = extrinsic_matrices[0, :, :3, 3]   
            K_batch = intrinsic_matrices[0]                

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

                 R_ortho = orthogonalize_rotation_matrix(R_np[i])
                 if R_ortho is None:
                     logging.warning(f"Failed to orthogonalize VGGT R_wc for frame {i}. Storing original.")
                     R_ortho = R_np[i] 
                 vggt_poses[i] = (R_ortho, t_np[i]) 
                 vggt_intrinsics[i] = K_np[i]

                 if i < 3: 
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

    all_results = {
        'frame_local_consistency_rot_var': [],  
        'frame_local_consistency_trans_var': [],
        'frame_reprojection_errors': [],        
        'frame_global_consistency_rot_var': [], 
        'frame_global_consistency_trans_var':[],
        'frame_depth_consistency_error': [],    
        'mean_local_relative_poses': [],        
        'vggt_relative_poses': [],              
        'num_valid_tracks_per_pair': [],        
        'num_pnp_poses_per_pair': [],           
        'skipped_pairs': 0,
    }

    prev_data = {}
    processed_pairs_count = 0
    for i in range(num_frames):
        frame_basename_no_ext = os.path.splitext(frame_files[i])[0]
        frame_filename = frame_files[i]
        logging.info(f"--- Processing Frame {i}/{num_frames-1} ({frame_filename}) ---")


        frame_path = os.path.join(args.frames_dir, frame_filename)
        depth_slice = depths_all[i]
        if args.mos_dir is not None:
            mos_path = os.path.join(args.mos_dir, frame_basename_no_ext + ".png")
            if not os.path.exists(mos_path): logging.warning(f"MOS file not found: {mos_path}"); mos_path = None 
        else:
            mos_path = None
        
        semantic_path = None
        if args.semantics_dir and i < len(semantic_files):

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

        if i not in vggt_intrinsics:
            logging.error(f"Skipping frame {i} ({frame_files_basenames[i]}): Missing VGGT intrinsics.")
            prev_data = {}
            if i > 0: 
                all_results['skipped_pairs'] += 1
                all_results['frame_local_consistency_rot_var'].append(np.nan)
                all_results['frame_local_consistency_trans_var'].append(np.nan)
                all_results['frame_reprojection_errors'].append(np.nan)
                all_results['frame_global_consistency_rot_var'].append(np.nan) 
                all_results['frame_global_consistency_trans_var'].append(np.nan)
                all_results['frame_depth_consistency_error'].append(np.nan)
                all_results['mean_local_relative_poses'].append(None)
                all_results['vggt_relative_poses'].append(None)
                all_results['num_valid_tracks_per_pair'].append(0)
                all_results['num_pnp_poses_per_pair'].append(0)
            continue

        K_curr = vggt_intrinsics[i]
        try:
            if not frame_path:
                 raise IOError(f"Missing required files for frame {i} (frame/mos).")

            frame, depth, mos_mask, semantic_mask, img_diagonal = load_frame_data(
                frame_path, depth_slice, mos_path, semantic_path, args.depth_scale
            )          
        except IOError as e:
            logging.error(f"Skipping frame {i} due to loading error: {e}")
            prev_data = {}
            if i > 0: 
                all_results['skipped_pairs'] += 1
                all_results['frame_local_consistency_rot_var'].append(np.nan)
                all_results['frame_local_consistency_trans_var'].append(np.nan)
                all_results['frame_reprojection_errors'].append(np.nan)
                all_results['frame_global_consistency_rot_var'].append(np.nan) 
                all_results['frame_global_consistency_trans_var'].append(np.nan)
                all_results['frame_depth_consistency_error'].append(np.nan)
                all_results['mean_local_relative_poses'].append(None)
                all_results['vggt_relative_poses'].append(None)
                all_results['num_valid_tracks_per_pair'].append(0)
                all_results['num_pnp_poses_per_pair'].append(0)
            continue
        except Exception as e:
             logging.error(f"Unexpected error loading data for frame {i}: {e}", exc_info=True)
             prev_data = {}
             if i > 0: 
                all_results['skipped_pairs'] += 1
                all_results['frame_local_consistency_rot_var'].append(np.nan)
                all_results['frame_local_consistency_trans_var'].append(np.nan)
                all_results['frame_reprojection_errors'].append(np.nan)
                all_results['frame_global_consistency_rot_var'].append(np.nan) 
                all_results['frame_global_consistency_trans_var'].append(np.nan)
                all_results['frame_depth_consistency_error'].append(np.nan)
                all_results['mean_local_relative_poses'].append(None)
                all_results['vggt_relative_poses'].append(None)
                all_results['num_valid_tracks_per_pair'].append(0)
                all_results['num_pnp_poses_per_pair'].append(0)
             continue

        static_frame, static_depth, static_mask = apply_masks(
            frame, depth, mos_mask, semantic_mask, args.static_semantic_labels
        )

        if args.debug_vis_interval > 0 and i % args.debug_vis_interval == 0:
             cv2.imwrite(os.path.join(static_debug_dir, f"{frame_basename_no_ext}_static_frame.png"), static_frame)
             depth_vis = cv2.normalize(static_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
             cv2.imwrite(os.path.join(static_debug_dir, f"{frame_basename_no_ext}_static_depth.png"), depth_vis)

        sub_area_segments, num_segments = segment_static_background(
            static_frame, static_depth, static_mask,
            semantic_mask=semantic_mask, static_labels=args.static_semantic_labels,
            n_depth_clusters=args.n_depth_clusters,
            n_slic_segments=args.n_slic_segments, compactness=args.slic_compactness,
            min_size_filter=args.min_segment_size
        )

        if args.debug_vis_interval > 0 and num_segments > 0 and i % args.debug_vis_interval == 0:
             seg_norm = ((sub_area_segments.astype(np.float32) / num_segments) * 255).astype(np.uint8)
             seg_colormap = cv2.applyColorMap(seg_norm, cv2.COLORMAP_JET)
             seg_colormap[sub_area_segments == 0] = [0, 0, 0] 
             cv2.imwrite(os.path.join(segments_debug_dir, f"{frame_basename_no_ext}_segments_colormap.png"), seg_colormap)
             overlay = cv2.addWeighted(static_frame, 0.7, seg_colormap, 0.3, 0)
             cv2.imwrite(os.path.join(segments_debug_dir, f"{frame_basename_no_ext}_segments_overlay.png"), overlay)

        current_data = {
            'depth': depth, 'sub_area_segments': sub_area_segments, 'num_segments': num_segments,
            'frame_basename': frame_basename_no_ext,
            'K': K_curr,
            'static_mask': static_mask,
            'frame_index': i
        }

        if i == 0:
            prev_data = current_data
            continue 

        tracks_for_pnp = None
        num_valid_pair_tracks = 0
        skip_pair = False
        if not prev_data or 'frame_index' not in prev_data:
            logging.warning(f"Skipping pair ({i-1}, {i}) because previous frame data is missing.")
            skip_pair = True
        else:
            prev_frame_idx = prev_data['frame_index']

            if prev_frame_idx not in vggt_poses or i not in vggt_poses or \
               prev_frame_idx not in vggt_intrinsics: 
                logging.warning(f"Skipping pair ({i-1}, {i}): Missing VGGT pose/intrinsics for frame {prev_frame_idx} or {i}.")
                skip_pair = True

        if not skip_pair:
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

                        visible_curr = (vis_curr > 0)
                        confident_enough = conf_curr >= args.min_track_confidence
                        valid_mask = visible_curr & confident_enough
                        num_valid_pair_tracks = np.sum(valid_mask)

                        if num_valid_pair_tracks >= 4: 
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


        all_results['num_valid_tracks_per_pair'].append(num_valid_pair_tracks)
        if skip_pair:
            all_results['skipped_pairs'] += 1
            all_results['frame_local_consistency_rot_var'].append(np.nan)
            all_results['frame_local_consistency_trans_var'].append(np.nan)
            all_results['frame_reprojection_errors'].append(np.nan)
            all_results['frame_global_consistency_rot_var'].append(np.nan) 
            all_results['frame_global_consistency_trans_var'].append(np.nan)
            all_results['frame_depth_consistency_error'].append(np.nan)
            all_results['mean_local_relative_poses'].append(None)
            all_results['vggt_relative_poses'].append(None)
            all_results['num_pnp_poses_per_pair'].append(0)

            if len(all_results['num_valid_tracks_per_pair']) == processed_pairs_count:
                all_results['num_valid_tracks_per_pair'].append(0) 

            if 'frame_index' in current_data: prev_data = current_data
            else: prev_data = {}
            continue 


        processed_pairs_count += 1
        K_prev = prev_data['K']

        sub_area_poses = estimate_pose_per_subarea(
            i, current_data['sub_area_segments'], current_data['num_segments'],
            tracks_for_pnp, prev_data['depth'],
            K_prev, K_curr, 
            None, 
            args.pnp_reprojection_error
        )
        num_estimated_poses = len(sub_area_poses)
        all_results['num_pnp_poses_per_pair'].append(num_estimated_poses)
        logging.info(f"Pair ({i-1}, {i}): Estimated local poses for {num_estimated_poses}/{current_data['num_segments']} sub-areas.")

        local_rot_var, local_trans_var, mean_R_local, mean_t_local = assess_local_consistency(sub_area_poses)
        all_results['frame_local_consistency_rot_var'].append(local_rot_var)
        all_results['frame_local_consistency_trans_var'].append(local_trans_var)
        all_results['mean_local_relative_poses'].append((mean_R_local, mean_t_local) if mean_R_local is not None else None) 
        log_level = logging.INFO if np.isfinite(local_rot_var) and np.isfinite(local_trans_var) else logging.WARNING
        logging.log(log_level, f"Pair ({i-1}, {i}): Local Consistency: R Var={local_rot_var:.4f} (deg^2), T Var={local_trans_var:.4f}")

        prev_frame_idx = prev_data['frame_index']
        curr_frame_idx = current_data['frame_index']
        global_rot_var, global_trans_var = np.nan, np.nan 
        R_vggt_rel, t_vggt_rel = None, None 

        pose_wc_prev_vggt = vggt_poses[prev_frame_idx]
        pose_wc_curr_vggt = vggt_poses[curr_frame_idx]

        R_vggt_rel, t_vggt_rel = calculate_relative_pose(pose_wc_prev_vggt, pose_wc_curr_vggt)
        all_results['vggt_relative_poses'].append((R_vggt_rel, t_vggt_rel) if R_vggt_rel is not None else None) 

        if R_vggt_rel is not None and t_vggt_rel is not None:
            logging.debug(f"Pair ({i-1}, {i}): Calculated relative VGGT pose successfully.")
            reference_pose_vggt = (R_vggt_rel, t_vggt_rel)
            global_rot_var, global_trans_var = calculate_variance_vs_reference_pose(
                sub_area_poses, reference_pose_vggt
            )
        else:
            logging.warning(f"Pair ({i-1}, {i}): Failed to calculate relative pose from VGGT. Cannot calculate global variance.")

        all_results['frame_global_consistency_rot_var'].append(global_rot_var) 
        all_results['frame_global_consistency_trans_var'].append(global_trans_var) 
        log_level_global = logging.INFO if np.isfinite(global_rot_var) and np.isfinite(global_trans_var) else logging.WARNING
        logging.log(log_level_global, f"Pair ({i-1}, {i}): Global Consistency vs VGGT: R Var={global_rot_var:.4f} (deg^2), T Var={global_trans_var:.4f}")

        points_prev_for_reproj = tracks_for_pnp['points_prev']
        points_curr_for_reproj = tracks_for_pnp['points_curr']
        points_3d_prev_reproj, _, valid_mask_reproj = get_3d_points(points_prev_for_reproj, prev_data['depth'], K_prev)
        avg_reproj_error_pair = np.nan
        if points_3d_prev_reproj.shape[0] > 0 and num_estimated_poses > 0:
            points_2d_curr_actual_reproj = points_curr_for_reproj[valid_mask_reproj]
            valid_reproj_errors_seg = []
            for seg_id, pose in sub_area_poses.items():
                reproj_err = calculate_reprojection_error(
                    points_3d_prev_reproj, points_2d_curr_actual_reproj, pose, K_curr,
                    current_data['sub_area_segments'], seg_id, None 
                )
                if np.isfinite(reproj_err):
                    valid_reproj_errors_seg.append(reproj_err)

            if valid_reproj_errors_seg:
                trimmed_errors = trim_outliers_iqr(np.array(valid_reproj_errors_seg), cov=98) 
                if trimmed_errors.size > 0:
                    avg_reproj_error_pair = np.mean(trimmed_errors)
                else: 
                    avg_reproj_error_pair = np.mean(valid_reproj_errors_seg) 
            else:
                avg_reproj_error_pair = np.nan

        else:
            if points_3d_prev_reproj.shape[0] == 0: logging.debug("No valid 3D points for reprojection.")
            if num_estimated_poses == 0: logging.debug("No estimated poses for reprojection.")

        all_results['frame_reprojection_errors'].append(avg_reproj_error_pair)
        log_level_reproj = logging.INFO if np.isfinite(avg_reproj_error_pair) else logging.WARNING
        logging.log(log_level_reproj, f"Pair ({i-1}, {i}): Avg Local Reprojection Error: {avg_reproj_error_pair:.4f} pixels")

        depth_error = np.nan
        pose_for_warp = (R_vggt_rel, t_vggt_rel) if R_vggt_rel is not None else None

        if pose_for_warp:
            logging.debug(f"Pair ({i-1}, {i}): Warping depth using relative VGGT pose.")
            depth_warped = warp_depth(prev_data['depth'], K_prev, K_curr, pose_for_warp, current_data['depth'].shape)

            static_mask_curr = current_data['static_mask']
            valid_comparison_mask = (depth_warped > 1e-5) & (current_data['depth'] > 1e-5) & static_mask_curr

            if np.any(valid_comparison_mask):
                depth_diff = np.abs(depth_warped[valid_comparison_mask] - current_data['depth'][valid_comparison_mask])
                depth_error = np.mean(depth_diff)
                logging.debug(f"Pair ({i-1}, {i}): Compared {np.sum(valid_comparison_mask)} pixels for depth consistency.")
            else:
                logging.warning(f"Pair ({i-1}, {i}): No valid overlapping pixels found for depth consistency check.")
                depth_error = np.nan

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

        prev_data = current_data

    logging.info(f"Finished processing. Total pairs processed successfully: {processed_pairs_count}/{max(0, num_frames - 1)}")
    if processed_pairs_count == 0:
         logging.error("No frame pairs were successfully processed. Cannot calculate final scores.")
         return None

    local_rot_vars   = all_results['frame_local_consistency_rot_var']
    local_trans_vars = all_results['frame_local_consistency_trans_var']
    reproj           = all_results['frame_reprojection_errors']
    global_rot_vars  = all_results['frame_global_consistency_rot_var'] 
    global_trans_vars= all_results['frame_global_consistency_trans_var']
    depth_err        = all_results['frame_depth_consistency_error']

    metrics = {
        'local_rot_var':   local_rot_vars,
        'local_trans_var': local_trans_vars,
        'reproj_error':      reproj,
        'global_rot_var':    global_rot_vars, 
        'global_trans_var':  global_trans_vars,
        'depth_error':       depth_err,
    }

    sns.set_theme(style="whitegrid")
    for name, values in metrics.items():

        valid_indices = [j for j, v in enumerate(values) if np.isfinite(v)]
        if not valid_indices:
            logging.warning(f"No valid data points to plot for metric: {name}")
            continue
        valid_values = [values[j] for j in valid_indices]

        plt.figure(figsize=(10, 4)) 
        sns.lineplot(x=valid_indices, y=valid_values, marker="o", legend=False) 

        plt.title(f'{name} over Processed Pairs')
        plt.xlabel('Processed Pair Index (Frame i-1 -> i)') 
        plt.ylabel(name)
        plt.xlim(left=-1, right=len(values)) 
        plt.tight_layout() 

        out_path = os.path.join(plot_debug_dir, f'{name}_over_pairs.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved plot: {out_path}")


    avg_local_rot_var = np.nanmean(local_rot_vars)
    avg_local_trans_var = np.nanmean(local_trans_vars)
    avg_reproj_error = np.nanmean(reproj)
    avg_global_rot_var = np.nanmean(global_rot_vars)     
    avg_global_trans_var = np.nanmean(global_trans_vars) 
    avg_depth_error = np.nanmean(depth_err)
    avg_valid_tracks = np.nanmean(all_results['num_valid_tracks_per_pair'])
    avg_pnp_poses = np.nanmean(all_results['num_pnp_poses_per_pair'])

    def nan_to_none(value):
        return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


    final_results = {

        'average_local_rotation_variance': nan_to_none(avg_local_rot_var),
        'average_local_translation_variance': nan_to_none(avg_local_trans_var),
        'average_reprojection_error': nan_to_none(avg_reproj_error),
        'average_global_rotation_variance': nan_to_none(avg_global_rot_var), 
        'average_global_translation_variance': nan_to_none(avg_global_trans_var),
        'average_depth_consistency_error': nan_to_none(avg_depth_error),
        'average_valid_tracks_per_pair': nan_to_none(avg_valid_tracks),
        'average_pnp_poses_per_pair': nan_to_none(avg_pnp_poses),

        'num_frames': num_frames,
        'num_processed_pairs': processed_pairs_count,
        'num_skipped_pairs': all_results['skipped_pairs'],
    }

    logging.info(f"--- Final Results (using VGGT, Variance-based Global) ---")
    logging.info(json.dumps(final_results, indent=4))
    return final_results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D consistency using local variance, global variance vs VGGT, reprojection, and depth.")

    parser.add_argument("--frames_dir", type=str, required=True, help="Base directory containing video frame images (subdirs per video).")
    parser.add_argument("--depth_dir", type=str, required=True, help="Base directory containing depth maps (NPZ with 'depths' key, subdirs per video).")
    parser.add_argument("--mos_dir", type=str, required=True, help="Base directory containing Moving Object Segmentation masks (subdirs per video).")
    parser.add_argument("--tracks_dir", type=str, required=True, help="Base directory containing pairwise tracking files (subdirs per video).")
    parser.add_argument("--output_json", type=str, default="consistency_results_globalvar.json", help="Path to save the final results JSON file (will be prefixed with video name).")
    parser.add_argument("--video_name", type=str, required=True, help="Name of the video subdirectory to process.")

    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Factor to divide raw depth values by (e.g., 1000.0 for mm to meters). Set <= 0 to use raw values.")
    parser.add_argument("--segmentation_method", type=str, default="depth", choices=["slic", "depth", "semantic","grid"], help="Method for segmenting static background.")
    parser.add_argument("--static_semantic_labels", type=int, nargs='+', default=[1, 2, 5], help="List of semantic labels considered static (used if segmentation_method is 'semantic').")
    parser.add_argument("--n_slic_segments", type=int, default=20, help="Approximate number of SLIC superpixels.")
    parser.add_argument("--slic_compactness", type=float, default=10.0, help="Compactness parameter for SLIC.")
    parser.add_argument("--n_depth_clusters", type=int, default=16, help="Number of clusters for depth-based segmentation.")
    parser.add_argument("--min_segment_size", type=int, default=200, help="Minimum pixel count for a segment to be considered.")
    parser.add_argument("--pnp_reprojection_error", type=float, default=8.0, help="RANSAC reprojection error threshold for PnP (pixels).")
    parser.add_argument("--min_track_confidence", type=float, default=0.5, help="Minimum confidence score (from tracks_file[:,:,3]) for a track point to be used (applied to current frame). Default 0.0 uses all visible.")

    parser.add_argument("--vggt_model_name", type=str, default="facebook/VGGT-1B", help="Name of the VGGT model to load from Hugging Face.")

    parser.add_argument("--debug_vis_interval", type=int, default=10, help="Interval (in frames) for saving debug visualizations. 0 to disable.")

    args = parser.parse_args()

    video_name = args.video_name
    logging.info(f"===== Evaluating Video: {video_name} =====")

    current_frames_dir = os.path.join(args.frames_dir, video_name)
    current_depth_dir = os.path.join(args.depth_dir, video_name)
    current_mos_dir = os.path.join(args.mos_dir, video_name)
    current_tracks_dir = os.path.join(args.tracks_dir, video_name)
    current_semantics_dir = os.path.join(args.semantics_dir, video_name) if args.semantics_dir else None

    if not os.path.isdir(current_frames_dir): logging.error(f"Frames directory not found: {current_frames_dir}"); exit()
    if not os.path.isdir(current_depth_dir): logging.error(f"Depth directory not found: {current_depth_dir}"); exit()
    if not os.path.isdir(current_mos_dir): current_mos_dir = None
    if not os.path.isdir(current_tracks_dir): logging.error(f"Tracks directory not found: {current_tracks_dir}"); exit()
    if args.semantics_dir and current_semantics_dir and not os.path.isdir(current_semantics_dir): logging.warning(f"Semantics directory specified but not found: {current_semantics_dir}")

    video_args = argparse.Namespace(**vars(args))
    video_args.frames_dir = current_frames_dir
    video_args.depth_dir = current_depth_dir
    video_args.mos_dir = current_mos_dir
    video_args.tracks_dir = current_tracks_dir
    video_args.semantics_dir = current_semantics_dir

    try:
        results = evaluate_3d_consistency(video_args) 
        video_name_top = video_name.split('_')[0] if '_' in video_name else video_name
        # video_name_top = "OpenVid"
        os.makedirs(os.path.join('result', video_name_top), exist_ok=True)
        args.output_json = os.path.join('result',video_name_top, f"{video_name}.json")
        if results:
            try:
                 with open(args.output_json, 'w') as f:
                    json.dump(results, f, indent=4)
                 logging.info(f"Results for {video_name} saved to {args.output_json}")
            except TypeError as e:
                logging.error(f"Could not serialize results to JSON for {video_name}. Error: {e}")
            except Exception as e:
                 logging.error(f"Could not save results to JSON '{args.output_json}': {e}")
    except Exception as e:
        logging.exception(f"An critical error occurred during the evaluation pipeline for video {video_name}:")

    logging.info(f"===== Finished Evaluating Video: {video_name} =====")