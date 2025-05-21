# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# import torch
# import torch.nn.functional as F
# from densetrack3d.models.model_utils import bilinear_sampler, get_points_on_a_grid, convert_trajs_uvd_to_trajs_3d
# from einops import einsum, rearrange, repeat


# class DensePredictor2D(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.interp_shape = model.model_resolution
#         self.model = model
#         self.n_iters = 6

#     @torch.inference_mode()
#     def forward(
#         self,
#         video,  # (B, T, 3, H, W)
#         grid_query_frame: int = 0,  # only for dense and regular grid tracks
#         scale_input: bool = True,
#         scale_to_origin: bool = True,
#         use_efficient_global_attn: bool = True,
#     ):
#         B, T, C, H, W = video.shape
#         device = video.device
#         src_step = grid_query_frame

#         ori_video = video.clone()

#         if scale_input:
#             video = F.interpolate(
#                 video.flatten(0, 1), tuple(self.interp_shape), mode="bilinear", align_corners=True
#             ).reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])


#         if use_efficient_global_attn:
#             sparse_xy = get_points_on_a_grid((36, 48), video.shape[3:]).long().float()
#             sparse_xy = torch.cat([src_step * torch.ones_like(sparse_xy[:, :, :1]), sparse_xy], dim=2).to(
#                 device
#             )  # B, N, C
#             sparse_queries = sparse_xy
#         else:
#             sparse_queries = None

#         sparse_predictions, dense_predictions, _ = self.model(
#             video=video[:, src_step:],
#             sparse_queries=sparse_queries,
#             iters=self.n_iters,
#             use_efficient_global_attn=use_efficient_global_attn,
#         )

#         dense_traj_e, dense_vis_e = (
#             dense_predictions["coords"],
#             dense_predictions["vis"],
#         )

#         if "conf" not in dense_predictions.keys():
#             dense_conf_e = dense_vis_e.clone()
#         else:
#             dense_conf_e = dense_predictions["conf"]

#         if scale_to_origin:
#             dense_traj_e[:, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
#             dense_traj_e[:, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

#         dense_traj_e = rearrange(dense_traj_e, "b t c h w -> b t (h w) c")
#         dense_vis_e = rearrange(dense_vis_e, "b t h w -> b t (h w)")
#         dense_conf_e = rearrange(dense_conf_e, "b t h w -> b t (h w)")

#         dense_vis_e = dense_vis_e > 0.8


#         out = {
#             "trajs_uv": dense_traj_e,
#             "vis": dense_vis_e,
#             "conf": dense_conf_e,
#             "dense_reso": self.interp_shape,
#         }

#         return out
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from densetrack3d.models.model_utils import bilinear_sampler, get_points_on_a_grid, convert_trajs_uvd_to_trajs_3d
from einops import rearrange # removed unused imports

class DensePredictor2D(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # Ensure model_resolution exists on the model object
        self.interp_shape = getattr(model, 'model_resolution', (256, 256)) # Example default
        self.model = model
        self.n_iters = getattr(model, 'n_iters', 6) # Use n_iters from model or default

    def _prepare_sparse_queries(self, query_frame_step, video_shape, device, grid_spec=(36, 48)):
        """Helper to prepare sparse queries for efficient attention."""
        sparse_xy = get_points_on_a_grid(grid_spec, video_shape[3:]).float() # Use float directly
        # Ensure sparse_xy is (1, N, 2) before cat
        if sparse_xy.dim() == 3 and sparse_xy.shape[0] != 1:
             sparse_xy = sparse_xy.unsqueeze(0) # Add batch dim if missing
        elif sparse_xy.dim() == 2:
             sparse_xy = sparse_xy.unsqueeze(0) # Add batch dim if missing

        time_coords = torch.full_like(sparse_xy[:, :, :1], float(query_frame_step))
        sparse_queries = torch.cat([time_coords, sparse_xy], dim=2).to(device) # B, N, 3
        return sparse_queries

    @torch.inference_mode()
    def forward(
        self,
        video,  # (B, T, 3, H, W)
        grid_query_frame: int = 0,
        backward_tracking: bool = False, # <<< ADDED FLAG
        scale_input: bool = True,
        scale_to_origin: bool = True,
        use_efficient_global_attn: bool = True, # Renamed from use_dense for clarity
        vis_threshold: float = 0.8, # Make threshold configurable
    ):
        B, T, C, H, W = video.shape
        device = video.device
        src_step = grid_query_frame

        # --- Input Scaling ---
        ori_video = video.clone() # Keep original for potential future use (not strictly needed here)
        if scale_input:
            video_scaled = F.interpolate(
                video.flatten(0, 1), tuple(self.interp_shape), mode="bilinear", align_corners=True
            ).reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
            interp_H, interp_W = self.interp_shape
        else:
            video_scaled = video # Use original video if not scaling
            interp_H, interp_W = H, W

        # --- Forward Pass ---
        fwd_traj_e = torch.empty((B, 0, 2, interp_H, interp_W), device=device, dtype=video.dtype)
        fwd_vis_e = torch.empty((B, 0, interp_H, interp_W), device=device, dtype=torch.bool)
        fwd_conf_e = torch.empty((B, 0, interp_H, interp_W), device=device, dtype=video.dtype)

        if src_step < T: # Only run forward if query frame is within video bounds
            if use_efficient_global_attn:
                fwd_sparse_queries = self._prepare_sparse_queries(0, video_scaled.shape, device) # Time is 0 relative to the sliced video
            else:
                fwd_sparse_queries = None

            # Model processes video starting from src_step
            _, fwd_dense_preds, _ = self.model(
                video=video_scaled[:, src_step:],
                sparse_queries=fwd_sparse_queries,
                iters=self.n_iters,
                use_efficient_global_attn=use_efficient_global_attn,
                use_dense=True # Ensure dense output is requested
            )
            # Output shapes: (B, T_fwd, C, H_interp, W_interp), (B, T_fwd, H_interp, W_interp)
            fwd_traj_e = fwd_dense_preds["coords"]
            fwd_vis_e = fwd_dense_preds["vis"]
            fwd_conf_e = fwd_dense_preds.get("conf", fwd_vis_e.clone()) # Handle optional conf

        # --- Backward Pass ---
        bwd_traj_e = torch.empty((B, 0, 2, interp_H, interp_W), device=device, dtype=video.dtype)
        bwd_vis_e = torch.empty((B, 0, interp_H, interp_W), device=device, dtype=torch.bool)
        bwd_conf_e = torch.empty((B, 0, interp_H, interp_W), device=device, dtype=video.dtype)

        if backward_tracking and src_step > 0:
            # Flip the *entire scaled* video for backward pass
            inv_video_scaled = video_scaled.flip(1)
            # Query frame index in the *reversed* video
            inv_src_step = T - 1 - src_step

            if use_efficient_global_attn:
                # Time is 0 relative to the sliced *reversed* video
                bwd_sparse_queries = self._prepare_sparse_queries(0, inv_video_scaled.shape, device)
            else:
                bwd_sparse_queries = None

            # Model processes the relevant part of the reversed video
            # This part starts at inv_src_step in the reversed timeline
            _, bwd_dense_preds, _ = self.model(
                video=inv_video_scaled[:, inv_src_step:],
                sparse_queries=bwd_sparse_queries,
                iters=self.n_iters,
                use_efficient_global_attn=use_efficient_global_attn,
                use_dense=True # Ensure dense output is requested
            )

            # Results are relative to reversed time, shape (B, T_bwd, ...)
            inv_traj_e = bwd_dense_preds["coords"]
            inv_vis_e = bwd_dense_preds["vis"]
            inv_conf_e = bwd_dense_preds.get("conf", inv_vis_e.clone())

            # Flip the time dimension of backward results to align with original time
            # Shape becomes (B, T_bwd, ...) corresponding to original time 0 to src_step-1
            bwd_traj_e = inv_traj_e.flip(1)
            bwd_vis_e = inv_vis_e.flip(1)
            bwd_conf_e = inv_conf_e.flip(1)

        # --- Combine Results ---
        # Concatenate along the time dimension (dim=1)
        # bwd results cover time [0, src_step-1]
        # fwd results cover time [src_step, T-1]
        combined_traj_e = torch.cat([bwd_traj_e, fwd_traj_e], dim=1) # B, T, C, H_int, W_int
        combined_vis_e = torch.cat([bwd_vis_e, fwd_vis_e], dim=1)   # B, T, H_int, W_int
        combined_conf_e = torch.cat([bwd_conf_e, fwd_conf_e], dim=1) # B, T, H_int, W_int

        # --- Post-processing (Scaling, Reshaping, Thresholding) ---
        # Apply scaling *after* combining if scale_to_origin is True
        if scale_to_origin and combined_traj_e.numel() > 0: # Check if tensor is not empty
            combined_traj_e[..., 0] *= (W - 1) / float(interp_W - 1)
            combined_traj_e[..., 1] *= (H - 1) / float(interp_H - 1)

        # Reshape to final dense format (B, T, N, C) and (B, T, N)
        if combined_traj_e.numel() > 0:
            dense_traj_e = rearrange(combined_traj_e, "b t c h w -> b t (h w) c")
            dense_vis_e = rearrange(combined_vis_e, "b t h w -> b t (h w)")
            dense_conf_e = rearrange(combined_conf_e, "b t h w -> b t (h w)")

            # Apply visibility threshold
            # dense_vis_e = dense_vis_e > vis_threshold
        else: # Handle case where no tracking was done (e.g., T=0 or invalid src_step combo)
            dense_traj_e = torch.empty((B, T, 0, 2), device=device, dtype=video.dtype)
            dense_vis_e = torch.empty((B, T, 0), device=device, dtype=torch.bool)
            dense_conf_e = torch.empty((B, T, 0), device=device, dtype=video.dtype)


        # --- Output ---
        out = {
            "trajs_uv": dense_traj_e, # Full trajectory
            "vis": dense_vis_e,       # Full visibility
            "conf": dense_conf_e,     # Full confidence
            # Report the resolution the tracks correspond to before scaling
            "dense_reso": (interp_H, interp_W),
        }

        return out