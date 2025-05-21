import math
import os
import sys
import types
from typing import List, Tuple, Union
import argparse
import numpy as np
# import timm
import torch
import torch.nn.modules.utils as nn_utils
from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm import trange
from glob import glob
import imageio as iio

def extract_and_save_features(
    input_img_path_list: List[str],
    saved_feat_path_list: List[str],
    img_shape: Tuple[int, int] = (640, 960),
    stride: int = 8,
    model_type: str = "dino_vitb8",
    batch_size: int = 8, # <<< Added batch_size parameter >>>
) -> Union[np.ndarray, None]:
    """
    Extracts DINO features from a list of images and saves them to disk using batch processing.
    # ... (docstring) ...
    """
    assert len(input_img_path_list) == len(
        saved_feat_path_list
    ), "input_img_path_list and saved_feat_path_list must have the same length."
    img_shape = list(img_shape)
    extractor = ViTExtractor(
        model_type=model_type,
        stride=stride,
        device="cuda"
    )
    prep = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_shape),
            transforms.Normalize(
                mean=extractor.mean,
                std=extractor.std,
            ),
        ]
    )
    # <<< Determine layer >>>
    # ... (layer determination logic remains the same) ...
    if "dinov2" in model_type:
        if "vitl" in model_type: layer_num = 23
        elif "vitg" in model_type: layer_num = 39
        else: layer_num = 11
    elif "dino" in model_type:
        layer_num = 11
    else:
        print(f"Warning: Unknown base model type for {model_type}. Defaulting to layer 11.")
        layer_num = 11
    layers_to_extract = [layer_num]

    total_images = len(input_img_path_list)
    for i in trange(
        0,
        total_images,
        batch_size,
        desc=f"Extracting features (batchsz={batch_size}, layer={layer_num}, skips existing)",
        dynamic_ncols=True,
    ):
        batch_start_idx = i
        batch_end_idx = min(i + batch_size, total_images)

        current_batch_img_pths_to_process = []
        current_batch_feat_pths_to_save = []
        for idx in range(batch_start_idx, batch_end_idx):
            img_pth = input_img_path_list[idx]
            feat_pth = saved_feat_path_list[idx]
            if not os.path.exists(feat_pth):
                current_batch_img_pths_to_process.append(img_pth)
                current_batch_feat_pths_to_save.append(feat_pth)

        if not current_batch_img_pths_to_process:
            continue

        images_to_process_tensors = []
        valid_feat_paths_in_batch = []
        for img_pth, feat_pth in zip(current_batch_img_pths_to_process, current_batch_feat_pths_to_save):
            try:
                img = Image.open(img_pth).convert("RGB")
                images_to_process_tensors.append(prep(img))
                valid_feat_paths_in_batch.append(feat_pth)
            except Exception as e:
                print(f"Warning: Failed to load/preprocess {img_pth}: {e}. Skipping this image.")
        if not images_to_process_tensors:
            continue

        preproc_image_batch = torch.stack(images_to_process_tensors, dim=0).to(extractor.device)

        with torch.no_grad():
            descriptors_batch = extractor.extract_descriptors(
                preproc_image_batch,
                layers_to_extract,
                "key",
                include_cls=False,
                bin=False
            )

        if extractor.num_patches is None:
             print("Error: extractor.num_patches not set after descriptor extraction.")
             continue

        h_patches, w_patches = extractor.num_patches
        actual_batch_size = descriptors_batch.shape[0]

        for batch_idx in range(actual_batch_size):
            feat_pth = valid_feat_paths_in_batch[batch_idx]
            os.makedirs(os.path.dirname(feat_pth), exist_ok=True)
            descriptor_single = descriptors_batch[batch_idx]
            try:
                reshaped_descriptor = descriptor_single.reshape(h_patches, w_patches, -1)
            except RuntimeError as e:
                print(f"Error reshaping descriptor for {feat_pth}. Shape was {descriptor_single.shape}, target H/W patches {h_patches}/{w_patches}. Error: {e}")
                continue
            final_descriptor_np = reshaped_descriptor.cpu().numpy()
            np.save(feat_pth, final_descriptor_np.astype(np.float16))

    # --- Cleanup ---
    del extractor
    # clear the cache
    torch.cuda.empty_cache()
    return {}


def delete_features(
    saved_feat_path_list: List[str],
    verbose: bool = True,
) -> None:
    """
    Deletes features from disk.
    """
    for i in trange(
        0,
        len(saved_feat_path_list),
        desc="Deleting features (will skip non-existing files)",
        dynamic_ncols=True,
    ):
        feat_pth = saved_feat_path_list[i]

        file_exists = os.path.exists(feat_pth)
        if file_exists:
            os.remove(feat_pth)
            if verbose:
                print(f"Deleted {feat_pth}")
        else:
            continue


class ViTExtractor:
    # Modified from https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py
    """This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(
        self,
        model_type: str = "dino_vits8",
        stride: int = 4,
        model: nn.Module = None,
        device: str = "cuda",
    ):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            try:
                self.model = ViTExtractor.create_model(model_type)
            except Exception as e:
                print(f"Error creating model {model_type} from torch hub: {e}")
                print("Please check model name and internet connection.")
                raise e

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        print(self.model)
        p = (
            self.model.patch_embed.patch_size
            if isinstance(self.model.patch_embed.patch_size, int)
            else self.model.patch_embed.patch_size[0]
        )
        self.p = p
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (
            (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        )
        self.std = (
            (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        )

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        Creates the model. Attempts to load directly from the local hub cache
        using source='local' if the cache directory exists, otherwise uses the
        standard torch.hub.load mechanism (which will use cache or download).
        """
        model = None
        hub_dir = torch.hub.get_dir() # Get the base hub cache directory (~/.cache/torch/hub)

        # Determine the expected repository name and local directory structure
        # based on common torch hub patterns. Adjust if models come from different sources.
        if "dinov2" in model_type:
            github_repo_name = "facebookresearch/dinov2"
            # Default branch/tag used by hub is typically 'main'
            # Construct the expected subdirectory name within the hub cache
            local_repo_dir_name = github_repo_name.replace('/', '_') + '_main'
            expected_local_path = os.path.join(hub_dir, local_repo_dir_name)
            standard_hub_source = github_repo_name # e.g., "facebookresearch/dinov2"

        elif "dino" in model_type: # Original DINO
            github_repo_name = "facebookresearch/dino"
            local_repo_dir_name = github_repo_name.replace('/', '_') + '_main'
            expected_local_path = os.path.join(hub_dir, local_repo_dir_name)
            standard_hub_source = 'facebookresearch/dino:main' # As used before
        else:
             # If other model types are added, define their repo/local path logic here
             raise NotImplementedError(
                 f"Model type '{model_type}' not explicitly handled for local cache check."
             )

        if os.path.isdir(expected_local_path):
            try:
                # Use source='local', providing the path to the cached directory
                model = torch.hub.load(
                    repo_or_dir=expected_local_path,
                    model=model_type,
                    source='local'
                    # trust_repo=True # May be needed depending on repo/torch version
                                    # Add if you encounter trust issues. DINOv2 likely doesn't need it.
                )
            except Exception as e:
                model = None # Ensure fallback happens by resetting model to None
        else:
            print(f"Local cache directory not found at expected path.")

        # --- Fallback to standard github loading if local failed or wasn't attempted ---
        if model is None:
            try:
                # Use the standard source (usually GitHub repo name)
                # torch.hub will handle finding the cache itself or downloading if needed.
                model = torch.hub.load(
                    repo_or_dir=standard_hub_source,
                    model=model_type
                    # trust_repo=True # Add if needed
                    )
                print(f"Model loaded via standard torch.hub.load.")
            except Exception as e:
                print(f"CRITICAL: Failed to load model using standard torch.hub.load: {e}")
                # Re-raise the exception if the standard method also fails,
                # as the model cannot be loaded at all.
                raise e

        # Final check
        if model is None:
             # This path should ideally not be reached if the logic is sound
             raise RuntimeError(f"Failed to create model for type '{model_type}' after all attempts.")

        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(
            self, x: torch.Tensor, w: int, h: int
        ) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all(
            [(patch_size // s_) * s_ == patch_size for s_ in stride]
        ), f"stride {stride} should divide patch_size {patch_size}"

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(
            ViTExtractor._fix_pos_enc(patch_size, stride), model
        )
        return model

    def preprocess(
        self,
        image_path,
        load_size: Union[int, Tuple[int, int]],
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        image = Image.open(image_path).convert("RGB")
        prep = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(load_size),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        prep_img = prep(image)[None, ...]
        return prep_img

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ["attn", "token"]:

            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = (
                module.qkv(input)
                .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(
                        block.register_forward_hook(self._get_hook(facet))
                    )
                elif facet == "attn":
                    self.hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(
                            self._get_hook(facet)
                        )
                    )
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(
                        block.attn.register_forward_hook(self._get_hook(facet))
                    )
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(
        self, batch: torch.Tensor, layers: List[int] = 11, facet: str = "key"
    ) -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (
            1 + (H - self.p) // self.stride[0],
            1 + (W - self.p) // self.stride[1],
        )
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(
            B, bin_x.shape[1], self.num_patches[0], self.num_patches[1]
        )
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3**k
            avg_pool = torch.nn.AvgPool2d(
                win_size, stride=1, padding=win_size // 2, count_include_pad=False
            )
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros(
            (B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])
        ).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3**k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(
                            x - kernel_size, x + kernel_size + 1, kernel_size
                        ):
                            if i == y and j == x and k != 0:
                                continue
                            if (
                                0 <= i < self.num_patches[0]
                                and 0 <= j < self.num_patches[1]
                            ):
                                bin_x[
                                    :,
                                    part_idx
                                    * sub_desc_dim : (part_idx + 1)
                                    * sub_desc_dim,
                                    y,
                                    x,
                                ] = avg_pools[k][:, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[
                                    :,
                                    part_idx
                                    * sub_desc_dim : (part_idx + 1)
                                    * sub_desc_dim,
                                    y,
                                    x,
                                ] = avg_pools[k][:, :, temp_i, temp_j]
                            part_idx += 1
        bin_x = (
            bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        )
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(
        self,
        batch: torch.Tensor,
        layer: List[int],
        facet: str = "key",
        bin: bool = False,
        include_cls: bool = False,
    ) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in [
            "key",
            "query",
            "value",
            "token",
        ], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, layer, facet)
        x = torch.concat(self._feats)
        # if facet == 'token':
        #     x.unsqueeze_(dim=1) #Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert (
                not bin
            ), "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = (
                x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)
            )  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        if "reg" in self.model_type:
            desc = desc[..., 4:, :]
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert (
            self.model_type == "dino_vits8"
        ), f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], "attn")
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0]  # Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1)  # Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (
            temp_maxs - temp_mins
        )  # normalize to range [0,1]
        return cls_attn_maps

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract features using ViT models with batching', # <<< Updated description
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_dir', type=str, default="current-data-dir/davis/DAVIS/moving/480p/boat", help="Directory containing input images.")
    parser.add_argument('--model_type', type=str, default="dinov2_vitb14", help="Type of ViT model (e.g., dino_vitb8, dinov2_vitb14).")
    parser.add_argument('--stride', type=int, default=7, help="Patch embed stride.") # <<< Clarified help text >>>
    parser.add_argument('--step', type=int, default=10, help="Frame sampling step.")
    # <<< Added batch_size argument >>>
    parser.add_argument('--batch_size', type=int, default=8, help="Number of images to process per batch.")
    # <<< Added optional output dir argument >>>
    parser.add_argument('--output_dir', type=str, default=None, help="Optional: Directory to save features. If None, defaults to ../dinos/ adjacent to image_dir.")
    # <<< Added optional image shape args >>>
    parser.add_argument('--img_height', type=int, default=None, help="Optional: Force image resize height. If None, calculated from first image.")
    parser.add_argument('--img_width', type=int, default=None, help="Optional: Force image resize width. If None, calculated from first image.")


    args = parser.parse_args()

    img_paths_all = sorted(glob(os.path.join(args.image_dir, "*.png"))) + \
                    sorted(glob(os.path.join(args.image_dir, "*.jpg"))) + \
                    sorted(glob(os.path.join(args.image_dir, "*.jpeg")))

    if not img_paths_all:
        print(f"Error: No images found in {args.image_dir}")
        sys.exit(1)

    num_frames = len(img_paths_all)
    q_ts = list(range(0, num_frames, args.step))
    img_paths = [img_paths_all[q] for q in q_ts if q < num_frames] # Ensure index is within bounds

    if not img_paths:
        print(f"Error: No images selected with step {args.step} from {num_frames} total frames.")
        sys.exit(1)

    if args.output_dir:
        save_dir = args.output_dir
    else:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(args.image_dir)), "dinos", os.path.basename(args.image_dir))

    if os.path.exists(save_dir):
        print(f"Output directory {save_dir} already exists. Features for existing files will be skipped.")
    os.makedirs(save_dir, exist_ok=True)

    frame_names = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
    save_paths = [os.path.join(save_dir,f"{n}.npy") for n in frame_names]

    # --- Determine Image Shape ---
    if args.img_height is not None and args.img_width is not None:
        H, W = args.img_height, args.img_width
        print(f"Using provided image shape: H={H}, W={W}")
    else:
        # <<< Minimal Change: Load first image to get shape >>>
        try:
            # Use Image.open to get dimensions without loading full data potentially
            with Image.open(img_paths[0]) as img_pil:
                W_orig, H_orig = img_pil.size # PIL uses W, H order
            # img = torch.from_numpy(np.array([iio.imread(img_paths[0])])).squeeze().permute(2, 0, 1) # [1, 512, 512, 3]
            # H_orig, W_orig = img.shape[1], img.shape[2]
            print(f"Detected shape from first image ({img_paths[0]}): H={H_orig}, W={W_orig}")
        except Exception as e:
            print(f"Error reading first image {img_paths[0]} to determine shape: {e}")
            print("Please provide --img_height and --img_width.")
            sys.exit(1)
        H, W = H_orig, W_orig # Use original H, W before padding

    # <<< Determine patch size from model_type for padding >>>
    # Heuristic, adjust if needed
    if '16' in args.model_type:
        patch_size = 16
    elif '14' in args.model_type: # DINOv2 uses 14
        patch_size = 14
    elif '8' in args.model_type:
        patch_size = 8
    else:
        print(f"Warning: Could not determine patch size from model_type '{args.model_type}'. Assuming 14.")
        patch_size = 14 # Default assumption

    # <<< Pad H, W to be divisible by patch size >>>
    # Stride affects feature map size, but ViT input needs padding based on patch size
    H_pad = (H + patch_size - 1) // patch_size * patch_size
    W_pad = (W + patch_size - 1) // patch_size * patch_size
    if H != H_pad or W != W_pad:
        print(f"Padding image shape to be divisible by patch size {patch_size}: H={H_pad}, W={W_pad}")

    target_shape = (H_pad, W_pad)

    return_dict = extract_and_save_features(
        img_paths,
        save_paths,
        target_shape, # Use padded shape
        args.stride,
        args.model_type,
        args.batch_size # Pass batch size argument
    )