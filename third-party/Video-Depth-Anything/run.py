import argparse
import numpy as np
import os
import torch
import time # Import the time module

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video, read_image_frames, save_depth_images_parallel

# Conditional import for EXR saving
EXR_AVAILABLE = False
try:
    import OpenEXR
    import Imath
    EXR_AVAILABLE = True
except ImportError:
    print("Warning: OpenEXR or Imath module not found. EXR saving will be disabled.")
    pass

if __name__ == '__main__':
    # overall_start_time = time.perf_counter() # Start overall timer

    # --- Argument Parsing ---
    # (Timing this is usually negligible, skipping timing block for clarity)
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default=None, help="Path to input video file")
    parser.add_argument("--images_dir", type=str, default=None, help="Path to directory containing input image frames")
    # Add argument for original FPS if reading from images
    parser.add_argument("--original_fps", type=int, default=24, help="Original FPS of the sequence (required if using --images_dir)")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Directory to save results")
    parser.add_argument('--input_size', type=int, default=518, help="Input size for the model")
    parser.add_argument('--max_res', type=int, default=1280, help="Maximum resolution for input frames (resizes if exceeded)")
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'], help="Model encoder type")
    parser.add_argument('--max_len', type=int, default=-1, help='Maximum number of frames to process, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='Target FPS for processing (samples frames), -1 means original FPS')
    parser.add_argument('--fp32', action='store_true', help='Use torch.float32 for inference (default is torch.float16)')
    parser.add_argument('--grayscale', action='store_true', help='Save depth visualization video as grayscale')
    parser.add_argument('--save_png', action='store_true', help='Save individual depth frames as uint16 grayscale PNG')
    parser.add_argument('--save_npz', action='store_true', help='Save depths as a single compressed npz file')
    parser.add_argument('--save_exr', action='store_true', help='Save individual depth frames as EXR files')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    checkpoint_path = f'third-party/Video-Depth-Anything/checkpoints/video_depth_anything_{args.encoder}.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    video_depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    frames = None
    read_fps = None # FPS used for reading/sampling
    if args.images_dir is not None:
        print(f"    Reading from image directory: {args.images_dir}")
        frames, read_fps = read_image_frames(
            image_folder_path=args.images_dir,
            process_length=args.max_len,
            original_fps=args.original_fps, # Use provided original_fps
            target_fps=args.target_fps,
            max_res=args.max_res
        )
    else: # args.input_video must be set
        print(f"    Reading from video file: {args.input_video}")
        frames, read_fps = read_video_frames(
            video_path=args.input_video,
            process_length=args.max_len,
            target_fps=args.target_fps,
            max_res=args.max_res
        )

    if frames is None or len(frames) == 0:
        print("Error: No frames were read or processed. Exiting.")
        exit(1)

    depths = None
    inference_fps = None
    # Use the FPS determined during reading as the input target FPS for inference
    depths, inference_fps = video_depth_anything.infer_video_depth(
        frames, read_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32
    )
    if depths is None or len(depths) == 0:
         print("Error: Depth inference did not return any results. Exiting.")
         exit(1)

    save_fps = inference_fps

    if args.images_dir is not None:
        base_name = os.path.basename(os.path.abspath(args.images_dir)) # Get dir name
        video_name = base_name if base_name else "image_sequence" # Fallback name
    else:
        video_name = os.path.basename(args.input_video)
    output_base_filename = os.path.splitext(video_name)[0]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # --- Saving Processed Source Video ---
    # save_src_start_time = time.perf_counter()
    # processed_video_path = os.path.join(args.output_dir, output_base_filename + '_src.mp4')
    # print(f"Saving processed source frames to: {processed_video_path}")
    # try:
    #     save_video(frames, processed_video_path, fps=save_fps)
    # except Exception as e:
    #     print(f"Error saving source video: {e}")
    # save_src_end_time = time.perf_counter()
    # print(f"    Saving source video took: {save_src_end_time - save_src_start_time:.2f} seconds")


    # # --- Saving Depth Visualization Video ---
    # save_vis_start_time = time.perf_counter()
    # depth_vis_path = os.path.join(args.output_dir, output_base_filename + '_vis.mp4')
    # print(f"Saving depth visualization video to: {depth_vis_path}")
    # try:
    #     save_video(depths, depth_vis_path, fps=save_fps, is_depths=True, grayscale=args.grayscale)
    # except Exception as e:
    #     print(f"Error saving depth visualization video: {e}")
    # save_vis_end_time = time.perf_counter()
    # print(f"    Saving depth visualization video took: {save_vis_end_time - save_vis_start_time:.2f} seconds")


    # --- Saving Individual Depth Images (PNG - Conditional) ---
    if args.save_png:
        depth_img_folder = os.path.join(args.output_dir, output_base_filename)
        # save_depth_images(
        #     frames=depths,
        #     output_folder_path=depth_img_folder,
        #     file_prefix="",
        #     file_format="png",
        #     grayscale=True,
        #     normalize_globally=False, # Save raw-like values scaled to uint16 range per frame
        #     output_dtype=np.uint16
        # )
        save_depth_images_parallel(
            frames=depths,
            output_folder_path=args.output_dir,
            file_prefix="",
            file_format="png",
            grayscale=True,             # Must be True for the worker's logic
            normalize_globally=False,   # Must be False for the worker's logic
            output_dtype=np.uint16,
            num_workers=32            # Use all available CPU cores (or specify a number e.g., 4)
        )

    # --- Saving NPZ (Conditional) ---
    if args.save_npz:
        save_npz_start_time = time.perf_counter()
        depth_npz_path = os.path.join(args.output_dir, output_base_filename + '_depths.npz')
        print(f"Saving depths to NPZ file: {depth_npz_path}")
        try:
            np.savez_compressed(depth_npz_path, depths=depths)
        except Exception as e:
             print(f"Error saving depths to NPZ: {e}")
        save_npz_end_time = time.perf_counter()
        print(f"    Saving NPZ took: {save_npz_end_time - save_npz_start_time:.2f} seconds")

    # --- Saving EXR (Conditional) ---
    if args.save_exr:
        if not EXR_AVAILABLE:
            print("Skipping EXR saving because OpenEXR/Imath module is not available.")
        else:
            save_exr_start_time = time.perf_counter()
            depth_exr_dir = os.path.join(args.output_dir, output_base_filename + '_depths_exr')
            print(f"Saving depths to EXR files in folder: {depth_exr_dir}")
            try:
                os.makedirs(depth_exr_dir, exist_ok=True)
                # Ensure depth data is float32 for EXR saving
                if depths.dtype != np.float32:
                    print(f"    Converting depths from {depths.dtype} to float32 for EXR saving.")
                    depths_float32 = depths.astype(np.float32)
                else:
                    depths_float32 = depths

                num_exr_frames = len(depths_float32)
                zfill_width = len(str(num_exr_frames - 1))

                for i, depth in enumerate(depths_float32):
                    if depth.ndim != 2:
                        print(f"Warning: Skipping EXR save for frame {i}, unexpected shape {depth.shape}")
                        continue
                    # Use zfill for consistent naming
                    output_exr_filename = f"frame_{str(i).zfill(zfill_width)}.exr"
                    output_exr = os.path.join(depth_exr_dir, output_exr_filename)
                    try:
                        header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                        # Define the channel for single-channel float depth ('R' or 'Z' are common conventions)
                        header["channels"] = { 'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) }
                        exr_file = OpenEXR.OutputFile(output_exr, header)
                        # Write the pixel data correctly
                        exr_file.writePixels({'Z': depth.tobytes()})
                        exr_file.close()
                    except Exception as e_inner:
                        print(f"Error saving frame {i} to EXR {output_exr}: {e_inner}")
                        # Decide whether to break or continue
                        # break
            except Exception as e_outer:
                 print(f"Error during EXR saving setup or loop: {e_outer}")
            save_exr_end_time = time.perf_counter()
            print(f"    Saving EXR files took: {save_exr_end_time - save_exr_start_time:.2f} seconds")

    # overall_end_time = time.perf_counter() # End overall timer
    # print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")