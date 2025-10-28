import os
from os import environ
import random
from typing import List, Dict, Any

import cv2
from PIL import Image
import torch
import numpy as np
import pandas as pd
import json
from fire import Fire
from diffsynth import ModelManager, save_video

from wan.pipeline import WanVideoPipeline

MAX_TASKS = 100


def get_start_end_index(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if "SLURM_ARRAY_TASK_ID" not in environ:
        return samples
    task_id = int(environ["SLURM_ARRAY_TASK_ID"])
    num_in_one_bucket = len(samples) // MAX_TASKS
    start, end = task_id * num_in_one_bucket, min(len(samples), (task_id + 1) * num_in_one_bucket)
    return samples[start:end]


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All random seeds set to: {seed}")


def initialize_pipeline(model_path, image_to_video: bool = False):
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            f"{model_path}/diffusion_pytorch_model.safetensors",
            f"{model_path}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{model_path}/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )
    if image_to_video:
        model_manager.load_models(
            [
                f"{model_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ],
            torch_dtype=torch.float32,
        )

    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    return pipe


def load_captions_from_dataset(data_path: str):
    data_path = os.path.expanduser(data_path)
    print(f"Loading data from: {data_path}")

    if not os.path.exists(data_path):
        print(f"Error: Metadata file not found at {data_path}")
        return []

    try:
        with open(data_path, 'r') as f:
            metadata = json.load(f)
        print(f"Successfully loaded {len(metadata)} samples from dataset")
        metadata = sorted(metadata, key=lambda x: x.get('original_video_path', '')) 
    except Exception as e:
        print(f"Error loading or parsing JSON from {data_path}: {e}")
        return []

    return metadata


def save_video_and_latents(video_frames, latents, condition, output_path, latent_path, condition_path, fps=15, quality=5):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(latent_path), exist_ok=True)
    os.makedirs(os.path.dirname(condition_path), exist_ok=True)

    save_video(video_frames, output_path, fps=fps, quality=quality)
    print(f"Saved video to {output_path}")

    if latents is not None:
        if latents.requires_grad:
            latents = latents.detach()
        if latents.device.type != 'cpu':
            latents = latents.cpu()

        torch.save(latents, latent_path)
        torch.save(condition, condition_path)
        print(f"Saved latents to {latent_path}")
    else:
        print("No latents to save")


def save_metadata(metadata_list, output_dir) -> str:
    if "SLURM_ARRAY_TASK_ID" not in environ:
        metadata_path = os.path.join(output_dir, "metadata.json")
    else:
        task_id = int(environ["SLURM_ARRAY_TASK_ID"])
        metadata_path = os.path.join(output_dir, f"metadata_{task_id}.json")

    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2, default=str)
    return metadata_path


def load_existing_metadata(output_dir):
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        metadata_path = os.path.join(output_dir, f"metadata_{task_id}.json")
    else:
        metadata_path = os.path.join(output_dir, "metadata.json")

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            print(f"Found existing metadata file with {len(existing_metadata)} entries")
            return existing_metadata
        except Exception as e:
            print(f"Error reading existing metadata file: {e}")

    return []


def extract_first_frame(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None


def generate_video(pipe, generation_config, image_to_video=False, original_video_path=None):
    config = generation_config.copy()

    if image_to_video and original_video_path and os.path.exists(original_video_path):
        first_frame = extract_first_frame(original_video_path)

        if first_frame:
            max_area = 480 * 832
            aspect_ratio = first_frame.height / first_frame.width
            mod_value = 16
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            first_frame = first_frame.resize((width, height))

            config["input_image"] = first_frame
            config["width"] = width
            config["height"] = height
    elif not image_to_video:
        config["tea_cache_l1_thresh"] = 0.05
        config["tea_cache_model_id"] = "Wan2.1-T2V-1.3B"

    return pipe(**config)


def main(
        data_path: str,
        json_path: str,
        output_dir: str = "output_videos",
        model_path: str = "/scratch/shared/beegfs/okupyn/VideoModels/Wan2.1-T2V-1.3B",
        image_to_video: bool = False,
        num_inference_steps: int = 40,
        fps: int = 15,
        save_interval: int = 10,
):
    """
    Main function for generating videos

    Args:
        data_path: Path to the dataset
        json_path: Path to the JSON metadata file
        model_path: Path to the model files
        output_dir: Folder to save output videos
        image_to_video: Whether to use image-to-video mode
        num_inference_steps: Number of inference steps
        fps: Frames per second for output video
        save_interval: Save metadata every N generations (0 to save only at the end)
    """
    # Set all seeds for reproducibility
    seeds = [42, 123, 987, 1000]

    videos_dir = os.path.join(output_dir, "videos")
    latents_dir = os.path.join(output_dir, "latents")
    condition_dir = os.path.join(output_dir, "condition")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)
    os.makedirs(condition_dir, exist_ok=True)

    pipe = initialize_pipeline(model_path, image_to_video=image_to_video)
    samples = get_start_end_index(load_captions_from_dataset(json_path))

    if not samples:
        print("No samples loaded from dataset. Exiting.")
        return
    negative_prompt = "static view, frozen image, still frame, jump cuts, rapid scene changes, abrupt transitions, temporal discontinuity, quick cuts, scene jumping"

    generated_metadata = load_existing_metadata(output_dir)

    for idx, sample in enumerate(samples):
        long_caption = sample.get('long_caption', '')
        short_caption = sample.get('original_short_caption', '')
        caption = sample.get('caption', short_caption)
        caption = caption.replace('\u201c', '"')
        caption = caption.replace('\u201d', '"')

        vtss_score = sample.get('vtss_score', 'N/A')
        dataset_source = sample.get('dataset_source', 'unknown')
        original_video_path = sample.get('original_video_path', f"unknown_{idx}")

        if dataset_source not in ['DL3DV-10K', 'RealEstate10K']:
            continue

        if not caption:
            print(f"Skipping sample {idx + 1} - no caption available")
            continue

        original_filename = os.path.basename(original_video_path)
        base_name = os.path.splitext(original_filename)[0]

        num_frames = 65

        print(f"\nGenerating video {idx + 1}/{len(samples)}")
        print(f"Original video: {original_video_path}")
        print(f"VTSS Score: {vtss_score}")
        print(f"Using caption: {caption}")
        for seed in seeds:
            set_all_seeds(seed)
            video_path = os.path.join(videos_dir, f"{base_name}_{seed}.mp4")
            latent_path = os.path.join(latents_dir, f"{base_name}_{seed}.pt")
            condition_path = os.path.join(condition_dir, f"{base_name}_{seed}.pt")
            if os.path.exists(video_path):
                continue
            generation_config = {
                "prompt": caption,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "tiled": True,
                "width": 832,
                "height": 480,
                "num_frames": num_frames,
            }
            full_video_path = os.path.join(data_path, original_video_path)
            try:
                result = generate_video(
                    pipe=pipe,
                    generation_config=generation_config,
                    image_to_video=image_to_video,
                    original_video_path=full_video_path
                )

                if isinstance(result, tuple) and len(result) == 3:
                    video_frames, latents, condition = result
                else:
                    video_frames = result
                    latents = None
                    condition = None
                    print("Warning: Could not obtain latents from pipeline")

                if latents is not None:
                    save_video_and_latents(video_frames, latents, condition, video_path, latent_path, condition_path, fps=fps, quality=5)
                else:
                    save_video(video_frames, video_path, fps=fps, quality=5)
                    print(f"Saved video to {video_path}")

                metadata_entry = {
                    "original_video_path": original_video_path,
                    "dataset_source": dataset_source,
                    "short_caption": short_caption,
                    "long_caption": long_caption,
                    "vtss_score": vtss_score,
                    "caption": caption,
                    "video_path": video_path,
                    "latent_path": latent_path if latents is not None else None,
                    "condition_path": condition_path if condition is not None else None,
                    "num_frames": num_frames,
                    "seed": seed,
                    "fps": fps,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
                generated_metadata.append(metadata_entry)
            except Exception as e:
                print(f"Error generating video for {original_video_path}: {e}")
                continue

            if save_interval > 0 and (idx + 1) % save_interval == 0:
                save_metadata(generated_metadata, output_dir)

    if generated_metadata:
        save_metadata(generated_metadata, output_dir)

    print(f"\nAll {len(generated_metadata)} videos generated successfully!")


if __name__ == "__main__":
    Fire(main)
