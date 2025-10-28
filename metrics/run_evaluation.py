import os
import json
import logging
import glob
from datetime import datetime
import sys
from typing import Dict, List, Any

from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

MAX_TASKS = 20


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('evaluation.log')
        ]
    )


def load_metadata_files(metadata_folder: str) -> List[Dict[str, Any]]:
    """
    Load and merge all JSON metadata files from the folder.
    Filter duplicates based on timestamp.

    Args:
        metadata_folder: Path to folder containing metadata JSON files

    Returns:
        List of unique metadata entries
    """
    json_files = glob.glob(os.path.join(metadata_folder, "*.json"))
    logging.info(f"Found {len(json_files)} JSON files in {metadata_folder}")

    unique_entries = {}
    duplicate_count = 0

    for json_file in tqdm(json_files, desc="Loading metadata files"):
        try:
            with open(json_file, 'r') as f:
                metadata_list = json.load(f)

            for entry in metadata_list:
                video_path = entry.get("video_path", "")
                timestamp = entry.get("timestamp", "")

                if not video_path:
                    continue

                if video_path in unique_entries:
                    existing_timestamp = unique_entries[video_path].get("timestamp", "")
                    if timestamp and existing_timestamp:
                        try:
                            existing_dt = datetime.fromisoformat(existing_timestamp)
                            current_dt = datetime.fromisoformat(timestamp)

                            if current_dt > existing_dt:
                                unique_entries[video_path] = entry

                            duplicate_count += 1
                        except ValueError:
                            unique_entries[video_path] = entry
                else:
                    unique_entries[video_path] = entry

        except Exception as e:
            logging.error(f"Error processing {json_file}: {e}")

    logging.info(f"Loaded {len(unique_entries)} unique entries, filtered {duplicate_count} duplicates")
    return list(unique_entries.values())


def evaluate_video(video_path: str, evaluators: List) -> Dict[str, Any]:
    """
    Evaluate a video using all evaluators.

    Args:
        video_path: Path to video file
        evaluators: List of instantiated evaluator objects

    Returns:
        Dictionary with evaluation results
    """
    results = {}

    for evaluator in evaluators:
        metric_name = evaluator.name
        main_metric, metrics = evaluator.evaluate_video(video_path)
        results[metric_name] = main_metric
        results[f"{metric_name}_metrics"] = metrics
    return results


def get_file_prefix():
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        return "full"
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    return f"task_{task_id}"


def process_metadata(metadata_entries: List[Dict[str, Any]], evaluators: List,
                     save_folder: str, save_frequency: int = 1000) -> List[Dict[str, Any]]:
    """
    Process metadata entries and run evaluations for each video.
    Saves intermediate results every save_frequency entries.
    Supports restarting by loading previously processed entries.

    Args:
        metadata_entries: List of metadata entries
        evaluators: List of instantiated evaluator objects
        save_folder: Folder to save intermediate results
        save_frequency: Number of entries to process before saving intermediate results

    Returns:
        Updated metadata entries with evaluation results
    """
    os.makedirs(save_folder, exist_ok=True)
    previously_processed = {}
    try:
        previous_entries = load_metadata_files(save_folder)
        for entry in previous_entries:
            video_path = entry.get("video_path", "")
            if video_path:
                previously_processed[video_path] = entry
        logging.info(f"Loaded {len(previously_processed)} previously processed entries from {save_folder}")
    except Exception as e:
        logging.warning(f"Could not load previous progress: {e}")

    # Initialize updated_entries with previously processed entries that are in current metadata_entries
    updated_entries = []
    video_paths_to_process = set(entry.get("video_path", "") for entry in metadata_entries)

    for video_path, entry in previously_processed.items():
        if video_path in video_paths_to_process:
            updated_entries.append(entry)

    logging.info(f"Initialized with {len(updated_entries)} already processed entries")

    # Get subsets based on task ID
    metadata_entries = get_start_end_index(metadata_entries)

    # Keep track of what we've already processed to avoid duplicates
    processed_video_paths = set(entry.get("video_path", "") for entry in updated_entries)

    # Create progress bar with remaining count
    remaining_entries = [entry for entry in metadata_entries
                         if entry.get("video_path", "") not in processed_video_paths]

    logging.info(f"Processing {len(remaining_entries)} new entries out of {len(metadata_entries)} total")
    pbar = tqdm(total=len(remaining_entries), desc="Processing videos")
    for i, entry in enumerate(remaining_entries):
        video_path = entry.get("video_path", "")

        if not video_path or not os.path.exists(video_path):
            logging.warning(f"Video not found: {video_path}")
            pbar.update(1)
            continue
        if video_path in processed_video_paths:
            logging.debug(f"Skipping already processed video: {video_path}")
            pbar.update(1)
            continue

        results = evaluate_video(video_path, evaluators)
        entry.update(results)
        updated_entries.append(entry)
        processed_video_paths.add(video_path)
        pbar.update(1)

        # Save intermediate results
        if (i + 1) % save_frequency == 0:
            intermediate_path = os.path.join(save_folder, f"metadata_{get_file_prefix()}.json")
            logging.info(f"Saving intermediate results ({i + 1}/{len(remaining_entries)}) to {intermediate_path}")

            with open(intermediate_path, 'w') as f:
                json.dump(updated_entries, f, indent=2)

    pbar.close()
    return updated_entries


def get_start_end_index(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        print("SLURM_ARRAY_TASK_ID not found, processing all samples")
        return samples
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    num_in_one_bucket = len(samples) // MAX_TASKS
    start, end = task_id * num_in_one_bucket, min(len(samples), (task_id + 1) * num_in_one_bucket)
    print(f"Processing samples from {start} to {end} for task ID {task_id}")
    return samples[start:end]


@hydra.main(version_base=None, config_path="config", config_name="evaluators")
def main(cfg: DictConfig):
    """
    Main function to run evaluations.

    Args:
        cfg: Hydra configuration
    """
    setup_logging()
    metadata_folder = cfg.metadata_folder
    save_folder = cfg.save_folder
    save_frequency = cfg.get("save_frequency", 1000)

    evaluators = []
    for evaluator_cfg in cfg.evaluators:
        evaluator = instantiate(evaluator_cfg)
        evaluators.append(evaluator)
        logging.info(f"Initialized evaluator: {evaluator.__class__.__name__}")

    metadata_entries = load_metadata_files(metadata_folder)
    updated_entries = process_metadata(
        metadata_entries,
        evaluators,
        save_folder,
        save_frequency
    )

    os.makedirs(save_folder, exist_ok=True)
    output_path = os.path.join(save_folder, f"metadata_{get_file_prefix()}.json")

    with open(output_path, 'w') as f:
        json.dump(updated_entries, f, indent=2)

    logging.info(f"Saved {len(updated_entries)} entries to {output_path}")
    logging.info("Evaluation complete!")


if __name__ == "__main__":
    main()
