import os
import json
import glob
import numpy as np
from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging


def evaluate_videos(evaluators, video_paths, metadata=None):
    """Evaluate videos using all provided evaluators."""
    results = []

    for video_path in tqdm(video_paths, desc="Evaluating videos"):
        if not os.path.exists(video_path):
            logging.warning(f"Video not found: {video_path}")
            continue

        # Start with basic entry info
        entry_result = {
            "video_path": video_path,
            "video_name": os.path.basename(video_path)
        }

        # Add metadata if available
        if metadata:
            for entry in metadata:
                if entry.get("video_path") == video_path:
                    for key, value in entry.items():
                        if key != "video_path":  # avoid duplication
                            entry_result[key] = value
                    break

        # Run evaluators
        for evaluator in evaluators:
            try:
                evaluator_name = evaluator.name
                metric_value, detailed_metrics = evaluator.evaluate_video(video_path)
                entry_result[evaluator_name] = metric_value
                entry_result[f"{evaluator_name}_detailed"] = detailed_metrics
                logging.info(f"Video {os.path.basename(video_path)} - {evaluator_name}: {metric_value:.4f}")
            except Exception as e:
                logging.error(f"Error evaluating {video_path} with {evaluator.__class__.__name__}: {e}")
                entry_result[evaluator_name] = None
                entry_result[f"{evaluator_name}_detailed"] = {"error": str(e)}

        results.append(entry_result)

    # Calculate aggregated statistics for each evaluator
    stats = {}
    for evaluator in evaluators:
        evaluator_name = evaluator.name
        metric_values = [r.get(evaluator_name) for r in results if
                         evaluator_name in r and r[evaluator_name] is not None]

        if metric_values:
            stats[evaluator_name] = {
                "count": len(metric_values),
                "mean": float(np.mean(metric_values)),
                "median": float(np.median(metric_values)),
                "min": float(np.min(metric_values)),
                "max": float(np.max(metric_values)),
                "std": float(np.std(metric_values))
            }
        else:
            stats[evaluator_name] = {
                "count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "std": 0
            }

    return results, stats


def collect_videos(directory, pattern="**/*.mp4"):
    """Collect all video paths in a directory matching a pattern."""
    if not os.path.exists(directory):
        logging.error(f"Directory not found: {directory}")
        return []

    video_paths = glob.glob(os.path.join(directory, pattern), recursive=True)
    logging.info(f"Found {len(video_paths)} videos in {directory}")
    return video_paths


def filter_videos(baseline_videos, lora_videos):
    """Filter out videos that are in both baseline and LoRA directories."""
    lora_set = set(os.path.basename(video) for video in lora_videos)
    filtered_videos = [video for video in baseline_videos if os.path.basename(video) in lora_set]
    logging.info(f"Filtered {len(baseline_videos) - len(filtered_videos)} videos from baseline")
    return filtered_videos


@hydra.main(config_path="config", config_name="test", version_base=None)
def main(cfg: DictConfig):
    """Main function to run the evaluation script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    evaluators = [instantiate(evaluator_cfg) for evaluator_cfg in cfg.evaluators]
    logging.info(f"Initialized {len(evaluators)} evaluators")

    results = {}

    baseline_dir = os.path.join(cfg.output_dir, "baseline")
    logging.info(f"Evaluating baseline videos from {baseline_dir}")
    baseline_videos = collect_videos(baseline_dir)

    lora_dir = os.path.join(cfg.output_dir, "lora")
    lora_videos = collect_videos(lora_dir)

    baseline_videos = filter_videos(baseline_videos, lora_videos)

    baseline_results, baseline_stats = evaluate_videos(evaluators, baseline_videos)

    baseline_results_path = os.path.join(output_dir, "baseline_results.json")
    with open(baseline_results_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)

    results["baseline"] = baseline_stats
    logging.info(f"Baseline evaluation completed. Results saved to {baseline_results_path}")

    logging.info(f"Evaluating LoRA videos from {lora_dir}")

    lora_results, lora_stats = evaluate_videos(evaluators, lora_videos)

    lora_results_path = os.path.join(output_dir, "lora_results.json")
    with open(lora_results_path, 'w') as f:
        json.dump(lora_results, f, indent=2)

    results["lora"] = lora_stats
    logging.info(f"LoRA evaluation completed. Results saved to {lora_results_path}")

    if "baseline" in results and "lora" in results:
        logging.info("\n=== Comparative Results ===")
        for metric_name in results["baseline"]:
            if metric_name in results["lora"]:
                baseline_val = results["baseline"][metric_name]["mean"]
                lora_val = results["lora"][metric_name]["mean"]
                diff = lora_val - baseline_val
                rel_improvement = (diff / baseline_val) * 100 if baseline_val != 0 else float('inf')

                logging.info(f"{metric_name}:")
                logging.info(f"  Baseline: {baseline_val:.4f}")
                logging.info(f"  LoRA: {lora_val:.4f}")
                logging.info(f"  Absolute Diff: {diff:.4f}")
                logging.info(f"  Relative Improvement: {rel_improvement:.2f}%")

    # Save overall results
    results["parameters"] = OmegaConf.to_container(cfg, resolve=True)
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"All evaluation results saved to {results_path}")


if __name__ == "__main__":
    main()
