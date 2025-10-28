import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Literal, Any
import logging
from collections import defaultdict
import numpy as np


MOTION_THRESHOLD = 0.9


class DPOLatentDataset(Dataset):
    """
    Dataset for DPO reward model training using latent pairs.
    Groups videos by original video path and creates a single pair
    of the best and worst samples for each group.
    """

    def __init__(
            self,
            metadata_path: str,
            metric_name: str,
            metric_mode: Literal["max", "min"] = "max",
            min_gap: float = 0.0,
            metric_threshold: float = None,
            filter_static: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            metadata_path: Path to metadata JSON file
            metric_name: Name of the metric to use (e.g., "shadow_realism")
            metric_mode: Whether higher ("max") or lower ("min") metric values are better
            min_gap: Minimum gap between winner and loser metrics
            metric_threshold: Threshold for winner metric quality
                            - For "min" mode: only keep pairs where winner < threshold 
                            - For "max" mode: only keep pairs where winner > threshold
            filter_static: Whether to filter out static videos
        """
        self.metadata_path = metadata_path
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.min_gap = min_gap
        self.metric_threshold = metric_threshold
        self.filter_static = filter_static
        self.pairs = self._load_and_organize_data()
        logging.info(f"Created dataset with {len(self.pairs)} pairs from {self.metadata_path}")

    def _validate_entry(self, entry: Dict[str, Any]) -> bool:
        if "condition_path" not in entry:
            return False
        dataset_source = entry.get("dataset_source", "")
        valid_datasets = ['DL3DV-10K', 'RealEstate10K']
        if dataset_source not in valid_datasets:
            return False
        if self.filter_static:
            if entry.get("motion_dynamics", 0.0) > MOTION_THRESHOLD:
                return False
        return True

    def _load_and_organize_data(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Load metadata and organize into best-worst pairs per group.

        Returns:
            List of (best_entry, worst_entry) tuples
        """
        
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        groups = defaultdict(list)
        
        dataset_sources = defaultdict(int)
        
        for entry in metadata:
            if not all(k in entry for k in ["original_video_path", "latent_path"]):
                continue

            dataset_source = entry.get("dataset_source", "")
            dataset_sources[dataset_source] += 1
            
            if not self._validate_entry(entry):
                continue

            metric_value = entry.get(self.metric_name, float('nan'))
            if np.isnan(metric_value) or metric_value < 0:
                continue

            groups[entry["original_video_path"]].append((entry, metric_value))

        logging.info(f"Filtering summary:")
        logging.info(f"  - Validation failed: {sum(dataset_sources.values()) - len(groups)}")
        logging.info(f"  - Valid entries: {sum(len(g) for g in groups.values())}")
        logging.info(f"  - Unique groups: {len(groups)}")
        logging.info(f"  - Dataset sources found: {dict(dataset_sources)}")

        all_pairs = []
        for group_entries in groups.values():
            if len(group_entries) < 2:
                continue

            sorted_entries = sorted(group_entries, key=lambda x: x[1], reverse=True)

            if self.metric_mode == "max":
                winner = sorted_entries[0][0]
                loser = sorted_entries[-1][0]
            else:
                winner = sorted_entries[-1][0]
                loser = sorted_entries[0][0]
            
            if self._is_valid_pair(winner, loser):
                all_pairs.append((winner, loser))

        return all_pairs

    def _is_valid_pair(self, winner: Dict[str, Any], loser: Dict[str, Any]) -> bool:
        winner_metric = winner[self.metric_name]
        loser_metric = loser[self.metric_name]
        metric_gap = abs(winner_metric - loser_metric)
        
        if metric_gap < self.min_gap:
            return False
        
        if self.metric_threshold is not None:
            if self.metric_mode == "min":
                if winner_metric >= self.metric_threshold:
                    return False
            else:
                if winner_metric <= self.metric_threshold:
                    return False
        
        return True

    def __len__(self) -> int:
        return len(self.pairs)

    def _get_metric_value(self, entry: Dict[str, Any]) -> float:
        return entry.get(self.metric_name, float('nan'))

    def _load_condition(self, condition_path: str) -> Dict[str, Any]:
        condition = torch.load(condition_path, map_location="cpu")
        return self._recursively_detach(condition)

    def _recursively_detach(self, item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            result = item.detach().clone()
            if result.size(0) == 1:
                result = result.squeeze(0)
            return result
        elif isinstance(item, dict):
            return {k: self._recursively_detach(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._recursively_detach(x) for x in item]
        elif isinstance(item, tuple):
            return tuple(self._recursively_detach(x) for x in item)
        else:
            return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a pair of win/lose items.

        Args:
            idx: Index of the pair

        Returns:
            Dictionary containing:
                - x_win: Winner latent (best sample)
                - x_lose: Loser latent (worst sample)
                - prompt_emb_win: Winner prompt embedding
                - prompt_emb_lose: Loser prompt embedding
                - image_emb_win: Winner image embedding (if exists and not empty)
                - image_emb_lose: Loser image embedding (if exists and not empty)
                - m_win: Winner metric value
                - m_lose: Loser metric value
        """
        winner, loser = self.pairs[idx]

        win_condition = self._load_condition(winner["condition_path"])
        lose_condition = self._load_condition(loser["condition_path"])
        
        win_has_img = "image_embedding" in win_condition and isinstance(win_condition.get("image_embedding"), dict) and len(win_condition["image_embedding"]) > 0
        lose_has_img = "image_embedding" in lose_condition and isinstance(lose_condition.get("image_embedding"), dict) and len(lose_condition["image_embedding"]) > 0

        x_win = torch.load(winner["latent_path"], map_location="cpu")
        x_lose = torch.load(loser["latent_path"], map_location="cpu")
        if x_win.dim() == 5 and x_win.size(0) == 1:
            x_win = x_win.squeeze(0)
        if x_lose.dim() == 5 and x_lose.size(0) == 1:
            x_lose = x_lose.squeeze(0)
        
        if 'prompt_emb' in win_condition:
            win_condition["prompt_embedding"] = win_condition["prompt_emb"]
        if 'prompt_emb' in lose_condition:
            lose_condition["prompt_embedding"] = lose_condition["prompt_emb"]
        
        prompt_emb_win = win_condition["prompt_embedding"]
        prompt_emb_lose = lose_condition["prompt_embedding"]
        
        m_win = self._get_metric_value(winner)
        m_lose = self._get_metric_value(loser)

        result = {
            "x_win": x_win,
            "x_lose": x_lose,
            "prompt_emb_win": prompt_emb_win,
            "prompt_emb_lose": prompt_emb_lose,
            "m_win": torch.tensor(m_win, dtype=torch.float32),
            "m_lose": torch.tensor(m_lose, dtype=torch.float32)
        }

        if win_has_img:
            result["image_emb_win"] = win_condition["image_embedding"]
        if lose_has_img:
            result["image_emb_lose"] = lose_condition["image_embedding"]
            
        return result
