from typing import Dict, Any, Tuple, List
import torch
import cv2
import numpy as np
from met3r import MEt3R

from metrics.video_evaluation.base import BaseEvaluator


def preprocess_frames(frames, max_size: int = 640):
    """
    Preprocess frames for MEt3R input:
    1. Resize to have maximum dimension of max_size while maintaining aspect ratio
    2. Pad to ensure dimensions are divisible by 32 (to handle DINO's even patch requirement)
    3. Normalize to [-1, 1] range

    Args:
        frames: List of frames as numpy arrays (BGR format)
        max_size: Maximum dimension size (default: 640)

    Returns:
        Tensor of shape (1, 2, 3, H, W) with pixel values in [-1, 1]
    """
    processed_frames = []

    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate resize dimensions while maintaining aspect ratio
        h, w = frame_rgb.shape[:2]
        scale = min(max_size / h, max_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        # Calculate padding to make dimensions divisible by 32
        # This ensures (H//16) and (W//16) will always be even numbers
        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32

        # Apply padding
        frame_padded = cv2.copyMakeBorder(
            frame_resized,
            0, pad_h, 0, pad_w,  # top, bottom, left, right
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # Verify dimensions are correct (H//16 and W//16 are even)
        h_padded, w_padded = frame_padded.shape[:2]
        assert (h_padded // 16) % 2 == 0, f"Padded height {h_padded} divided by 16 is not even"
        assert (w_padded // 16) % 2 == 0, f"Padded width {w_padded} divided by 16 is not even"

        # Convert to float and normalize to [-1, 1]
        frame_norm = (frame_padded.astype(np.float32) / 127.5) - 1.0

        # Convert to tensor and add channel dimension
        frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1)  # (3, H, W)
        processed_frames.append(frame_tensor)

    # Stack frames and add batch dimension
    stacked = torch.stack(processed_frames, dim=0)  # (2, 3, H, W)
    batched = stacked.unsqueeze(0)  # (1, 2, 3, H, W)

    return batched


class MEt3REvaluator(BaseEvaluator):
    """
    Evaluator that analyzes 3D consistency between frames using MEt3R.
    """

    def __init__(self, sampling_rate: int = 15, max_size: int = 640, device: str = "cuda"):
        """
        Initialize the MEt3R evaluator.

        Args:
            sampling_rate: Process every Nth frame (default: 15 = process every second of a 15fps video)
            max_size: Maximum size for frame dimension (default: 640)
            device: Device to run MEt3R on (default: "cuda")
        """
        super().__init__(sampling_rate)
        self.max_size = max_size
        self.device = device

        self.metric = MEt3R(
            img_size=None,
            use_norm=True,
            feat_backbone="dino16",
            featup_weights="mhamilton723/FeatUp",
            dust3r_weights="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
            use_mast3r_dust3r=True
        ).to(device)

        # Store extracted frames
        self.frames = []

    def process_frame(self, frame):
        """
        Store frame for later processing. Frames will be compared in pairs.

        Args:
            frame: The video frame (numpy array in BGR format)

        Returns:
            Empty dictionary (metrics are computed later in batches)
        """
        self.frames.append(frame.copy())
        return {}

    def compute_metrics(self, frame_pairs: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Compute MEt3R metrics for pairs of frames.

        Args:
            frame_pairs: List of frame pairs to compare

        Returns:
            List of dictionaries with metrics for each pair
        """
        results = []

        batch_size = 1
        for i in range(0, len(frame_pairs), batch_size):
            batch_pairs = frame_pairs[i:i + batch_size]
            batch_inputs = []

            for frame1, frame2 in batch_pairs:
                # Preprocess frame pair
                frames_input = preprocess_frames([frame1, frame2], self.max_size)
                batch_inputs.append(frames_input)

            if not batch_inputs:
                continue

            # Stack batch inputs
            stacked_inputs = torch.cat(batch_inputs, dim=0).to(self.device)

            with torch.no_grad():
                scores, *_ = self.metric(
                    images=stacked_inputs,
                    return_overlap_mask=False,
                    return_score_map=False,
                    return_projections=False
                )

            # Store results
            for j, score in enumerate(scores):
                results.append({
                    'met3r_score': score.item()
                })

        return results

    def evaluate_video(self, video_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Override the base evaluate_video method to compute pairwise metrics.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (main_score, detailed_metrics_dict)
        """
        self.frames = []  # Reset stored frames

        # Read video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adjust sampling rate if fps is very different from 15
        actual_sampling_rate = self.sampling_rate
        if fps > 0 and fps != 15:
            actual_sampling_rate = int(self.sampling_rate * (fps / 15))
            actual_sampling_rate = max(1, actual_sampling_rate)  # Ensure at least 1

        # Extract frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % actual_sampling_rate == 0:
                self.frames.append(frame.copy())

            frame_idx += 1

        cap.release()

        # Create frame pairs (1st and 2nd, 2nd and 3rd, etc.)
        frame_pairs = []
        for i in range(len(self.frames) - 1):
            frame_pairs.append((self.frames[i], self.frames[i + 1]))

        # Compute metrics for all pairs
        if frame_pairs:
            pair_metrics = self.compute_metrics(frame_pairs)

            # Aggregate metrics
            avg_score, result = self.aggregate_metrics(pair_metrics)

            # Add some additional metadata
            result.update({
                'original_fps': float(fps),
                'total_frames': total_frames,
                'sampling_rate': actual_sampling_rate,
                'sampled_frames': len(self.frames),
                'frame_pairs_evaluated': len(frame_pairs)
            })

            return avg_score, result
        else:
            print(f"Warning: Not enough frames extracted from {video_path}")
            return -1, {'error': 'Not enough frames for evaluation'}

    @property
    def name(self):
        """Return the name of this evaluator."""
        return "met3r_consistency"

    def aggregate_metrics(self, frame_metrics) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate per-pair metrics into final metrics.

        Args:
            frame_metrics: List of dictionaries with per-pair metrics

        Returns:
            Tuple of (main_score, detailed_metrics_dict)
        """
        if len(frame_metrics) == 0:
            return -1, {'mean_met3r_score': -1, 'total_pairs': 0}

        met3r_scores = [m['met3r_score'] for m in frame_metrics]
        mean_score = float(sum(met3r_scores) / len(met3r_scores))

        result = {
            'mean_met3r_score': mean_score,
            'total_pairs': len(frame_metrics),
        }

        return mean_score, result

    @classmethod
    def from_config(cls, config):
        return cls(
            sampling_rate=config.get('sampling_rate', 15),
            max_size=config.get('max_size', 640),
            device=config.get('device', 'cuda')
        )
