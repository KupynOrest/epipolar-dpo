from typing import Dict, Any, Tuple, List
import numpy as np
from skimage.metrics import structural_similarity as ssim

from metrics.video_evaluation.base import BaseEvaluator


class DynamicsEvaluator(BaseEvaluator):
    """
    Evaluator that analyzes motion dynamics in videos by measuring
    the dissimilarity between frames and the first frame.

    A higher score indicates more dynamic content (more motion/change).
    """

    def __init__(self, sampling_rate: int = 1, multichannel: bool = True):
        """
        Initialize the dynamics evaluator.

        Args:
            sampling_rate: Process every Nth frame (default: 1 = process all frames)
            multichannel: Whether to evaluate SSIM on multichannel (RGB) images
        """
        super().__init__(sampling_rate)
        self.first_frame = None
        self.multichannel = multichannel

    def compute_metrics(self, frame):
        """
        Compute dynamics metrics for a single frame compared to the first frame.

        Args:
            frame: The video frame (numpy array in RGB format)

        Returns:
            Dictionary with metrics for this frame
        """
        if self.first_frame is None:
            self.first_frame = frame
            return None

        if frame.shape != self.first_frame.shape:
            return None

        # Calculate SSIM between current frame and first frame
        # Higher SSIM means more similarity (less change)
        ssim_value = ssim(
            self.first_frame,
            frame,
            channel_axis=2 if self.multichannel else None,
            data_range=frame.max() - frame.min()
        )

        # Convert to dynamics score: 1 - SSIM
        # This way, higher scores mean more dynamics (motion)
        dynamics_score = 1.0 - ssim_value

        return {
            'ssim': ssim_value,
            'dynamics_score': dynamics_score
        }

    @property
    def name(self):
        """Return the name of this evaluator."""
        return "motion_dynamics"

    def aggregate_metrics(self, frame_metrics: List[Dict]) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate per-frame metrics into final metrics.

        Args:
            frame_metrics: List of dictionaries with per-frame metrics

        Returns:
            Tuple containing (main_score, all_metrics_dict)
        """
        if len(frame_metrics) == 0:
            return -1, {'mean_dynamics_score': -1, 'total_frames': 0}

        dynamics_scores = [m['dynamics_score'] for m in frame_metrics if m is not None]
        mean_dynamics = float(np.mean(dynamics_scores))
        min_dynamics = float(np.min(dynamics_scores))
        max_dynamics = float(np.max(dynamics_scores))

        result = {
            'mean_dynamics_score': mean_dynamics,
            'min_dynamics_score': min_dynamics,
            'max_dynamics_score': max_dynamics,
            'total_frames': len(frame_metrics)
        }
        return mean_dynamics, result

    @classmethod
    def from_config(cls, config):
        """Create an instance from configuration dictionary."""
        return cls(
            sampling_rate=config.get('sampling_rate', 1),
            multichannel=config.get('multichannel', True)
        )
