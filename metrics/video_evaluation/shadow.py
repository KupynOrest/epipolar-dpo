from typing import Dict, Any, Tuple

from metrics.projective_geometry.shadow import ShadowPredictor, ShadowClassifier
from metrics.video_evaluation.base import BaseEvaluator


class ShadowEvaluator(BaseEvaluator):
    """
    Evaluator that analyzes shadow realism in videos.
    """

    def __init__(self, shadow_config_path: str, shadow_weights_path: str, classifier_path: str, sampling_rate: int =1):
        """
        Initialize the shadow evaluator.

        Args:
            shadow_config_path: Path to shadow detector config
            shadow_weights_path: Path to shadow detector weights
            classifier_path: Path to shadow realism classifier weights
            sampling_rate: Process every Nth frame (default: 1 = process all frames)
        """
        super().__init__(sampling_rate)
        self.shadow_predictor = ShadowPredictor(
            config_path=shadow_config_path,
            weights_path=shadow_weights_path
        )
        self.classifier = ShadowClassifier(
            model_path=classifier_path
        )

    def compute_metrics(self, frame):
        """
        Compute shadow realism metrics for a single frame.

        Args:
            frame: The video frame (numpy array in RGB format)

        Returns:
            Dictionary with metrics for this frame
        """
        # Get shadow predictions
        masks = self.shadow_predictor.predict(frame)
        if masks is None:
            return None

        classification = self.classifier.predict(
            object_mask=masks["object_mask"],
            shadow_mask=masks["shadow_mask"]
        )
        realism_score = classification['probabilities'][0]
        return {
            'realism_score': realism_score,
            'classification': classification['class'],
        }

    @property
    def name(self):
        """Return the name of this evaluator."""
        return "shadow_realism"

    def aggregate_metrics(self, frame_metrics) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate per-frame metrics into final metrics.

        Args:
            frame_metrics: List of dictionaries with per-frame metrics

        Returns:
            Dictionary with final aggregated metrics
        """
        if len(frame_metrics) == 0:
            return -1, {'mean_realism_score': -1, 'total_frames': 0, 'real_frame_ratio': 0}

        realism_scores = [m['realism_score'] for m in frame_metrics]
        mean_realism = float(sum(realism_scores) / len(realism_scores))

        # Count frames classified as real
        real_frames = sum(1 for m in frame_metrics if m['classification'] == 'real')

        result = {
            'mean_realism_score': mean_realism,
            'total_frames': len(frame_metrics),
            'real_frame_ratio': float(real_frames / len(frame_metrics))
        }

        return mean_realism, result

    @classmethod
    def from_config(cls, config):
        return cls(
            shadow_config_path=config.get('shadow_config_path'),
            shadow_weights_path=config.get('shadow_weights_path'),
            classifier_path=config.get('classifier_path'),
            sampling_rate=config.get('sampling_rate', 1)
        )
