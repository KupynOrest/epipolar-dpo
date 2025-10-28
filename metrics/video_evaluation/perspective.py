from typing import Dict, Any, Tuple

from metrics.projective_geometry.perspective_field.perspective2d import PerspectiveFields
from metrics.projective_geometry.perspective_field.classifier import PerspectiveFieldClassifier
from metrics.video_evaluation.base import BaseEvaluator
from metrics.video_evaluation.utils import resize_and_center_crop


class PerspectiveEvaluator(BaseEvaluator):
    """
    Evaluator that analyzes perspective field realism in videos.
    """

    def __init__(self, classifier_path: str, sampling_rate: int = 1):
        """
        Initialize the perspective evaluator.

        Args:
            classifier_path: Path to perspective field realism classifier weights
            sampling_rate: Process every Nth frame (default: 1 = process all frames)
        """
        super().__init__(sampling_rate)
        self.pf_model = PerspectiveFields().eval().cuda()
        self.classifier = PerspectiveFieldClassifier(model_path=classifier_path)

    def compute_metrics(self, frame):
        """
        Compute perspective field realism metrics for a single frame.

        Args:
            frame: The video frame (numpy array in BGR format)

        Returns:
            Dictionary with metrics for this frame
        """
        processed_frame = resize_and_center_crop(frame)
        predictions = self.pf_model.inference(img_bgr=processed_frame)
        classification = self.classifier.predict(predictions)
        realism_score = classification['probabilities'][0]  # Probability of 'real'

        return {
            'realism_score': realism_score,
            'classification': classification['class'],
        }

    @property
    def name(self):
        """Return the name of this evaluator."""
        return "perspective_realism"

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
        real_frames = sum(1 for m in frame_metrics if m['classification'] == 'real')

        result = {
            'mean_realism_score': mean_realism,
            'total_frames': len(frame_metrics),
            'real_frame_ratio': float(real_frames / len(frame_metrics)),
        }

        return mean_realism, result

    @classmethod
    def from_config(cls, config):
        return cls(
            classifier_path=config.get('classifier_path'),
            sampling_rate=config.get('sampling_rate', 1)
        )
