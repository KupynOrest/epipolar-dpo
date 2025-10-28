from typing import Dict, Any, Tuple

from metrics.projective_geometry.line_segment import LineClassifier, LinesPredictor
from metrics.video_evaluation.base import BaseEvaluator
from metrics.video_evaluation.utils import resize_and_center_crop


class LinesEvaluator(BaseEvaluator):
    """
    Evaluator that analyzes line segment realism in videos.
    """

    def __init__(self, predictor_path: str, classifier_path: str, sampling_rate: int = 1):
        """
        Initialize the line segment evaluator.

        Args:
            predictor_path: Path to the line segment predictor model
            classifier_path: Path to line segment realism classifier weights
            sampling_rate: Process every Nth frame (default: 1 = process all frames)
        """
        super().__init__(sampling_rate)
        self.predictor = LinesPredictor(predictor_path)
        self.classifier = LineClassifier(classifier_path)

    def compute_metrics(self, frame):
        """
        Compute line segment realism metrics for a single frame.

        Args:
            frame: The video frame (numpy array in BGR format)

        Returns:
            Dictionary with metrics for this frame
        """
        processed_frame = resize_and_center_crop(frame)
        lines_prediction = self.predictor.predict(processed_frame)
        classification = self.classifier.predict(lines_prediction['lines'])
        realism_score = classification['probabilities'][0]

        line_count = len(lines_prediction['lines'])

        return {
            'realism_score': realism_score,
            'classification': classification['class'],
            'line_count': line_count,
        }

    @property
    def name(self):
        """Return the name of this evaluator."""
        return "lines_realism"

    def aggregate_metrics(self, frame_metrics) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate per-frame metrics into final metrics.

        Args:
            frame_metrics: List of dictionaries with per-frame metrics

        Returns:
            Tuple of (main_score, detailed_metrics_dict)
        """
        if len(frame_metrics) == 0:
            return -1, {'mean_realism_score': -1, 'total_frames': 0, 'real_frame_ratio': 0, 'avg_line_count': 0}

        realism_scores = [m['realism_score'] for m in frame_metrics]
        mean_realism = float(sum(realism_scores) / len(realism_scores))
        real_frames = sum(1 for m in frame_metrics if m['classification'] == 'real')
        avg_line_count = float(sum(m['line_count'] for m in frame_metrics) / len(frame_metrics))

        result = {
            'mean_realism_score': mean_realism,
            'total_frames': len(frame_metrics),
            'real_frame_ratio': float(real_frames / len(frame_metrics)),
            'avg_line_count': avg_line_count,
        }

        return mean_realism, result

    @classmethod
    def from_config(cls, config):
        return cls(
            predictor_path=config.get('predictor_path'),
            classifier_path=config.get('classifier_path'),
            sampling_rate=config.get('sampling_rate', 1)
        )
