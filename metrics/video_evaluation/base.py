import os
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

import cv2
from tqdm import tqdm


class BaseEvaluator(ABC):
    """
    Abstract base class for video evaluation.
    """

    def __init__(self, sampling_rate: int = 1):
        """
        Initialize the evaluator.

        Args:
            sampling_rate: Process every Nth frame (default: 1 = process all frames)
        """
        self.sampling_rate = sampling_rate

    @classmethod
    def from_config(cls, config):
        """Create an evaluator from a configuration dictionary."""
        pass

    def evaluate_video(self, video_path):
        """
        Evaluate a video file by processing frames and computing metrics.

        Args:
            video_path: Path to the MP4 video file

        Returns:
            Dictionary with evaluation metrics
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video info
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {os.path.basename(video_path)}")
        print(f"Dimensions: {width}x{height}, {frame_count} frames, {fps} fps")
        print(f"Processing every {self.sampling_rate} frame(s)")

        # Initialize metrics storage
        frame_metrics = []

        # Process frames
        frame_idx = 0
        pbar = tqdm(total=frame_count)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames according to sampling rate
            if frame_idx % self.sampling_rate == 0:
                # Compute metrics for this frame
                try:
                    metrics = self.compute_metrics(frame)
                    if metrics is not None:
                        frame_metrics.append(metrics)
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    continue

            frame_idx += 1
            pbar.update(1)

        # Release resources
        cap.release()
        pbar.close()

        main_metric, all_metrics = self.aggregate_metrics(frame_metrics)

        return main_metric, all_metrics

    @abstractmethod
    def compute_metrics(self, frame):
        """
        Compute metrics for a single frame.

        Args:
            frame: The video frame (numpy array in BGR format)

        Returns:
            Dictionary with metrics for this frame
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """Return the name of this evaluator."""
        pass

    @abstractmethod
    def aggregate_metrics(self, frame_metrics) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate per-frame metrics into final metrics.

        Args:
            frame_metrics: List of dictionaries with per-frame metrics

        Returns:
            Dictionary with final aggregated metrics
        """
        pass
