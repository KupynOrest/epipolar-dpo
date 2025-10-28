from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import cv2
import torch
from abc import ABC, abstractmethod
from kornia.geometry.epipolar import find_fundamental, sampson_epipolar_distance
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

from metrics.video_evaluation.base import BaseEvaluator


class KeypointMatcher(ABC):
    """Abstract base class for keypoint detection and matching algorithms."""
    
    @abstractmethod
    def get_matched_points(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, Any]]:
        """
        Extract keypoints from two frames and return matched points.
        
        Args:
            frame1: First frame (numpy array)
            frame2: Second frame (numpy array)
            
        Returns:
            Tuple of (pts1, pts2, num_matches, metadata)
            pts1, pts2: Matched points coordinates (None if failed)
            num_matches: Number of successful matches
            metadata: Additional information about the matching process
        """
        pass


class SIFTMatcher(KeypointMatcher):
    """SIFT-based keypoint detection and matching."""
    
    def __init__(self, ratio_thresh: float = 0.75, min_matches: int = 20):
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches
        self.sift = cv2.SIFT_create()
    
    def detect_and_compute(self, frame: np.ndarray):
        """Detect SIFT keypoints and compute descriptors."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        kp, desc = self.sift.detectAndCompute(gray, None)
        return kp, desc
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray):
        """Match features using Lowe's ratio test."""
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def get_matched_points(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, Any]]:
        """Extract and match SIFT keypoints between two frames."""
        # Detect features
        kp1, desc1 = self.detect_and_compute(frame1)
        kp2, desc2 = self.detect_and_compute(frame2)
        
        metadata = {
            'keypoints1': len(kp1),
            'keypoints2': len(kp2),
            'descriptor_type': 'sift'
        }
        
        if len(kp1) < 8 or len(kp2) < 8:
            metadata['error'] = 'Not enough keypoints detected'
            return None, None, 0, metadata
        
        if desc1 is None or desc2 is None:
            metadata['error'] = 'Failed to compute descriptors'
            return None, None, 0, metadata
        
        # Match features
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < self.min_matches:
            metadata['error'] = f'Too few matches ({len(matches)}) - minimum {self.min_matches} required'
            return None, None, len(matches), metadata
        
        # Get matched point coordinates
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
        
        return pts1, pts2, len(matches), metadata


class LightGlueMatcher(KeypointMatcher):
    """LightGlue-based keypoint detection and matching."""
    
    def __init__(self, min_matches: int = 20):
        self.min_matches = min_matches
        self.processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
        self.model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")
    
    def get_matched_points(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, Any]]:
        """Extract and match LightGlue keypoints between two frames."""
        try:
            # Convert frames to PIL Images
            # Handle both BGR and RGB formats
            if len(frame1.shape) == 3:
                image1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
                image2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            else:
                image1 = Image.fromarray(frame1)
                image2 = Image.fromarray(frame2)
            
            # Process with LightGlue
            inputs = self.processor([image1, image2], return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            image_sizes = [[(image.height, image.width) for image in [image1, image2]]]
            results = self.processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
            
            if not results:
                return None, None, 0, {'error': 'No results from LightGlue', 'descriptor_type': 'lightglue'}
            
            result = results[0]  # First (and only) image pair
            num_matches = len(result["keypoints0"])
            
            metadata = {
                'descriptor_type': 'lightglue',
                'threshold': 0.2,
                'total_matches': num_matches
            }
            
            if num_matches < self.min_matches:
                metadata['error'] = f'Too few matches ({num_matches}) - minimum {self.min_matches} required'
                return None, None, num_matches, metadata
            
            # Convert to numpy arrays
            pts1 = result["keypoints0"].cpu().numpy()
            pts2 = result["keypoints1"].cpu().numpy()
            
            return pts1, pts2, num_matches, metadata
            
        except Exception as e:
            return None, None, 0, {'error': f'LightGlue processing failed: {str(e)}', 'descriptor_type': 'lightglue'}


class EpipolarEvaluator(BaseEvaluator):
    """
    Evaluator that analyzes 3D consistency between frames using epipolar geometry.
    Computes Sampson distance between matched features to quantify inconsistency.
    """

    def __init__(self, sampling_rate: int = 15, descriptor_type: str = "sift",
                 ratio_thresh: float = 0.75, ransac_thresh: float = 1.0, 
                 min_matches: int = 20):
        """
        Initialize the epipolar consistency evaluator.

        Args:
            sampling_rate: Process every Nth frame (default: 15)
            descriptor_type: Type of descriptor to use ("sift" or "lightglue")
            ratio_thresh: Threshold for Lowe's ratio test for SIFT matching
            ransac_thresh: RANSAC threshold for fundamental matrix estimation
            min_matches: Minimum number of matches required for valid estimation
        """
        super().__init__(sampling_rate)
        
        self.descriptor_type = descriptor_type
        self.ransac_thresh = ransac_thresh
        self.frames = []
        
        # Initialize the appropriate matcher
        if descriptor_type == "sift":
            self.matcher = SIFTMatcher(ratio_thresh=ratio_thresh, min_matches=min_matches)
        elif descriptor_type == "lightglue":
            self.matcher = LightGlueMatcher(min_matches=min_matches)
        else:
            raise ValueError(f"Unsupported descriptor type: {descriptor_type}. Choose 'sift' or 'lightglue'")

    def compute_fundamental_matrix(self, pts1: np.ndarray, pts2: np.ndarray):
        """Compute fundamental matrix using Kornia (handles normalization internally)."""
        try:
            # Convert to tensors for Kornia - shape (B, N, 2)
            points1_tensor = torch.from_numpy(pts1).float().unsqueeze(0)
            points2_tensor = torch.from_numpy(pts2).float().unsqueeze(0)
            
            # Kornia's find_fundamental handles normalization internally
            F_matrix = find_fundamental(points1_tensor, points2_tensor)
            
            if F_matrix is None or torch.isnan(F_matrix).any():
                return None, None, None
                
            return F_matrix, points1_tensor, points2_tensor
            
        except Exception:
            return None, None, None

    def compute_sampson_distances(self, F_matrix, points1_tensor, points2_tensor):
        """Compute Sampson distances using Kornia."""
        try:
            sampson_dist_squared = sampson_epipolar_distance(
                points1_tensor, points2_tensor, F_matrix, squared=True
            )
            
            sampson_dist = torch.sqrt(sampson_dist_squared + 1e-8).squeeze()
            return sampson_dist.cpu().numpy()
            
        except Exception:
            return None

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
        Compute epipolar geometry metrics for pairs of frames.

        Args:
            frame_pairs: List of frame pairs to compare

        Returns:
            List of dictionaries with metrics for each pair
        """
        results = []
        for frame1, frame2 in frame_pairs:
            result = self.compute_metric_for_pair(frame1, frame2)
            results.append(result)
        return results

    def compute_metric_for_pair(self, frame1: np.ndarray, frame2: np.ndarray):
        """
        Compute epipolar consistency metric between two frames.

        Args:
            frame1: First video frame (numpy array)
            frame2: Second video frame (numpy array)

        Returns:
            Dictionary with metric results
        """
        # Get matched points using the configured matcher
        pts1, pts2, num_matches, metadata = self.matcher.get_matched_points(frame1, frame2)
        
        base_result = {
            'num_matches': num_matches,
            'descriptor_type': self.descriptor_type,
            **metadata
        }
        
        if pts1 is None or pts2 is None:
            return {
                **base_result,
                'epipolar_error': None,
                'inlier_rate': None,
                'error': metadata.get('error', 'Failed to get matched points')
            }

        # Compute fundamental matrix using Kornia
        F_matrix, points1_tensor, points2_tensor = self.compute_fundamental_matrix(pts1, pts2)

        if F_matrix is None:
            return {
                **base_result,
                'epipolar_error': None,
                'inlier_rate': None,
                'error': 'Failed to compute fundamental matrix'
            }

        # Compute Sampson distances using Kornia
        sampson_distances = self.compute_sampson_distances(F_matrix, points1_tensor, points2_tensor)

        if sampson_distances is None:
            return {
                **base_result,
                'epipolar_error': None,
                'inlier_rate': None,
                'error': 'Failed to compute Sampson distances'
            }

        mean_sampson = np.mean(sampson_distances)
        
        # Calculate inlier rate (points within 5 pixels)
        inlier_threshold = 5.0
        inliers = sampson_distances <= inlier_threshold
        inlier_rate = np.mean(inliers)

        return {
            **base_result,
            'epipolar_error': mean_sampson,
            'inlier_rate': inlier_rate
        }

    def evaluate_video(self, video_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Override the base evaluate_video method to compute pairwise metrics every N frames.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (main_score, detailed_metrics_dict)
        """
        self.frames = []

        # Read video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened():
            return -1, {'error': f'Could not open video: {video_path}'}

        # Extract frames at the specified sampling rate
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sampling_rate == 0:
                self.frames.append(frame.copy())

            frame_idx += 1

        cap.release()

        # Create frame pairs (consecutive sampled frames)
        frame_pairs = []
        for i in range(len(self.frames) - 1):
            frame_pairs.append((self.frames[i], self.frames[i + 1]))

        # Compute metrics for all pairs
        if frame_pairs:
            pair_metrics = self.compute_metrics(frame_pairs)
            avg_score, result = self.aggregate_metrics(pair_metrics)

            result.update({
                'video_path': video_path,
                'original_fps': float(fps),
                'total_frames': total_frames,
                'sampling_rate': self.sampling_rate,
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
        return "epipolar_consistency"

    def aggregate_metrics(self, frame_metrics) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate per-pair metrics into final metrics.

        Args:
            frame_metrics: List of dictionaries with per-pair metrics

        Returns:
            Tuple of (main_score, detailed_metrics_dict)
        """
        if len(frame_metrics) == 0:
            return -1, {'mean_epipolar_error': -1, 'mean_inlier_rate': -1, 'total_pairs': 0}

        # Filter out None values and infinity values
        valid_metrics = [m for m in frame_metrics
                         if m.get('epipolar_error') is not None
                         and not np.isinf(m.get('epipolar_error', float('inf')))]

        if not valid_metrics:
            return -1, {'mean_epipolar_error': -1,
                        'mean_inlier_rate': -1,
                        'total_pairs': len(frame_metrics),
                        'valid_pairs': 0}

        # Extract metrics
        epipolar_errors = [m['epipolar_error'] for m in valid_metrics]
        match_counts = [m['num_matches'] for m in valid_metrics]
        inlier_rates = [m['inlier_rate'] for m in valid_metrics if m.get('inlier_rate') is not None]

        # Calculate means
        result = {
            'mean_epipolar_error': float(np.mean(epipolar_errors)),
            'mean_matches': float(np.mean(match_counts)),
            'mean_inlier_rate': float(np.mean(inlier_rates)) if inlier_rates else -1,
            'total_pairs': len(frame_metrics),
            'valid_pairs': len(valid_metrics),
        }

        return result['mean_epipolar_error'], result

    @classmethod
    def from_config(cls, config):
        return cls(
            sampling_rate=config.get('sampling_rate', 15),
            descriptor_type=config.get('descriptor_type', "sift"),
            ratio_thresh=config.get('ratio_thresh', 0.75),
            ransac_thresh=config.get('ransac_thresh', 1.0),
            min_matches=config.get('min_matches', 20)
        )
