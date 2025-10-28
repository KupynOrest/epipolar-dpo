import cv2
import torch
import numpy as np
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger

from .ssis.config import _C


class ShadowPredictor:
    """
    A predictor class that processes images to identify objects and their shadows.
    Returns separate grayscale masks for objects and shadows.
    """

    def __init__(self, config_path: str, weights_path: str, confidence_threshold: float = 0.1):
        """
        Initialize the ShadowPredictor.

        Args:
            config_path: Path to the Detectron2 config file
            weights_path: Path to the model weights
            confidence_threshold: Threshold for detection confidence (default: 0.1)
        """
        self.logger = setup_logger()
        self.cfg = self._setup_config(config_path, weights_path, confidence_threshold)
        self.predictor = DefaultPredictor(self.cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_config(self, config_path, weights_path, confidence_threshold):
        """Set up the configuration for the predictor."""
        cfg = self._get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path

        # Set confidence thresholds
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
        cfg.MODEL.MEInst.INFERENCE_TH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold

        cfg.freeze()
        return cfg

    def _get_cfg(self):
        """Get a copy of the default config."""
        return _C.clone()

    def predict(self, image):
        """
        Process an image and return object and shadow masks.

        Args:
            image: BGR image (numpy array) as loaded by cv2.imread

        Returns:
            dict: A dictionary containing 'object_mask' and 'shadow_mask' as grayscale images
        """
        predictions = self.predictor(image)[0]

        instances = predictions["instances"]
        if instances is None:
            return None
        instances = instances.to("cpu")
        if not instances.has("pred_masks") or len(instances) == 0:
            h, w = image.shape[:2]
            return {
                "object_mask": np.zeros((h, w), dtype=np.uint8),
                "shadow_mask": np.zeros((h, w), dtype=np.uint8)
            }

        masks = instances.pred_masks.numpy()
        classes = instances.pred_classes.numpy()

        h, w = image.shape[:2]
        object_mask = np.zeros((h, w), dtype=np.uint8)
        shadow_mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(len(masks)):
            mask = masks[i]
            class_id = int(classes[i])

            if class_id == 0:
                object_mask = np.logical_or(object_mask, mask).astype(np.uint8) * 255
            elif class_id == 1:
                shadow_mask = np.logical_or(shadow_mask, mask).astype(np.uint8) * 255

        return {
            "object_mask": object_mask,
            "shadow_mask": shadow_mask
        }

    def visualize(self, image, masks):
        """
        Visualize the predictions on the image.

        Args:
            image: BGR image from OpenCV
            masks: Dictionary containing 'object_mask' and 'shadow_mask'

        Returns:
            Matplotlib figure with visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Display the original image
        ax.imshow(image_rgb)

        # Define colors for object (class 0) and shadow (class 1)
        colors = [
            (0, 0.7, 0, 0.5),  # Green for objects
            (0.7, 0, 0, 0.5)  # Red for shadows
        ]

        # Create mask overlays
        h, w = image.shape[:2]

        # Object mask overlay
        if masks["object_mask"] is not None and np.any(masks["object_mask"]):
            mask_rgba = np.zeros((h, w, 4))
            mask_rgba[masks["object_mask"] > 0, :] = colors[0]
            ax.imshow(mask_rgba)

        # Shadow mask overlay
        if masks["shadow_mask"] is not None and np.any(masks["shadow_mask"]):
            mask_rgba = np.zeros((h, w, 4))
            mask_rgba[masks["shadow_mask"] > 0, :] = colors[1]
            ax.imshow(mask_rgba)

        # Add legend
        legend_elements = [
            Patch(facecolor=colors[0], label='Object'),
            Patch(facecolor=colors[1], label='Shadow')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title("Objects and Shadows")
        ax.axis('off')

        return fig
