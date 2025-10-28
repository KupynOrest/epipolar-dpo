import torch
import cv2
import matplotlib.pyplot as plt


from metrics.projective_geometry.line_segment.deeplsd.models.deeplsd_inference import DeepLSD


class LinesPredictor:
    """
    A predictor class that processes images to detect line segments using DeepLSD.
    Returns line segments found in the image.
    """

    def __init__(self, checkpoint_path: str, device=None):
        """
        Initialize the LinesPredictor.

        Args:
            checkpoint_path: Path to the DeepLSD model weights
            device: Computation device ('cuda' or 'cpu'), will auto-detect if None
        """
        self.checkpoint_path = checkpoint_path

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Default configuration
        self.conf = {
            'detect_lines': True,  # Whether to detect lines or only DF/AF
            'line_detection_params': {
                'merge': False,  # Whether to merge close-by lines
                'filtering': True,  # Whether to filter out lines based on the DF/AF
                'grad_thresh': 3,
                'grad_nfa': True,  # Use image gradient and NFA score for thresholding
            }
        }

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the DeepLSD model and move it to the specified device."""
        # Load checkpoint
        ckpt = torch.load(str(self.checkpoint_path), weights_only=False, map_location='cpu')

        # Initialize model
        self.model = DeepLSD(self.conf)
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.to(self.device).eval()

    def predict(self, image):
        """
        Process an image and return detected line segments.

        Args:
            image: RGB or BGR image (numpy array) as loaded by cv2.imread

        Returns:
            dict: A dictionary containing 'lines' as numpy array of shape (N, 4)
                  where each row represents a line segment with [x1, y1, x2, y2]
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[:, :, 0]
        else:
            gray_image = image

        # Prepare input tensor
        inputs = {
            'image': torch.tensor(gray_image, dtype=torch.float, device=self.device)[None, None] / 255.
        }

        # Run inference
        with torch.no_grad():
            out = self.model(inputs)
            pred_lines = out['lines'][0]

        return {
            "lines": pred_lines
        }

    def visualize(self, image, predictions):
        """
        Visualize the predicted lines on the image.

        Args:
            image: BGR or RGB image from OpenCV
            predictions: Dictionary containing 'lines'

        Returns:
            Matplotlib figure with visualization
        """
        # Convert BGR to RGB for display if needed
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        else:
            # If grayscale, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Display the original image
        ax.imshow(image_rgb)

        # Plot lines
        lines = predictions["lines"]
        if lines is not None and len(lines) > 0:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line
                ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1.5)

        ax.set_title(f"Detected Lines ({len(lines)} lines)")
        ax.axis('off')

        return fig
