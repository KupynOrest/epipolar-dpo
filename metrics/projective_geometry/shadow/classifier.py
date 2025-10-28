import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ShadowClassifier:
    """
    A classifier that determines if an object-shadow pair is realistic.
    """

    def __init__(self, model_path, device=None):
        """
        Initialize the classifier.

        Args:
            model_path: Path to the pre-trained model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        self.classes = ['real', 'generated']

    def _load_model(self, model_path):
        """Load the pre-trained model."""
        model = models.resnet50(weights=None)

        model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Error loading model weights: {e}")

        model = model.to(self.device)
        model.eval()

        return model

    def _preprocess_masks(self, object_mask, shadow_mask):
        """
        Preprocess object and shadow masks for input to the model.

        Args:
            object_mask: Grayscale object mask (numpy array)
            shadow_mask: Grayscale shadow mask (numpy array)

        Returns:
            Tensor ready for model input
        """
        object_mask = (object_mask > 0).astype(np.float32)
        shadow_mask = (shadow_mask > 0).astype(np.float32)

        object_img = Image.fromarray(object_mask)
        shadow_img = Image.fromarray(shadow_mask)

        object_tensor = self.transform(object_img)
        shadow_tensor = self.transform(shadow_img)
        combined_tensor = torch.cat([shadow_tensor, object_tensor], dim=0)
        combined_tensor = combined_tensor.unsqueeze(0)

        return combined_tensor

    def predict(self, object_mask, shadow_mask):
        """
        Predict if the object-shadow pair is realistic.

        Args:
            object_mask: Grayscale object mask (numpy array)
            shadow_mask: Grayscale shadow mask (numpy array)

        Returns:
            dict: Contains prediction class, probability, and class index
        """
        # Preprocess masks
        input_tensor = self._preprocess_masks(object_mask, shadow_mask)

        # Move input to device
        input_tensor = input_tensor.to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'class': self.classes[predicted_class],
            'class_index': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }

    def evaluate_batch(self, object_masks, shadow_masks):
        """
        Evaluate a batch of object-shadow pairs.

        Args:
            object_masks: List of object masks
            shadow_masks: List of shadow masks

        Returns:
            List of prediction dictionaries
        """
        results = []

        for object_mask, shadow_mask in zip(object_masks, shadow_masks):
            prediction = self.predict(object_mask, shadow_mask)
            results.append(prediction)

        return results
