import torch
import torch.nn as nn
import torchvision.models as models


class PerspectiveFieldClassifier:
    """
    A classifier that determines if a perspective field is realistic.
    """

    def __init__(self, model_path: str, device=None):
        """
        Initialize the perspective field classifier.

        Args:
            model_path: Path to the pre-trained model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.classes = ['real', 'generated']

    def _load_model(self, model_path):
        """Load the pre-trained model."""
        model = models.resnet50(weights=None)

        nr_filters = model.fc.in_features
        model.fc = nn.Linear(nr_filters, 2)

        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('resnet.'):
                new_key = key[len('resnet.'):]  # Strip the 'resnet.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)

        model = model.to(self.device)
        model.eval()

        return model

    def transform_maps(self, latitude_map, gravity_maps):
        """
        Transform perspective field maps for input to the model.

        Args:
            latitude_map: Latitude map tensor
            gravity_maps: Gravity maps tensor (3 channels)

        Returns:
            Transformed tensor ready for model input
        """
        latitude_map = latitude_map / 90.0
        joined_maps = torch.cat([latitude_map.unsqueeze(0), gravity_maps], dim=0)
        return joined_maps

    def predict(self, field):
        """
        Predict if the perspective field is realistic.

        Args:
            field: Dictionary containing 'pred_latitude_original' and 'pred_gravity_original'

        Returns:
            dict: Contains prediction class, probability, and class index
        """
        # Extract and transform the maps
        latitude_map = field['pred_latitude_original']
        gravity_maps = field['pred_gravity_original']
        joined_maps = self.transform_maps(latitude_map, gravity_maps)

        # Add batch dimension if not present
        if joined_maps.dim() == 3:
            joined_maps = joined_maps.unsqueeze(0)

        # Move input to device
        joined_maps = joined_maps.to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(joined_maps)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'class': self.classes[predicted_class],
            'class_index': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }
