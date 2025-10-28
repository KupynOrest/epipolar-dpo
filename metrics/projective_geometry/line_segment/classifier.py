import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 256, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(256)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(128)

        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(128, self.output_dim*self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 256)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class BasePointNet(nn.Module):

    def __init__(self, point_dimension):
        super(BasePointNet, self).__init__()
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)

        self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        self.conv_3 = nn.Conv1d(64, 64, 1)
        self.conv_4 = nn.Conv1d(64, 128, 1)
        self.conv_5 = nn.Conv1d(128, 256, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(256)


    def forward(self, x, plot=False):
        num_points = x.shape[1]

        input_transform = self.input_transform(x) # T-Net tensor [batch, 3, 3]
        x = torch.bmm(x, input_transform) # Batch matrix-matrix product
        x = x.transpose(2, 1)
        tnet_out=x.cpu().detach().numpy()

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = x.transpose(2, 1)

        feature_transform = self.feature_transform(x) # T-Net tensor [batch, 64, 64]
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x, ix = nn.MaxPool1d(num_points, return_indices=True)(x)  # max-pooling
        x = x.view(-1, 256)  # global feature vector

        return x, feature_transform, tnet_out, ix


class ClassificationPointNet(nn.Module):

    def __init__(self, num_classes, dropout=0.3, point_dimension=2):
        super(ClassificationPointNet, self).__init__()
        self.base_pointnet = BasePointNet(point_dimension=point_dimension)

        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, num_classes)

        self.bn_1 = nn.BatchNorm1d(128)
        self.bn_2 = nn.BatchNorm1d(64)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x, feature_transform, tnet_out, ix_maxpool = self.base_pointnet(x)

        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout_1(x)
        x = self.fc_3(x)

        return x


class LineClassifier:
    """
    A classifier that determines if line segments are real or generated.
    Uses a PointNet architecture to process line segments as point clouds.
    """

    def __init__(self, model_path: str, device=None, number_of_lines=250):
        """
        Initialize the line classifier.

        Args:
            model_path: Path to the pre-trained model weights
            device: Device to run the model on ('cuda' or 'cpu')
            number_of_lines: Fixed number of lines to use as input (padding/sampling is applied)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.number_of_lines = number_of_lines
        self.model = self._load_model(model_path)
        self.classes = ['real', 'generated']

    def _load_model(self, model_path):
        """Load the pre-trained PointNet model."""
        model = ClassificationPointNet(num_classes=2, point_dimension=4)
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        model = model.to(self.device)
        model.eval()

        return model

    def preprocess_lines(self, lines):
        """
        Preprocess line segments for the PointNet model.

        Args:
            lines: Numpy array of shape (N, 4) where each row is [x1, y1, x2, y2]
                  or (N, 2, 2) where each element is [[x1, y1], [x2, y2]]

        Returns:
            Tensor of shape (1, number_of_lines, 4) ready for model input
        """
        # Reshape if needed - from (N, 2, 2) to (N, 4)
        if len(lines.shape) == 3 and lines.shape[1] == 2 and lines.shape[2] == 2:
            flattened_lines = lines.reshape(lines.shape[0], 4)
        else:
            flattened_lines = lines

        # Handle case where we have too few or too many lines
        line_disparity = self.number_of_lines - flattened_lines.shape[0]

        if line_disparity > 0:
            # Duplicate lines to reach required number
            sampling_indices = np.random.choice(flattened_lines.shape[0], line_disparity)
            new_lines = flattened_lines[sampling_indices, :]
            flattened_lines = np.concatenate((flattened_lines, new_lines), axis=0)
        elif line_disparity < 0:
            # Randomly sample lines to reduce to required number
            sampling_indices = np.random.choice(flattened_lines.shape[0], self.number_of_lines, replace=False)
            flattened_lines = flattened_lines[sampling_indices, :]

        # Convert to tensor and add batch dimension
        lines_tensor = torch.tensor(flattened_lines, dtype=torch.float32).unsqueeze(0)
        return lines_tensor

    def predict(self, lines):
        """
        Predict if the line segments are real or generated.

        Args:
            lines: Numpy array of line segments in format [x1, y1, x2, y2] or [[x1, y1], [x2, y2]]

        Returns:
            dict: Contains prediction class, confidence, and probabilities
        """
        # Preprocess lines
        processed_lines = self.preprocess_lines(lines)
        processed_lines = processed_lines.to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(processed_lines)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'class': self.classes[predicted_class],
            'class_index': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }

