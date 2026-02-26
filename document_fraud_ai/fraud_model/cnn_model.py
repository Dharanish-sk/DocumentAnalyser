"""
CNN model for document fraud classification.

Architecture: A lightweight CNN that takes ELA-processed images
and classifies them as genuine or tampered.
Also supports using a pretrained EfficientNet backbone for
higher accuracy when fine-tuned on document datasets.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ImageNet normalization constants required for EfficientNet inference
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class FraudCNN(nn.Module):
    """
    Lightweight CNN for ELA-based fraud detection.

    Input: ELA image tensor of shape (batch, 3, 128, 128)
    Output: Probability of the document being tampered.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class FraudEfficientNet(nn.Module):
    """
    EfficientNet-B0 based model for higher accuracy fraud detection.
    Uses transfer learning with a pretrained backbone.
    """

    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class FraudEfficientNetB3(nn.Module):
    """EfficientNet-B3 with custom 2-layer fraud classification head.
    Input: (batch, 3, 300, 300)   Output: (batch, 1) fraud probability
    """

    def __init__(self, pretrained: bool = True, freeze_backbone_epochs: int = 0):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1536
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def freeze_backbone(self):
        """Freeze backbone feature extractor (train classifier head only)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_model(model_path: str = None, model_type: str = "efficientnet_b3", device: str = "cpu") -> nn.Module:
    """
    Load a fraud detection model.

    Args:
        model_path: Path to saved model weights. If None, returns untrained model.
        model_type: 'efficientnet_b3' (default), 'efficientnet' (B0), or 'cnn'.
        device: 'cpu' or 'cuda'.

    Returns:
        Loaded PyTorch model in eval mode.
    """
    if model_type == "efficientnet_b3":
        model = FraudEfficientNetB3(pretrained=(model_path is None))
    elif model_type == "efficientnet":
        model = FraudEfficientNet()
    else:
        model = FraudCNN()

    if model_path:
        # weights_only=True (safer) was added in PyTorch 2.4; fall back for 2.1-2.3
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
