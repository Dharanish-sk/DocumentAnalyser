"""
CNN model for document fraud classification — Optimized V2.

Changes from V1:
  - FraudEfficientNetB3V2: outputs raw LOGITS (no Sigmoid) for BCEWithLogitsLoss
  - SE (Squeeze-and-Excitation) attention block in classifier head
  - BatchNorm between linear layers for training stability
  - GELU activation (smoother gradients than SiLU)
  - Scheduled dropout (decreasing through layers)
  - Backward-compatible: old models (FraudCNN, FraudEfficientNet, FraudEfficientNetB3) unchanged
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ImageNet normalization constants required for EfficientNet inference
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.
    
    Learns per-channel importance weights.  Cheap (~0.1% params) but
    consistently improves classification accuracy by 0.5-1.5%.
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        w = self.excite(x)  # (batch, features)
        return x * w


class FraudCNN(nn.Module):
    """
    Lightweight CNN for ELA-based fraud detection.
    Input: ELA image tensor of shape (batch, 3, 128, 128)
    Output: Probability of the document being tampered (sigmoid applied).
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
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
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class FraudEfficientNet(nn.Module):
    """EfficientNet-B0 based model (sigmoid output, backward compatible)."""

    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class FraudEfficientNetB3(nn.Module):
    """EfficientNet-B3 V1 (sigmoid output, backward compatible).
    Input: (batch, 3, 300, 300)   Output: (batch, 1) probability
    """

    def __init__(self, pretrained: bool = True, freeze_backbone_epochs: int = 0):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)
        in_features = self.backbone.classifier[1].in_features
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
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class FraudEfficientNetB3V2(nn.Module):
    """EfficientNet-B3 V2 — Optimized for fraud detection.

    Key improvements over V1:
      - Outputs raw LOGITS (no Sigmoid) for use with BCEWithLogitsLoss
        which is numerically more stable (log-sum-exp trick avoids underflow).
      - SE (Squeeze-and-Excitation) attention in classifier head learns
        which backbone features matter most for fraud detection.
      - BatchNorm between linear layers stabilizes training, especially
        with mixed precision (FP16) and aggressive augmentation.
      - GELU activation has smoother gradients than SiLU near zero,
        which helps with label-smoothed targets.
      - Wider hidden layer (768 vs 512) captures more complex ELA patterns.
      - Scheduled dropout: heavy at input (0.5), lighter deeper (0.15).

    Input:  (batch, 3, 300, 300)
    Output: (batch, 1) raw logits — apply torch.sigmoid() at inference time.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1536

        self.backbone.classifier = nn.Identity()  # Remove default head

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            SEBlock(768, reduction=16),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, 1),
            # NO Sigmoid — use BCEWithLogitsLoss for numerical stability
        )

        # Initialize head weights with small values for stable start
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freeze backbone feature extractor (train classifier head only)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (batch, 1536)
        logits = self.head(features)  # (batch, 1)
        return logits


def load_model(model_path: str = None, model_type: str = "efficientnet_b3_v2", device: str = "cpu") -> nn.Module:
    """
    Load a fraud detection model.

    Args:
        model_path: Path to saved model weights. If None, returns untrained model.
        model_type: 'efficientnet_b3_v2' (default, recommended),
                    'efficientnet_b3' (V1, backward compat),
                    'efficientnet' (B0), or 'cnn'.
        device: 'cpu' or 'cuda'.

    Returns:
        Loaded PyTorch model in eval mode.
    """
    if model_type == "efficientnet_b3_v2":
        model = FraudEfficientNetB3V2(pretrained=(model_path is None))
    elif model_type == "efficientnet_b3":
        model = FraudEfficientNetB3(pretrained=(model_path is None))
    elif model_type == "efficientnet":
        model = FraudEfficientNet()
    else:
        model = FraudCNN()

    if model_path:
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
