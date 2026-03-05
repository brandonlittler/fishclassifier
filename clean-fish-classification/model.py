# model.py - ONLY the class definition
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FishClassifier(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)