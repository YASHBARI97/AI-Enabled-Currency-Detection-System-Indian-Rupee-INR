"""Transfer-learning model factory (PyTorch/torchvision)."""
from __future__ import annotations


def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2):
    import torch.nn as nn
    from torchvision import models as tvm

    arch = arch.lower()
    if arch == "mobilenet_v3_small":
        w = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = tvm.mobilenet_v3_small(weights=w)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    elif arch == "mobilenet_v3_large":
        w = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = tvm.mobilenet_v3_large(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "efficientnet_b0":
        w = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = tvm.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[-1].in_features, num_classes))
    elif arch == "resnet18":
        w = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        m = tvm.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "small_cnn":
        m = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))
    else:
        raise ValueError(f"Unknown arch '{arch}'")
    return m
