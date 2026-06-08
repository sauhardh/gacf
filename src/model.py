import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=10):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_conv_layers(model):
    return [
        m for m in model.modules() if isinstance(m, nn.Conv2d) and m.out_channels > 1
    ]
