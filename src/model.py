import torch
import torch.nn as nn
from torchvision import models
import timm


def get_model(num_classes=10):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_conv_layers(model):
    return [
        m for m in model.modules() if isinstance(m, nn.Conv2d) and m.out_channels > 1
    ]


def build_model(cfg: dict, mode: str = "full_finetune") -> nn.Module:
    """
    Mode declares the type of frozen:
        full_finetune -> nothing frozen
        linear_probe  -> entire backbone frozen
        fixed_freeze  -> early layer forzen
    """

    num_classes = cfg["dataset"]["num_classes"]
    model = timm.create_model(cfg["backbone"], pretrained=True, num_classes=num_classes)

    if mode == "linear_probe":
        for name, param in model.named_parameters():
            # Freeze everything; only the final head (fc/classifier) updates
            if not any(k in name for k in ["fc", "classifier", "head"]):
                param.requires_grad = False

    elif mode == "fixed_freeze":
        # Freeze the first two layer groups — generic low-level features
        # For ResNet: conv1, bn1, layer1, layer2
        freeze_prefixes = ["conv1", "bn1", "layer1", "layer2"]
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in freeze_prefixes):
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[{mode}] trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )

    return model
