from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEFAULT_DISPLAY_NAMES: Dict[str, str] = {
    "mammals": "Млекопитающее",
    "birds": "Птица",
    "fish": "Рыба",
    "reptiles": "Рептилия",
    "amphibians": "Амфибия",
    "insects": "Насекомое",
    "arachnids": "Паукообразное",
    "crustaceans": "Ракообразное",
}


def build_resnet18(num_classes: int, pretrained: bool = False, freeze_backbone: bool = False) -> nn.Module:
    """Create a ResNet18 classifier for the selected number of animal groups."""
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_transforms(image_size: int = 224, train: bool = False):
    """Return image transformations compatible with ImageNet-pretrained ResNet."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_checkpoint(model_path: str | Path, device: str | torch.device = "cpu") -> Tuple[nn.Module, List[str], Dict[str, str], int]:
    """Load a saved classifier checkpoint."""
    model_path = Path(model_path)
    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint["class_names"]
    display_names = checkpoint.get("display_names", DEFAULT_DISPLAY_NAMES)
    image_size = int(checkpoint.get("image_size", 224))

    model = build_resnet18(num_classes=len(class_names), pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, display_names, image_size


@torch.inference_mode()
def predict_image(
    image: Image.Image,
    model: nn.Module,
    class_names: List[str],
    display_names: Dict[str, str],
    image_size: int = 224,
    device: str | torch.device = "cpu",
):
    """Predict animal group from a PIL image."""
    image = image.convert("RGB")
    transform = get_transforms(image_size=image_size, train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    best_index = int(torch.argmax(probabilities).item())
    best_class = class_names[best_index]
    result = {
        "class_key": best_class,
        "class_name": display_names.get(best_class, best_class),
        "confidence": float(probabilities[best_index].item()),
        "probabilities": {
            display_names.get(class_names[i], class_names[i]): float(probabilities[i].item())
            for i in range(len(class_names))
        },
    }
    return result
