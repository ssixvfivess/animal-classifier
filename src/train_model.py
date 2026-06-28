from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from model_utils import DEFAULT_DISPLAY_NAMES, build_resnet18, get_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "animal_photos"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "animal_group_classifier.pt"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "models" / "confusion_matrix.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an animal group image classifier.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder with class subfolders.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true", help="Train only the final classifier layer.")
    parser.add_argument("--no_pretrained", action="store_true", help="Do not use ImageNet pretrained weights.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics_path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--confusion_matrix_path", type=Path, default=DEFAULT_MATRIX_PATH)
    return parser.parse_args()


def check_dataset(data_dir: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_count = sum(1 for path in data_dir.rglob("*") if path.suffix.lower() in image_extensions)
    class_folders = [path for path in data_dir.iterdir() if path.is_dir()]

    if len(class_folders) < 2 or image_count == 0:
        raise RuntimeError(
            "Dataset is empty or has fewer than 2 classes.\n"
            "Put images into folders like:\n"
            "data/animal_photos/mammals/*.jpg\n"
            "data/animal_photos/birds/*.jpg\n"
            "data/animal_photos/fish/*.jpg\n"
        )


def make_loaders(data_dir: Path, image_size: int, batch_size: int, val_ratio: float, seed: int):
    base_dataset = ImageFolder(root=data_dir)
    train_dataset = ImageFolder(root=data_dir, transform=get_transforms(image_size=image_size, train=True))
    val_dataset = ImageFolder(root=data_dir, transform=get_transforms(image_size=image_size, train=False))

    total_size = len(base_dataset)
    val_size = max(1, int(total_size * val_ratio))
    train_size = total_size - val_size
    if train_size < 1:
        raise RuntimeError("Not enough images to create a train/validation split.")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(range(total_size), [train_size, val_size], generator=generator)

    train_dataset = torch.utils.data.Subset(train_dataset, train_subset.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return base_dataset.classes, train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, device: torch.device, train: bool):
    model.train(train)
    losses: List[float] = []
    all_true: List[int] = []
    all_pred: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        losses.append(float(loss.item()))
        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(predictions.cpu().numpy().tolist())

    accuracy = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return float(np.mean(losses)), float(accuracy), float(macro_f1), all_true, all_pred


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    check_dataset(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names, train_loader, val_loader = make_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = build_resnet18(
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)

    history = []
    best_state = None
    best_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1, y_true, y_pred = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_f1={train_f1:.4f} | val_f1={val_f1:.4f} | val_acc={val_acc:.4f}"
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    _, val_acc, val_f1, y_true, y_pred = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    display_names: Dict[str, str] = {name: DEFAULT_DISPLAY_NAMES.get(name, name) for name in class_names}
    torch.save(
        {
            "model_name": "resnet18_transfer_learning",
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "display_names": display_names,
            "image_size": args.image_size,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        },
        args.model_path,
    )

    metrics = {
        "classes": class_names,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "classification_report": report,
        "history": history,
        "model_path": str(args.model_path),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_confusion_matrix(cm, class_names, args.confusion_matrix_path)

    print("\nTraining finished")
    print(f"Classes: {class_names}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation macro-F1: {val_f1:.4f}")
    print(f"Saved model: {args.model_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
