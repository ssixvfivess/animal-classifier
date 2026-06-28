from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from model_utils import load_checkpoint, predict_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "animal_group_classifier.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict animal group from image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, display_names, image_size = load_checkpoint(args.model_path, device=device)
    image = Image.open(args.image_path)
    result = predict_image(image, model, class_names, display_names, image_size=image_size, device=device)

    print(f"Класс: {result['class_name']}")
    print(f"Уверенность: {result['confidence']:.2%}")
    print("Вероятности:")
    for class_name, probability in sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True):
        print(f"  {class_name}: {probability:.2%}")


if __name__ == "__main__":
    main()
