"""Task 7: Model Export and Inference Service Development.

This file converts the optimized training checkpoint into a deployment checkpoint
with model metadata, then demonstrates a reusable inference service.

Usage:
    python course/07_model_export_and_inference_service_development.py --image path/to/test_leaf.jpg

Optional:
    --model course/artifacts/optimized/optimized_model.pt
    --export course/artifacts/deployment/pothos_classifier.pt
"""

from pathlib import Path
import argparse

import torch
from PIL import Image

from common import CLASS_NAMES, IMAGE_SIZE, ResNet9, basic_transform


MODEL_VERSION = "resnet9-optimized-v1"


def export_model(training_checkpoint: Path, export_path: Path):
    """Package trained weights with the metadata required for inference."""
    if not training_checkpoint.is_file():
        raise FileNotFoundError(
            f"Optimized model not found: {training_checkpoint}\n"
            "Run Task 6 first or provide --model with a valid checkpoint."
        )

    state_dict = torch.load(training_checkpoint, map_location="cpu")

    export_package = {
        "model_version": MODEL_VERSION,
        "architecture": "ResNet9",
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "model_state_dict": state_dict,
    }

    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_package, export_path)
    return export_path


class LeafHealthInferenceService:
    """Small inference service around the exported ResNet9 classifier."""

    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        package = torch.load(model_path, map_location=self.device)
        self.class_names = package["class_names"]
        self.model_version = package["model_version"]
        self.image_size = package["image_size"]

        self.model = ResNet9(number_of_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(package["model_state_dict"])
        self.model.eval()

        self.transform = basic_transform()

    def predict(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_id = probabilities.max(dim=0)

        return {
            "class_id": int(class_id.item()),
            "label": self.class_names[class_id.item()],
            "confidence": float(confidence.item()),
            "probabilities": {
                class_name: float(probabilities[index].item())
                for index, class_name in enumerate(self.class_names)
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("course/artifacts/optimized/optimized_model.pt"),
        help="Task 6 optimized state_dict checkpoint",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("course/artifacts/deployment/pothos_classifier.pt"),
        help="Deployment checkpoint to create",
    )
    parser.add_argument("--image", type=Path, required=True, help="Unseen leaf image for inference")
    args = parser.parse_args()

    exported_path = export_model(args.model, args.export)
    print(f"Exported deployment model: {exported_path.resolve()}")

    service = LeafHealthInferenceService(exported_path)
    prediction = service.predict(args.image)

    print("\nInference result")
    print("================")
    print(f"Model version: {service.model_version}")
    print(f"Predicted class: {prediction['label']}")
    print(f"Class ID:        {prediction['class_id']}")
    print(f"Confidence:      {prediction['confidence']:.4f}")
    print("Class probabilities:")
    for class_name, probability in prediction["probabilities"].items():
        print(f"  {class_name:10s}: {probability:.4f}")


if __name__ == "__main__":
    main()
