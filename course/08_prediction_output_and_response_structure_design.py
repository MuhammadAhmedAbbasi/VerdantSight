"""Task 8: Prediction Output and Response Structure Design.

This file shows how a raw classifier prediction is converted into a structured
JSON response that another application can consume.

Usage:
    python course/08_prediction_output_and_response_structure_design.py --image path/to/test_leaf.jpg
"""

from pathlib import Path
import argparse
import json
import time

import torch
from PIL import Image

from common import ResNet9, basic_transform


DEFAULT_MODEL = Path("course/artifacts/deployment/pothos_classifier.pt")


class PredictionResponseService:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Deployment model not found: {model_path}\n"
                "Run Task 7 first or provide --model with a valid exported checkpoint."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        package = torch.load(model_path, map_location=self.device)

        self.model_version = package["model_version"]
        self.class_names = package["class_names"]

        self.model = ResNet9(number_of_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(package["model_state_dict"])
        self.model.eval()
        self.transform = basic_transform()

    def predict_response(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_id = probabilities.max(dim=0)

        processing_time_ms = (time.perf_counter() - start_time) * 1000.0

        response = {
            "model_version": self.model_version,
            "image": {
                "file_name": image_path.name,
                "width": original_width,
                "height": original_height,
            },
            "prediction": {
                "class_id": int(class_id.item()),
                "label": self.class_names[class_id.item()],
                "confidence": round(float(confidence.item()), 4),
            },
            "class_probabilities": {
                class_name: round(float(probabilities[index].item()), 4)
                for index, class_name in enumerate(self.class_names)
            },
            "processing_time_ms": round(processing_time_ms, 2),
        }

        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("course/artifacts/prediction_response.json"),
    )
    args = parser.parse_args()

    service = PredictionResponseService(args.model)
    response = service.predict_response(args.image)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(response, indent=2), encoding="utf-8")

    print("Prediction response")
    print("===================")
    print(json.dumps(response, indent=2))
    print(f"\nSaved JSON response to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
