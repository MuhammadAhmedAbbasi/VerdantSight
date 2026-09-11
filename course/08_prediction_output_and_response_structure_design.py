"""Task 8: Prediction Output and Response Structure Design

Run after Task 7:
    python course/08_prediction_output_and_response_structure_design.py --image path/to/test_leaf.jpg
"""

from pathlib import Path
import argparse
import json
import time

import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision import transforms


IMAGE_SIZE = 256
DEFAULT_MODEL = Path("course/artifacts/deployment/pothos_classifier.pt")


def make_square(image):
    width, height = image.size
    size = max(width, height)
    left = (size - width) // 2
    top = (size - height) // 2
    right = size - width - left
    bottom = size - height - top
    return ImageOps.expand(image, (left, top, right, bottom), fill=(0, 0, 0))


INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Lambda(make_square),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def block(input_channels, output_channels, pool=False):
    layers = [
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, number_of_classes=3):
        super().__init__()
        self.conv1 = block(3, 64)
        self.conv2 = block(64, 128, pool=True)
        self.res1 = nn.Sequential(block(128, 128), block(128, 128))
        self.conv3 = block(128, 256, pool=True)
        self.conv4 = block(256, 512, pool=True)
        self.res2 = nn.Sequential(block(512, 512), block(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, number_of_classes))

    def forward(self, image):
        output = self.conv2(self.conv1(image))
        output = self.res1(output) + output
        output = self.conv4(self.conv3(output))
        output = self.res2(output) + output
        return self.classifier(output)


class PredictionResponseService:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Deployment model not found: {model_path}\nRun Task 7 first."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        package = torch.load(model_path, map_location=self.device)

        self.model_version = package["model_version"]
        self.class_names = package["class_names"]

        self.model = ResNet9(len(self.class_names)).to(self.device)
        self.model.load_state_dict(package["model_state_dict"])
        self.model.eval()

    def predict_response(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_id = probabilities.max(dim=0)
        processing_time_ms = (time.perf_counter() - start) * 1000.0

        return {
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
                name: round(float(probabilities[index].item()), 4)
                for index, name in enumerate(self.class_names)
            },
            "processing_time_ms": round(processing_time_ms, 2),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("course/artifacts/prediction_response.json"))
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
