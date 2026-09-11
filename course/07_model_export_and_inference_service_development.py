"""Task 7: Model Export and Inference Service Development

Run after Task 6:
    python course/07_model_export_and_inference_service_development.py --image path/to/test_leaf.jpg
"""

from pathlib import Path
import argparse

import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision import transforms


CLASS_NAMES = ["healthy", "chlorosis", "necrosis"]
IMAGE_SIZE = 256
MODEL_VERSION = "resnet9-optimized-v1"


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


def export_model(training_checkpoint: Path, export_path: Path):
    if not training_checkpoint.is_file():
        raise FileNotFoundError(
            f"Optimized model not found: {training_checkpoint}\nRun Task 6 first."
        )

    export_package = {
        "model_version": MODEL_VERSION,
        "architecture": "ResNet9",
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "model_state_dict": torch.load(training_checkpoint, map_location="cpu"),
    }

    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_package, export_path)
    return export_path


class LeafHealthInferenceService:
    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        package = torch.load(model_path, map_location=self.device)

        self.class_names = package["class_names"]
        self.model_version = package["model_version"]
        self.model = ResNet9(len(self.class_names)).to(self.device)
        self.model.load_state_dict(package["model_state_dict"])
        self.model.eval()

    def predict(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_id = probabilities.max(dim=0)

        return {
            "class_id": int(class_id.item()),
            "label": self.class_names[class_id.item()],
            "confidence": float(confidence.item()),
            "probabilities": {
                name: float(probabilities[index].item())
                for index, name in enumerate(self.class_names)
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("course/artifacts/optimized/optimized_model.pt"))
    parser.add_argument("--export", type=Path, default=Path("course/artifacts/deployment/pothos_classifier.pt"))
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    exported_path = export_model(args.model, args.export)
    service = LeafHealthInferenceService(exported_path)
    prediction = service.predict(args.image)

    print(f"Exported deployment model: {exported_path.resolve()}")
    print("\nInference result")
    print("================")
    print(f"Model version:   {service.model_version}")
    print(f"Predicted class: {prediction['label']}")
    print(f"Class ID:        {prediction['class_id']}")
    print(f"Confidence:      {prediction['confidence']:.4f}")
    print("Class probabilities:")
    for name, probability in prediction["probabilities"].items():
        print(f"  {name:10s}: {probability:.4f}")


if __name__ == "__main__":
    main()
