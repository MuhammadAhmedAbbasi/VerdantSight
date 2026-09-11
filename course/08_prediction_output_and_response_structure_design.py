"""Task 8: Prediction Output and Response Structure Design

Change TEST_IMAGE_PATH to an unseen leaf image.
This file loads the exported model from Task 7 and creates a JSON response.
"""

from pathlib import Path
import json
import time

import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps


# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
MODEL_PATH = Path("course/artifacts/deployment/pothos_classifier.pt")
TEST_IMAGE_PATH = Path("test_leaf.jpg")
OUTPUT_PATH = Path("course/artifacts/prediction_response.json")

IMAGE_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------------------------------------------------------------
# 2. ResNet9 model definition
# -----------------------------------------------------------------------------
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

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, number_of_classes),
        )

    def forward(self, image):
        output = self.conv1(image)
        output = self.conv2(output)
        output = self.res1(output) + output

        output = self.conv3(output)
        output = self.conv4(output)
        output = self.res2(output) + output

        return self.classifier(output)


# -----------------------------------------------------------------------------
# 3. Load exported model and metadata
# -----------------------------------------------------------------------------
package = torch.load(MODEL_PATH, map_location=device)

CLASS_NAMES = package["class_names"]
MODEL_VERSION = package["model_version"]

model = ResNet9(number_of_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(package["model_state_dict"])
model.eval()


# -----------------------------------------------------------------------------
# 4. Prepare input image
# -----------------------------------------------------------------------------
def make_square(image):
    width, height = image.size
    size = max(width, height)

    left = (size - width) // 2
    top = (size - height) // 2
    right = size - width - left
    bottom = size - height - top

    return ImageOps.expand(image, (left, top, right, bottom), fill=(0, 0, 0))


transform = transforms.Compose([
    transforms.Lambda(make_square),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

image = Image.open(TEST_IMAGE_PATH).convert("RGB")
original_width, original_height = image.size
image_tensor = transform(image).unsqueeze(0).to(device)


# -----------------------------------------------------------------------------
# 5. Make prediction
# -----------------------------------------------------------------------------
start_time = time.perf_counter()

with torch.inference_mode():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)[0]
    confidence, class_id = probabilities.max(dim=0)

processing_time_ms = (time.perf_counter() - start_time) * 1000

predicted_label = CLASS_NAMES[class_id.item()]


# -----------------------------------------------------------------------------
# 6. Build a structured response
# -----------------------------------------------------------------------------
response = {
    "model_version": MODEL_VERSION,
    "image": {
        "file_name": TEST_IMAGE_PATH.name,
        "width": original_width,
        "height": original_height,
    },
    "prediction": {
        "class_id": class_id.item(),
        "label": predicted_label,
        "confidence": round(confidence.item(), 4),
    },
    "class_probabilities": {
        class_name: round(probabilities[index].item(), 4)
        for index, class_name in enumerate(CLASS_NAMES)
    },
    "processing_time_ms": round(processing_time_ms, 2),
}


# -----------------------------------------------------------------------------
# 7. Print and save the JSON response
# -----------------------------------------------------------------------------
print("\nPrediction Response")
print("-------------------")
print(json.dumps(response, indent=4))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(response, file, indent=4)

print("\nSaved response to:", OUTPUT_PATH)
