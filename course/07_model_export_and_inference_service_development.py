"""Task 7: Model Export and Inference Service Development

Change TEST_IMAGE_PATH to any unseen leaf image before running this file.
The optimized model should already exist from Task 6.
"""

from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps


# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
MODEL_PATH = Path("course/artifacts/optimized/optimized_model.pt")
EXPORT_PATH = Path("course/artifacts/deployment/pothos_classifier.pt")
TEST_IMAGE_PATH = Path("test_leaf.jpg")

CLASS_NAMES = ["healthy", "chlorosis", "necrosis"]
IMAGE_SIZE = 256
MODEL_VERSION = "resnet9-optimized-v1"

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
# 3. Export the trained model with useful metadata
# -----------------------------------------------------------------------------
trained_weights = torch.load(MODEL_PATH, map_location="cpu")

export_package = {
    "model_version": MODEL_VERSION,
    "class_names": CLASS_NAMES,
    "image_size": IMAGE_SIZE,
    "model_state_dict": trained_weights,
}

EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(export_package, EXPORT_PATH)

print("Exported deployment model to:", EXPORT_PATH)


# -----------------------------------------------------------------------------
# 4. Load the exported model for inference
# -----------------------------------------------------------------------------
package = torch.load(EXPORT_PATH, map_location=device)

model = ResNet9(number_of_classes=len(package["class_names"])).to(device)
model.load_state_dict(package["model_state_dict"])
model.eval()


# -----------------------------------------------------------------------------
# 5. Prepare one new image
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
image_tensor = transform(image).unsqueeze(0).to(device)


# -----------------------------------------------------------------------------
# 6. Run inference
# -----------------------------------------------------------------------------
with torch.inference_mode():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)[0]
    confidence, class_id = probabilities.max(dim=0)

predicted_label = CLASS_NAMES[class_id.item()]

print("\nInference Result")
print("----------------")
print("Predicted class:", predicted_label)
print("Confidence:", round(confidence.item(), 4))

print("\nClass probabilities:")
for index, class_name in enumerate(CLASS_NAMES):
    print(class_name, ":", round(probabilities[index].item(), 4))
