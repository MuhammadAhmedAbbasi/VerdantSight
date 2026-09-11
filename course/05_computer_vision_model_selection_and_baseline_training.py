"""Task 5: Computer Vision Model Selection and Baseline Training

This is the intentionally simple baseline model.
Change DATASET_PATH if your dataset is stored somewhere else.
"""

from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
DATASET_PATH = Path("dataset")
SPLIT_PATH = Path("course/artifacts/splits")
OUTPUT_PATH = Path("course/artifacts/baseline")

CLASS_NAMES = ["healthy", "chlorosis", "necrosis"]
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01   # intentionally aggressive for the baseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------------------------------------------------------------
# 2. Read CSV split files created in Task 3
# -----------------------------------------------------------------------------
def read_csv(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["label"] = int(row["label"])
            data.append(row)
    return data


train_data = read_csv(SPLIT_PATH / "train.csv")
val_data = read_csv(SPLIT_PATH / "val.csv")
test_data = read_csv(SPLIT_PATH / "test.csv")


# -----------------------------------------------------------------------------
# 3. Dataset class and simple preprocessing
#    No augmentation is used in the baseline.
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


class LeafDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]["image_path"]
        label = self.data[index]["label"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


train_loader = DataLoader(
    LeafDataset(train_data, transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    LeafDataset(val_data, transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    LeafDataset(test_data, transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# -----------------------------------------------------------------------------
# 4. ResNet9 model
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


model = ResNet9(number_of_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -----------------------------------------------------------------------------
# 5. Accuracy function
# -----------------------------------------------------------------------------
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


# -----------------------------------------------------------------------------
# 6. Baseline training
# -----------------------------------------------------------------------------
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = F.cross_entropy(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_accuracy = calculate_accuracy(model, train_loader)
    val_accuracy = calculate_accuracy(model, val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


# -----------------------------------------------------------------------------
# 7. Save baseline model
# -----------------------------------------------------------------------------
MODEL_PATH = OUTPUT_PATH / "baseline_model.pt"
torch.save(model.state_dict(), MODEL_PATH)
print("\nSaved baseline model to:", MODEL_PATH)


# -----------------------------------------------------------------------------
# 8. Test the baseline model
# -----------------------------------------------------------------------------
y_true = []
y_pred = []

model.eval()

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        predictions = model(images).argmax(dim=1).cpu()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

print("\nBaseline Test Results")
print("---------------------")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
