"""Task 6: Model Optimization and Generalization

This file improves the Task 5 baseline using:
- data augmentation
- class-balanced sampling
- better learning rate
- more epochs
- best validation checkpoint
"""

from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
SPLIT_PATH = Path("course/artifacts/splits")
OUTPUT_PATH = Path("course/artifacts/optimized")

CLASS_NAMES = ["healthy", "chlorosis", "necrosis"]
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------------------------------------------------------------
# 2. Read dataset splits from Task 3
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
# 3. Dataset and preprocessing
# -----------------------------------------------------------------------------
def make_square(image):
    width, height = image.size
    size = max(width, height)

    left = (size - width) // 2
    top = (size - height) // 2
    right = size - width - left
    bottom = size - height - top

    return ImageOps.expand(image, (left, top, right, bottom), fill=(0, 0, 0))


# Training transform with augmentation
train_transform = transforms.Compose([
    transforms.Lambda(make_square),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
    transforms.ToTensor(),
])

# Validation and test images are not augmented
eval_transform = transforms.Compose([
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


# -----------------------------------------------------------------------------
# 4. Balance the classes using WeightedRandomSampler
# -----------------------------------------------------------------------------
class_counts = []

for class_id in range(len(CLASS_NAMES)):
    count = sum(item["label"] == class_id for item in train_data)
    class_counts.append(count)

print("Training class counts:")
for class_name, count in zip(CLASS_NAMES, class_counts):
    print(class_name, count)

sample_weights = []

for item in train_data:
    label = item["label"]
    weight = 1.0 / class_counts[label]
    sample_weights.append(weight)

sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float64),
    num_samples=len(sample_weights),
    replacement=True,
)


train_loader = DataLoader(
    LeafDataset(train_data, train_transform),
    batch_size=BATCH_SIZE,
    sampler=sampler,
)

val_loader = DataLoader(
    LeafDataset(val_data, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    LeafDataset(test_data, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# -----------------------------------------------------------------------------
# 5. ResNet9 model
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
# 6. Accuracy function
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
# 7. Train and save the best model
# -----------------------------------------------------------------------------
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_PATH / "optimized_model.pt"

best_val_accuracy = 0.0
best_epoch = 0

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

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        torch.save(model.state_dict(), MODEL_PATH)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


# -----------------------------------------------------------------------------
# 8. Load the best model and test it
# -----------------------------------------------------------------------------
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        predictions = model(images).argmax(dim=1).cpu()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

print("\nOptimized Model Results")
print("-----------------------")
print("Best validation epoch:", best_epoch)
print("Best validation accuracy:", round(best_val_accuracy, 4))
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nSaved optimized model to:", MODEL_PATH)
