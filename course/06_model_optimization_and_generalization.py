"""Task 6: Model Optimization and Generalization

This task improves the baseline using:
- data augmentation
- class-balanced sampling
- Adam learning rate 0.001
- 20 epochs
- best validation checkpoint selection

Run:
    python course/06_model_optimization_and_generalization.py --dataset path/to/dataset
"""

from pathlib import Path
import argparse
import csv
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageOps
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix


CLASS_NAMES = ["healthy", "chlorosis", "necrosis"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 256
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


def set_seed():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def make_square(image):
    width, height = image.size
    size = max(width, height)
    left = (size - width) // 2
    top = (size - height) // 2
    right = size - width - left
    bottom = size - height - top
    return ImageOps.expand(image, (left, top, right, bottom), fill=(0, 0, 0))


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Lambda(make_square),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
    transforms.ToTensor(),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Lambda(make_square),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


class LeafDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(record["label"])


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


def collect_records(dataset_path):
    records = []
    for class_name in CLASS_NAMES:
        folder = dataset_path / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "image_path": str(path.resolve()),
                    "class_name": class_name,
                    "label": CLASS_TO_INDEX[class_name],
                })
    return records


def split_records(records):
    rng = random.Random(RANDOM_SEED)
    train, val, test = [], [], []
    for class_name in CLASS_NAMES:
        items = [r.copy() for r in records if r["class_name"] == class_name]
        rng.shuffle(items)
        n = len(items)
        a = int(n * 0.75)
        b = a + int(n * 0.10)
        train.extend(items[:a])
        val.extend(items[a:b])
        test.extend(items[b:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def make_weighted_sampler(records):
    counts = [
        sum(int(record["label"]) == class_index for record in records)
        for class_index in range(len(CLASS_NAMES))
    ]
    weights = [1.0 / counts[int(record["label"])] for record in records]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float64),
        num_samples=len(records),
        replacement=True,
    )
    return sampler, counts


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / total if total else 0.0


def collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().tolist()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions)
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("course/artifacts/optimized"))
    args = parser.parse_args()

    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records = collect_records(args.dataset)
    train_records, val_records, test_records = split_records(records)
    sampler, counts = make_weighted_sampler(train_records)

    print("Training class counts:")
    for class_name, count in zip(CLASS_NAMES, counts):
        print(f"  {class_name:10s}: {count}")

    train_loader = DataLoader(LeafDataset(train_records, TRAIN_TRANSFORM), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(LeafDataset(val_records, EVAL_TRANSFORM), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(LeafDataset(test_records, EVAL_TRANSFORM), batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet9(number_of_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = -1.0
    best_epoch = 0
    best_model_path = args.output_dir / "optimized_model.pt"
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            samples += labels.size(0)

        train_loss = running_loss / samples
        train_accuracy = accuracy(model, train_loader, device)
        val_accuracy = accuracy(model, val_loader, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "best_val_accuracy_so_far": best_val_accuracy,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | val_acc={val_accuracy:.4f} | "
            f"best={best_val_accuracy:.4f}"
        )

    history_path = args.output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_accuracy = accuracy(model, test_loader, device)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    print("\nOptimized model results")
    print("=======================")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nSaved optimized model: {best_model_path.resolve()}")


if __name__ == "__main__":
    main()
