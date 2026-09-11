"""Shared utilities for the VerdantSight course tasks.

The course focuses on the three-class leaf-health classifier:
healthy, chlorosis, and necrosis.
"""

from pathlib import Path
import csv
import random

import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms


CLASS_NAMES = ["healthy", "chlorosis", "necrosis"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 256
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    size = max(width, height)
    left = (size - width) // 2
    top = (size - height) // 2
    right = size - width - left
    bottom = size - height - top
    return ImageOps.expand(image, (left, top, right, bottom), fill=(0, 0, 0))


def basic_transform():
    """Baseline/evaluation preprocessing: no augmentation."""
    return transforms.Compose([
        transforms.Lambda(make_square),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def optimized_train_transform():
    """Training-time augmentation used by the optimized model."""
    return transforms.Compose([
        transforms.Lambda(make_square),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
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
        label = int(record["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def block(input_channels, output_channels, pool=False):
    layers = [
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """ResNet9 architecture used by the VerdantSight classifier."""

    def __init__(self, number_of_classes=len(CLASS_NAMES)):
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


def read_manifest(csv_path: Path):
    records = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["label"] = int(row["label"])
            records.append(row)
    return records


def save_manifest(records, csv_path: Path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "class_name", "label"],
        )
        writer.writeheader()
        writer.writerows(records)


def collect_labeled_images(dataset_path: Path):
    dataset_path = Path(dataset_path)
    records = []

    for class_name in CLASS_NAMES:
        class_dir = dataset_path / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class folder '{class_dir}'. "
                f"Required class folders: {CLASS_NAMES}"
            )

        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            records.append({
                "image_path": str(image_path.resolve()),
                "class_name": class_name,
                "label": CLASS_TO_INDEX[class_name],
            })

    if not records:
        raise RuntimeError(f"No supported images found under {dataset_path}")

    return records


def create_stratified_splits(records, train_ratio=0.75, val_ratio=0.10, seed=RANDOM_SEED):
    rng = random.Random(seed)
    train_records, val_records, test_records = [], [], []

    for class_name in CLASS_NAMES:
        class_records = [r.copy() for r in records if r["class_name"] == class_name]
        rng.shuffle(class_records)

        n = len(class_records)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_records.extend(class_records[:train_end])
        val_records.extend(class_records[train_end:val_end])
        test_records.extend(class_records[val_end:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return train_records, val_records, test_records


def ensure_manifests(dataset_path: Path, split_dir: Path):
    split_dir = Path(split_dir)
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        return train_csv, val_csv, test_csv

    records = collect_labeled_images(dataset_path)
    train_records, val_records, test_records = create_stratified_splits(records)
    save_manifest(train_records, train_csv)
    save_manifest(val_records, val_csv)
    save_manifest(test_records, test_csv)
    return train_csv, val_csv, test_csv


def accuracy_from_loader(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    return correct / total if total else 0.0
