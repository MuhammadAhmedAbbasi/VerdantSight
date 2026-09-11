"""Task 3: Data Labeling and Dataset Organization

Run:
    python course/03_data_labeling_and_dataset_organization.py --dataset path/to/dataset

Expected dataset structure:
    dataset/
        healthy/
        chlorosis/
        necrosis/

For image classification, the folder name is the class label.
This script assigns numeric labels and creates reproducible train/validation/test CSV files.
"""

from pathlib import Path
import argparse
import csv
import random


CLASS_NAMES = ["healthy", "chlorosis", "necrosis"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10


def collect_labeled_images(dataset_path: Path):
    records = []

    for class_name in CLASS_NAMES:
        class_folder = dataset_path / class_name
        if not class_folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_folder}")

        label = CLASS_TO_INDEX[class_name]
        image_paths = sorted(
            path for path in class_folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            records.append({
                "image_path": str(image_path.resolve()),
                "class_name": class_name,
                "label": label,
            })

    if not records:
        raise RuntimeError("No supported images were found in the dataset.")

    return records


def create_stratified_splits(records):
    """Split each class separately so every split contains all classes."""
    rng = random.Random(RANDOM_SEED)
    train_records, val_records, test_records = [], [], []

    for class_name in CLASS_NAMES:
        class_records = [r.copy() for r in records if r["class_name"] == class_name]
        rng.shuffle(class_records)

        n = len(class_records)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        train_records.extend(class_records[:train_end])
        val_records.extend(class_records[train_end:val_end])
        test_records.extend(class_records[val_end:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return train_records, val_records, test_records


def save_manifest(records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "class_name", "label"])
        writer.writeheader()
        writer.writerows(records)


def print_summary(name, records):
    print(f"\n{name}")
    print("-" * len(name))
    for class_name in CLASS_NAMES:
        count = sum(r["class_name"] == class_name for r in records)
        print(f"{class_name:10s}: {count}")
    print(f"Total      : {len(records)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Path to cleaned dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("course/artifacts/splits"),
        help="Folder where split CSV files will be saved",
    )
    args = parser.parse_args()

    records = collect_labeled_images(args.dataset)
    train_records, val_records, test_records = create_stratified_splits(records)

    save_manifest(train_records, args.output / "train.csv")
    save_manifest(val_records, args.output / "val.csv")
    save_manifest(test_records, args.output / "test.csv")

    print("Class label mapping")
    print("-------------------")
    for class_name, class_index in CLASS_TO_INDEX.items():
        print(f"{class_name:10s} -> {class_index}")

    print_summary("Training set", train_records)
    print_summary("Validation set", val_records)
    print_summary("Test set", test_records)

    print(f"\nSaved split files to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
