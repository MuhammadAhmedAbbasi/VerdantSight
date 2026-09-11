"""Task 3: Data Labeling and Dataset Organization.

Usage:
    python course/03_data_labeling_and_dataset_organization.py --dataset path/to/dataset
"""

from pathlib import Path
import argparse

from common import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    collect_labeled_images,
    create_stratified_splits,
    save_manifest,
)


def print_summary(name, records):
    print(f"\n{name}")
    print("-" * len(name))
    for class_name in CLASS_NAMES:
        count = sum(r["class_name"] == class_name for r in records)
        print(f"{class_name:10s}: {count}")
    print(f"Total      : {len(records)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Path to cleaned dataset folder")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("course/artifacts/splits"),
        help="Folder for train/validation/test CSV manifests",
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

    print(f"\nSaved split manifests to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
