"""Task 6: Model Optimization and Generalization.

This task improves the Task 5 baseline using the stronger training choices from
the VerdantSight project:
- image augmentation;
- class-balanced sampling;
- Adam with learning rate 0.001;
- 20 training epochs;
- best-validation checkpoint selection.

Usage:
    python course/06_model_optimization_and_generalization.py --dataset path/to/dataset
"""

from pathlib import Path
import argparse
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

from common import (
    CLASS_NAMES,
    LeafDataset,
    ResNet9,
    accuracy_from_loader,
    basic_transform,
    ensure_manifests,
    optimized_train_transform,
    read_manifest,
    set_seed,
)


OPTIMIZED_EPOCHS = 20
OPTIMIZED_LEARNING_RATE = 0.001
BATCH_SIZE = 32


def make_weighted_sampler(records):
    class_counts = [
        sum(int(record["label"]) == class_index for record in records)
        for class_index in range(len(CLASS_NAMES))
    ]

    sample_weights = [
        1.0 / class_counts[int(record["label"])]
        for record in records
    ]

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(records),
        replacement=True,
    ), class_counts


def collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    model.eval()

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            predictions = model(images).argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("course/artifacts/splits"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("course/artifacts/optimized"),
    )
    args = parser.parse_args()

    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_csv, val_csv, test_csv = ensure_manifests(args.dataset, args.split_dir)
    train_records = read_manifest(train_csv)
    val_records = read_manifest(val_csv)
    test_records = read_manifest(test_csv)

    sampler, class_counts = make_weighted_sampler(train_records)
    print("Training class counts:")
    for class_name, count in zip(CLASS_NAMES, class_counts):
        print(f"  {class_name:10s}: {count}")

    train_loader = DataLoader(
        LeafDataset(train_records, optimized_train_transform()),
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )
    val_loader = DataLoader(
        LeafDataset(val_records, basic_transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        LeafDataset(test_records, basic_transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ResNet9(number_of_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=OPTIMIZED_LEARNING_RATE)

    best_val_accuracy = -1.0
    best_epoch = 0
    best_model_path = args.output_dir / "optimized_model.pt"
    history = []

    for epoch in range(1, OPTIMIZED_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        samples_seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            samples_seen += batch_size

        epoch_loss = running_loss / samples_seen
        train_accuracy = accuracy_from_loader(model, train_loader, device)
        val_accuracy = accuracy_from_loader(model, val_loader, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "best_val_accuracy_so_far": best_val_accuracy,
        })

        print(
            f"Epoch {epoch:02d}/{OPTIMIZED_EPOCHS} | "
            f"loss={epoch_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"best={best_val_accuracy:.4f}"
        )

    history_path = args.output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Final evaluation uses the best validation checkpoint, not the last epoch.
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_accuracy = accuracy_from_loader(model, test_loader, device)
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
    print(f"Saved history:         {history_path.resolve()}")


if __name__ == "__main__":
    main()
