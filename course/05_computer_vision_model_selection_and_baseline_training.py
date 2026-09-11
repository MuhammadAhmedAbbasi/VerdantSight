"""Task 5: Computer Vision Model Selection and Baseline Training.

The baseline deliberately uses a limited training setup so Task 6 can demonstrate
how better training choices improve generalization.

Baseline limitations:
- no data augmentation;
- no class-balanced sampler;
- only 5 epochs;
- deliberately aggressive Adam learning rate (0.01);
- saves the final epoch rather than selecting the best validation checkpoint.

Usage:
    python course/05_computer_vision_model_selection_and_baseline_training.py --dataset path/to/dataset
"""

from pathlib import Path
import argparse
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from common import (
    CLASS_NAMES,
    LeafDataset,
    ResNet9,
    accuracy_from_loader,
    basic_transform,
    ensure_manifests,
    read_manifest,
    set_seed,
)


BASELINE_EPOCHS = 5
BASELINE_LEARNING_RATE = 0.01
BATCH_SIZE = 32


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
        default=Path("course/artifacts/baseline"),
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

    # Baseline preprocessing only: square -> resize -> tensor.
    transform = basic_transform()

    train_loader = DataLoader(
        LeafDataset(train_records, transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        LeafDataset(val_records, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        LeafDataset(test_records, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ResNet9 is selected as the baseline architecture for this project.
    model = ResNet9(number_of_classes=len(CLASS_NAMES)).to(device)

    # Intentionally suboptimal learning rate for the baseline experiment.
    optimizer = torch.optim.Adam(model.parameters(), lr=BASELINE_LEARNING_RATE)

    history = []

    for epoch in range(1, BASELINE_EPOCHS + 1):
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

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch:02d}/{BASELINE_EPOCHS} | "
            f"loss={epoch_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_acc={val_accuracy:.4f}"
        )

    # Baseline intentionally saves the last epoch rather than the best epoch.
    model_path = args.output_dir / "baseline_model.pt"
    torch.save(model.state_dict(), model_path)

    history_path = args.output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    test_accuracy = accuracy_from_loader(model, test_loader, device)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    print("\nBaseline test results")
    print("=====================")
    print(f"Test accuracy: {test_accuracy:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f"\nSaved baseline model: {model_path.resolve()}")
    print(f"Saved history:        {history_path.resolve()}")


if __name__ == "__main__":
    main()
