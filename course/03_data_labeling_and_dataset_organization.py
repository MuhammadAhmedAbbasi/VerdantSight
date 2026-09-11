"""Task 3: Data Labeling and Dataset Organization

Change DATASET_PATH below to the folder that contains:

healthy/
chlorosis/
necrosis/

Running this file creates train.csv, val.csv, and test.csv.
"""

from pathlib import Path
import csv
import random


# -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------
DATASET_PATH = Path("dataset")
OUTPUT_PATH = Path("course/artifacts/splits")

CLASS_NAMES = ["healthy", "chlorosis", "necrosis"]
CLASS_TO_INDEX = {
    "healthy": 0,
    "chlorosis": 1,
    "necrosis": 2,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# 2. Read images and assign labels from folder names
# -----------------------------------------------------------------------------
all_data = []

for class_name in CLASS_NAMES:
    class_folder = DATASET_PATH / class_name
    class_label = CLASS_TO_INDEX[class_name]

    for image_path in class_folder.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            all_data.append({
                "image_path": str(image_path.resolve()),
                "class_name": class_name,
                "label": class_label,
            })


# -----------------------------------------------------------------------------
# 3. Create train, validation, and test splits
#    75% training, 10% validation, 15% testing
# -----------------------------------------------------------------------------
random.seed(RANDOM_SEED)

train_data = []
val_data = []
test_data = []

for class_name in CLASS_NAMES:
    class_data = [item for item in all_data if item["class_name"] == class_name]
    random.shuffle(class_data)

    total = len(class_data)
    train_end = int(total * 0.75)
    val_end = train_end + int(total * 0.10)

    train_data.extend(class_data[:train_end])
    val_data.extend(class_data[train_end:val_end])
    test_data.extend(class_data[val_end:])


# -----------------------------------------------------------------------------
# 4. Save the splits as CSV files
# -----------------------------------------------------------------------------
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

for file_name, data in [
    ("train.csv", train_data),
    ("val.csv", val_data),
    ("test.csv", test_data),
]:
    with open(OUTPUT_PATH / file_name, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "class_name", "label"])
        writer.writeheader()
        writer.writerows(data)


# -----------------------------------------------------------------------------
# 5. Show the result
# -----------------------------------------------------------------------------
print("Class mapping:")
for class_name, label in CLASS_TO_INDEX.items():
    print(class_name, "->", label)

print("\nDataset split:")
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Testing images:", len(test_data))

for class_name in CLASS_NAMES:
    train_count = sum(item["class_name"] == class_name for item in train_data)
    val_count = sum(item["class_name"] == class_name for item in val_data)
    test_count = sum(item["class_name"] == class_name for item in test_data)

    print(
        f"{class_name}: "
        f"train={train_count}, val={val_count}, test={test_count}"
    )

print("\nSaved split files in:", OUTPUT_PATH)
