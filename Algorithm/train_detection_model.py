"""Train YOLO26m on the prepared custom plant-leaf dataset."""

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Models" / "yolo26n.pt"
DATASET_YAML = ROOT / "Datasets" / "Custom_plant_leaf_detection_dataset_one_class" / "data.yaml"
RUNS_DIR = ROOT / "Algorithm" / "runs" / "detect" / "health_leaf"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}")

    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATASET_YAML),
        task="detect",
        epochs=100,
        imgsz= 640,
        batch=0.9,
        device=0,
        workers=4,
        patience=20,
        project=str(RUNS_DIR),
        name="office_plant_nano",
        exist_ok=True,
        pretrained=True,
        plots=True,
        seed=42,
    )
    print(f"Training complete. Results: {RUNS_DIR / 'custom_plant'}")
    print(f"Best model: {RUNS_DIR / 'custom_plant' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
