"""Test the healthy/unhealthy leaf detector on an image or video.

Examples::

    python Algorithm/test_trained_yolo.py --source path/to/photo.jpg
    python Algorithm/test_trained_yolo.py --source path/to/video.mp4 --no-display
"""

import argparse

from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Algorithm" / "runs" / "detect" / "health_leaf" / "office_plant_one" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "Algorithm" / "predictions" / "health_leaf"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_DISPLAY_WIDTH = 1000
MAX_DISPLAY_HEIGHT = 800


def fit_to_window(frame):
    """Resize without stretching, keeping the complete image visible."""
    height, width = frame.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def show_image(model: YOLO, path: Path, confidence: float, output_dir: Path, display: bool) -> None:
    result = model.predict(str(path), conf=confidence, verbose=False)[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_predicted{path.suffix}"
    cv2.imwrite(str(output_path), result.plot())
    frame = fit_to_window(result.plot())
    counts = {name: 0 for name in model.names.values()}
    for class_id in result.boxes.cls.int().tolist():
        counts[model.names[class_id]] += 1
    print(f"Detected leaves: {len(result.boxes)} ({counts})")
    print(f"Saved annotated image: {output_path}")
    if display:
        cv2.namedWindow("Leaf Detection - press any key to close", cv2.WINDOW_NORMAL)
        cv2.imshow("Leaf Detection - press any key to close", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_video(model: YOLO, path: Path, confidence: float, output_dir: Path, display: bool) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_predicted.mp4"
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    counts = {name: 0 for name in model.names.values()}
    frames = 0
    if display:
        cv2.namedWindow("Leaf Detection - press q to quit", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, conf=confidence, verbose=False)[0]
        annotated = result.plot()
        writer.write(annotated)
        frames += 1
        for class_id in result.boxes.cls.int().tolist():
            counts[model.names[class_id]] += 1
        if display:
            cv2.imshow("Leaf Detection - press q to quit", fit_to_window(annotated))
        if display and cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()
    print(f"Processed frames: {frames}; detections by frame: {counts}")
    print(f"Saved annotated video: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Image or video to test; prompts when omitted")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence (default: 0.25)")
    parser.add_argument("--no-display", action="store_true", help="Save output without opening a window")
    args = parser.parse_args()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    entered = str(args.source) if args.source else input("Enter the path to an image or video: ").strip().strip('"')
    media_path = Path(entered).expanduser().resolve()
    if not media_path.is_file():
        raise FileNotFoundError(f"File not found: {media_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    model = YOLO(str(model_path))
    if media_path.suffix.lower() in IMAGE_EXTENSIONS:
        show_image(model, media_path, args.conf, args.output_dir.resolve(), not args.no_display)
    else:
        show_video(model, media_path, args.conf, args.output_dir.resolve(), not args.no_display)


if __name__ == "__main__":
    main()
