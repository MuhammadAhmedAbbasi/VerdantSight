"""Detect leaves in Plant images/videos and save each leaf as a new image."""

from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Algorithm" / "runs" / "detect" / "health_leaf" / "office_plant_one" / "weights" / "best.pt"
INPUT_FOLDER = ROOT / "Plant"
OUTPUT_FOLDER = ROOT / "Datasets" / "custom_classification_dataset"

CONFIDENCE = 0.25
VIDEO_SECONDS_BETWEEN_FRAMES = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def save_leaf_crops(model, image, file_prefix):
    result = model.predict(image, conf=CONFIDENCE, verbose=False)[0]
    saved = 0

    for leaf_number, box in enumerate(result.boxes.xyxy.int().tolist(), start=1):
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        output_path = OUTPUT_FOLDER / f"{file_prefix}_leaf_{leaf_number:03d}.jpg"
        cv2.imwrite(str(output_path), crop)
        saved += 1

    return saved


def process_image(model, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path.name}")
        return 0

    saved = save_leaf_crops(model, image, image_path.stem)
    print(f"Image {image_path.name}: saved {saved} leaves")
    return saved


def process_video(model, video_path):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print(f"Could not read video: {video_path.name}")
        return 0

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, round(fps * VIDEO_SECONDS_BETWEEN_FRAMES))
    frame_number = 0
    total_saved = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        if frame_number % frame_step == 0:
            prefix = f"{video_path.stem}_frame_{frame_number:06d}"
            total_saved += save_leaf_crops(model, frame, prefix)

        frame_number += 1

    video.release()
    print(f"Video {video_path.name}: saved {total_saved} leaves")
    return total_saved


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not INPUT_FOLDER.is_dir():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    total_saved = 0

    for file_path in sorted(INPUT_FOLDER.iterdir()):
        extension = file_path.suffix.lower()

        if extension in IMAGE_EXTENSIONS:
            total_saved += process_image(model, file_path)
        elif extension == ".mp4":
            total_saved += process_video(model, file_path)

    print(f"\nFinished. Total leaf crops saved: {total_saved}")
    print(f"Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
