import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import websockets
from src.config import API_KEY


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
MIN_CLASSIFICATION_CONFIDENCE = 0.90
COLORS = {
    "healthy": (0, 200, 0),
    "chlorosis": (0, 255, 255),
    "necrosis": (0, 0, 255),
}
REPORT_FOLDER = Path(__file__).resolve().parent / "results"
CARE_ADVICE = {
    "healthy": "No treatment needed. Continue normal watering, light, and routine observation.",
    "chlorosis": "Check overwatering, drainage, light, roots, and nutrients. Correct the cause gradually; do not add fertilizer until the cause is known.",
    "necrosis": "Remove badly dead tissue with clean tools. Check watering, harsh sunlight, cold damage, pests, and root health. Isolate the plant if damage is spreading.",
}


def collect_samples(frame, result, samples):
    for detection in result.get("detections", []):
        confidence = detection["confidence"]
        if confidence < MIN_CLASSIFICATION_CONFIDENCE:
            continue
        label = detection["label"]
        x1, y1, x2, y2 = detection["box"]
        crop = frame[max(0, y1):min(frame.shape[0], y2),
                     max(0, x1):min(frame.shape[1], x2)]
        if crop.size and (label not in samples or confidence > samples[label][0]):
            samples[label] = (confidence, crop.copy())


def draw_result(frame, result):
    for detection in result.get("detections", []):
        x1, y1, x2, y2 = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]
        if confidence < MIN_CLASSIFICATION_CONFIDENCE:
            continue
        color = COLORS.get(label, (255, 255, 255))
        text = f"{label} {confidence * 100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_y = max(y1, text_size[1] + 12)
        cv2.rectangle(frame, (x1, text_y - text_size[1] - 12),
                      (x1 + text_size[0] + 12, text_y), color, -1)
        cv2.putText(frame, text, (x1 + 6, text_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


async def send_one_frame(websocket, frame):
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Could not encode video frame")

    await websocket.send(encoded.tobytes())
    result = json.loads(await websocket.recv())
    return draw_result(frame.copy(), result), result


def write_wrapped(image, text, x, y, max_width):
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        if width > max_width and line:
            cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (30, 30, 30), 1, cv2.LINE_AA)
            y += 25
            line = word
        else:
            line = candidate
    if line:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (30, 30, 30), 1, cv2.LINE_AA)


def show_report(samples, source_name):
    labels = [name for name in ("healthy", "chlorosis", "necrosis") if name in samples]
    if not labels:
        print("No detections reached 90% confidence, so no report was created.")
        return

    report = np.full((600, 1250, 3), 245, dtype=np.uint8)
    cv2.putText(report, "Demonstration Summary", (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(report, "Detected symptom types (confidence >= 90%)", (35, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 1, cv2.LINE_AA)

    card_width = 390
    for index, label in enumerate(labels):
        confidence, crop = samples[label]
        x = 25 + index * 410
        y = 125
        sample = np.full((240, 340, 3), 230, dtype=np.uint8)
        crop_h, crop_w = crop.shape[:2]
        scale = min(340 / crop_w, 240 / crop_h)
        resized = cv2.resize(crop, (max(1, int(crop_w * scale)),
                                    max(1, int(crop_h * scale))))
        sample_y = (240 - resized.shape[0]) // 2
        sample_x = (340 - resized.shape[1]) // 2
        sample[sample_y:sample_y + resized.shape[0],
               sample_x:sample_x + resized.shape[1]] = resized
        report[y:y + 240, x:x + 340] = sample
        color = COLORS[label]
        cv2.rectangle(report, (x, y), (x + 340, y + 240), color, 5)
        cv2.putText(report, f"{label.upper()} - sample {confidence * 100:.1f}%",
                    (x, y + 285), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2, cv2.LINE_AA)
        write_wrapped(report, CARE_ADVICE[label], x, y + 330, card_width - 25)

    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_FOLDER / f"{source_name}_report.jpg"
    cv2.imwrite(str(report_path), report)
    cv2.imshow("Completed demonstration report - press any key to close", report)
    cv2.waitKey(0)
    print(f"Report saved: {report_path}")


async def run_image(websocket, path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Could not open image: {path}")

    output, result = await send_one_frame(websocket, frame)
    samples = {}
    collect_samples(frame, result, samples)
    cv2.imshow("Glasses live result - press any key to close", output)
    cv2.waitKey(0)
    show_report(samples, path.stem)


async def run_video(websocket, path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    samples = {}
    completed = True
    while True:
        success, frame = capture.read()
        if not success:
            break

        # Send one frame and wait for its result before sending the next.
        output, result = await send_one_frame(websocket, frame)
        collect_samples(frame, result, samples)
        cv2.imshow("Glasses live result - press ESC to stop", output)
        if cv2.waitKey(1) & 0xFF == 27:
            completed = False
            break

    capture.release()
    if completed:
        show_report(samples, path.stem)
    else:
        print("Video stopped early. Report is shown only after all frames complete.")


async def main():
    path = Path(input("Enter image or video path: ").strip().strip('"')).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    url = f"ws://127.0.0.1:8000/ws/glasses?api_key={API_KEY}"
    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as websocket:
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            await run_video(websocket, path)
        else:
            await run_image(websocket, path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
