# VerdantSight

Real-time leaf health intelligence for connected glasses.

VerdantSight detects leaves in live images, classifies visible symptoms, and returns bounding boxes that a glasses client can draw immediately. It is designed as a continuous WebSocket service: the glasses keep sending frames, the server keeps returning detections, and old frames are dropped when inference cannot keep up.

This repository contains:

- a FastAPI WebSocket inference service
- a YOLO leaf detector
- a three-class ResNet9 leaf-symptom classifier
- local image/video test clients
- training and dataset-preparation code

The public product name is **VerdantSight**. Internal folders such as `backend_plant_disease` can remain as the engineering repository name.

## What the system does

VerdantSight does **not** diagnose a specific biological disease. It classifies visible leaf symptoms:

| Class | Meaning | Typical appearance |
| --- | --- | --- |
| `healthy` | No meaningful yellowing or dead tissue | Intact green leaf |
| `chlorosis` | Abnormal yellow or pale living tissue | Yellowing without clearly dead tissue |
| `necrosis` | Dead or collapsing tissue | Brown, black, dry, tan, or damaged tissue |

If a leaf shows both yellowing and dead tissue, the intended labeling rule is:

```text
necrosis present -> necrosis
yellowing without dead tissue -> chlorosis
no meaningful damage -> healthy
```

The current classifier was trained primarily on Pothos / *Epipremnum aureum* leaf crops. It is not a general plant-disease identifier for tomato, corn, apple, or mixed greenhouse crops.

## Architecture

```text
Glasses / test client
        |
        | JPEG frames over WebSocket
        v
FastAPI handler
        |
        | bounded queue (latest frames only)
        v
Inference worker
        |
        +--> YOLO leaf detector
        |
        +--> ResNet9 classifier for each cropped leaf
        |
        v
JSON detections
        |
        v
Client draws boxes locally
```

The service is split so the WebSocket layer stays responsive while model inference runs in a worker thread.

```text
src/
├── main.py                         FastAPI app, health/ready endpoints
├── config.py                       API key, limits, model version
├── schemas.py                      validated response format
├── app/
│   └── app.py                      WebSocket handler and inference queue
├── model_service/
│   └── inference_service.py        YOLO + ResNet9 pipeline
├── models/
│   ├── leaf_detector.pt            trained leaf detector
│   └── pothos_classifier.pt        trained 3-class classifier
└── testing/
    ├── test_image.py               direct model test without WebSocket
    └── websocket_client.py         live image/video client with overlay
```

## Pipeline

1. The glasses send a JPEG frame as raw bytes.
2. The server authenticates the connection with an API key.
3. Oversized frames are rejected.
4. The frame is placed in a queue of size 3.
5. If the queue is full, the oldest frame is dropped.
6. YOLO detects leaf bounding boxes.
7. Each crop is padded to a square, resized to `256×256`, and classified.
8. The server returns JSON containing boxes, labels, confidence, image size, model version, and processing time.
9. The client draws only high-confidence detections, currently `>= 90%` in the demo viewer.

The square-padding step exists because YOLO crops are usually rectangles. Directly resizing a tall or wide crop to `256×256` would stretch the leaf and distort color/shape cues that the classifier uses.

## API

### HTTP endpoints

These are for operators and monitoring, not for the glasses stream.

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Confirms the API is running |
| `GET /health` | Process status, connected clients, queue size, received/processed/dropped/error counts |
| `GET /ready` | `loading`, `ready`, or `error` depending on model load state |

Example:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/ready
```

### WebSocket endpoint

```text
ws://SERVER_IP:8000/ws/glasses?api_key=YOUR_KEY
```

The glasses send JPEG bytes. The server replies with JSON.

Successful response:

```json
{
  "frame_id": 12,
  "model_version": "leaf-pothos-v1",
  "image_width": 1920,
  "image_height": 1080,
  "processing_time_ms": 118.4,
  "detections": [
    {
      "box": [120, 80, 650, 500],
      "label": "chlorosis",
      "confidence": 0.91
    }
  ]
}
```

`box` is `[x1, y1, x2, y2]` in original-image pixels, with `(0, 0)` at the top-left.

Error response:

```json
{
  "type": "error",
  "message": "Image is too large. Maximum is 10485760 bytes."
}
```

The client must check for `"type": "error"` before drawing boxes.

The server does not send annotated images. Sending JSON is lighter for glasses and lets the client overlay boxes on the original camera frame.

## Quick start

From the project root, with the virtual environment activated:

```powershell
.\plant_project\Scripts\python.exe -m src.main
```

Wait until the models are loaded, then in another terminal:

```powershell
.\plant_project\Scripts\python.exe -m src.testing.websocket_client
```

Enter an image or video path. For video, press `ESC` to stop. A demonstration report is created only after all video frames complete.

Direct model test without WebSocket:

```powershell
.\plant_project\Scripts\python.exe -m src.testing.test_image
```

Default local WebSocket URL:

```text
ws://127.0.0.1:8000/ws/glasses?api_key=change-this-key
```

Change the key before any shared or production use:

```powershell
$env:PLANT_API_KEY = "your-long-secret-key"
```

## Configuration

Settings are read from environment variables in `src/config.py`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `PLANT_API_KEY` | `change-this-key` | WebSocket password |
| `MAX_IMAGE_BYTES` | `10485760` (10 MB) | Reject oversized frames |
| `QUEUE_SIZE` | `3` | Latest-frame buffer |
| `MIN_DETECTION_CONFIDENCE` | `0.25` | YOLO detection threshold |
| `MODEL_VERSION` | `leaf-pothos-v1` | Version string returned to clients |

The API key is not issued by a third-party provider. You create it. The glasses must send the same value in the URL.

The 10 MB limit protects GPU and RAM. A typical glasses JPEG is much smaller.

## Why the queue exists

The glasses can send frames faster than the models can process them. An unlimited queue would create growing delay.

The current design keeps only the newest frames:

```text
receive frame
    ↓
if queue is full, drop the oldest frame
    ↓
keep the latest 3 frames
    ↓
process the next available frame
```

This is live-video backpressure, not classic rate limiting. For one glasses demo it is the correct first protection. Rate limiting becomes more important if many devices or a public internet API are added.

## Models

| Model | Path | Role |
| --- | --- | --- |
| YOLO detector | `src/models/leaf_detector.pt` | Find leaves in a full plant image |
| ResNet9 classifier | `src/models/pothos_classifier.pt` | Classify each cropped leaf |

The detector is intended for whole-plant images with many leaves. The classifier is intended for cropped single-leaf images.

Training code lives under `Algorithm/`. The classifier classes are:

```text
healthy
chlorosis
necrosis
```

The original PlantVillage-style 38-class classifier is **not** used in this service. That model performed well on isolated lab leaves and failed on real YOLO crops because of domain shift.

## Dataset notes

The project evolved through several datasets:

1. PlantDoc disease names were reduced to healthy/unhealthy leaf detection labels.
2. Custom and recorded plant images were merged for detector training.
3. YOLO crops of Pothos leaves were labeled into `healthy`, `chlorosis`, and `necrosis`.

Important labeling rules:

- Keep original crop sizes. Resize during training, not while saving crops.
- Pad to square before resizing so leaf shape is not stretched.
- Keep mild color augmentation. Strong hue shifts can turn a healthy leaf yellow and poison the chlorosis class.
- Split by source photo/session, not randomly by crop, to avoid leakage.
- Jade Pothos is safer than Golden Pothos for chlorosis training because natural variegation can look like yellowing.

## Current production status

This is a working prototype with several production safeguards already added:

- API-key authentication
- maximum image size
- bounded latest-frame queue
- connection tracking
- structured JSON responses
- model versioning
- processing-time measurement
- `/health` and `/ready`
- logging
- error messages
- graceful worker shutdown

It is suitable for:

- one glasses device
- local or private-network demos
- internal testing

It is not yet a complete commercial deployment. Remaining industry work includes:

- TLS/`wss://` behind a reverse proxy
- Docker or Windows service restart
- stronger auth such as JWT for multiple customers
- automated tests for disconnects, invalid images, and queue overflow
- model artifact storage instead of committing large `.pt` files
- GPU/latency monitoring
- rate limiting if the API is public

## Disclaimer

VerdantSight reports visible leaf symptoms. It is not a substitute for a plant pathologist, agronomist, or laboratory test. Care suggestions shown in the demo report are general guidance, not confirmed treatment plans.

## License

Add your chosen license before publishing. If the repository includes third-party datasets or pretrained weights, include their original licenses and citation requirements.