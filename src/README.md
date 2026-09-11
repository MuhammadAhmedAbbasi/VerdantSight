# Plant Leaf API

Run the local server:

```powershell
.\plant_project\Scripts\python.exe -m src.main
```

Local WebSocket address:

```text
ws://127.0.0.1:8000/ws/glasses?api_key=change-this-key
```

For deployment, set these environment variables before starting the service:

```text
PLANT_API_KEY=a-long-secret-value
MAX_IMAGE_BYTES=10485760
QUEUE_SIZE=3
MIN_DETECTION_CONFIDENCE=0.25
MODEL_VERSION=leaf-pothos-v1
```

Useful endpoints:

```text
GET /health   - process and queue status
GET /ready    - models loaded and ready for inference
```

Put the service behind an HTTPS reverse proxy in production. The proxy should provide TLS so glasses use `wss://`, not `ws://`.
