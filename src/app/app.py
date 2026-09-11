import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.config import API_KEY, MAX_IMAGE_BYTES, MODEL_VERSION, QUEUE_SIZE
from src.model_service.inference_service import InferenceService
from src.schemas import InferenceResponse

router = APIRouter()
logger = logging.getLogger("plant_api")
frame_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
clients = set()
metrics = {"received": 0, "processed": 0, "dropped": 0, "errors": 0}
service_ready = False

def status():
    return {"clients": len(clients), "queue_size": frame_queue.qsize(), **metrics}

async def send_error(websocket, message):
    await websocket.send_json({"type": "error", "message": message})

@router.websocket("/ws/glasses")
async def glasses_handler(websocket: WebSocket):
    if websocket.query_params.get("api_key") != API_KEY:
        await websocket.close(code=1008, reason="Invalid API key")
        return
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]
    clients.add(websocket)
    frame_id = 0
    logger.info("client_connected id=%s clients=%s", client_id, len(clients))
    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            if len(image_bytes) > MAX_IMAGE_BYTES:
                await send_error(websocket, f"Image is too large. Maximum is {MAX_IMAGE_BYTES} bytes.")
                continue
            frame_id += 1
            metrics["received"] += 1
            if frame_queue.full():
                _, _, _, old_client = frame_queue.get_nowait()
                frame_queue.task_done()
                metrics["dropped"] += 1
                logger.info("frame_dropped client=%s", old_client)
            await frame_queue.put((websocket, frame_id, image_bytes, client_id))
    except WebSocketDisconnect:
        logger.info("client_disconnected id=%s", client_id)
    finally:
        clients.discard(websocket)

async def inference_worker():
    global service_ready
    service = InferenceService()
    service_ready = True
    logger.info("models_loaded version=%s", MODEL_VERSION)
    while True:
        websocket, frame_id, image_bytes, client_id = await frame_queue.get()
        try:
            start = time.perf_counter()
            result = await asyncio.to_thread(service.process_image, image_bytes)
            result.update({"frame_id": frame_id, "model_version": MODEL_VERSION,
                           "processing_time_ms": round((time.perf_counter() - start) * 1000, 2)})
            response = InferenceResponse(**result)
            response_data = response.model_dump() if hasattr(response, "model_dump") else response.dict()
            await websocket.send_json(response_data)
            metrics["processed"] += 1
        except WebSocketDisconnect:
            logger.info("result_not_sent_client_disconnected id=%s", client_id)
        except Exception as error:
            metrics["errors"] += 1
            logger.exception("frame_failed id=%s error=%s", frame_id, error)
            try:
                await send_error(websocket, "Frame processing failed")
            except Exception:
                pass
        finally:
            frame_queue.task_done()
