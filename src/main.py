import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import adpp as app_module


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
worker_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_error
    worker = asyncio.create_task(app_module.inference_worker())
    try:
        await asyncio.sleep(0.1)
        yield
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        except Exception as error:
            worker_error = str(error)


app = FastAPI(title="Plant Leaf Detection API", lifespan=lifespan)
app.include_router(app_module.router)


@app.get("/")
def home():
    return {"message": "Plant leaf API is running", "websocket": "/ws/glasses"}


@app.get("/health")
def health():
    return {"status": "ok", **app_module.status()}


@app.get("/ready")
def ready():
    if worker_error:
        return {"status": "error", "error": worker_error}
    return {"status": "ready" if app_module.service_ready else "loading"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000)
