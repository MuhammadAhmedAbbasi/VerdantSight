from pydantic import BaseModel
from typing import List


class Detection(BaseModel):
    box: List[int]
    label: str
    confidence: float


class InferenceResponse(BaseModel):
    frame_id: int
    model_version: str
    image_width: int
    image_height: int
    processing_time_ms: float
    detections: List[Detection]
