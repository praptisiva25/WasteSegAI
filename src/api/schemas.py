from pydantic import BaseModel
from typing import List

class DetectedObject(BaseModel):
    class_id: int
    confidence: float
    bbox: List[float]  # [x_min, y_min, x_max, y_max]

class PredictionResponse(BaseModel):
    detections: List[DetectedObject]
