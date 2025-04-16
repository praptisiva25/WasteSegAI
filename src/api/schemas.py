from pydantic import BaseModel
from typing import List, Optional

class DetectedObject(BaseModel):
    class_id: int
    confidence: float
    bbox: List[float]

class PredictionResponse(BaseModel):
    detections: List[DetectedObject]
    image_path: Optional[str] = None
