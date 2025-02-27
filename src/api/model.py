from ultralytics import YOLO
from api.schemas import PredictionResponse, DetectedObject
from pathlib import Path

# Load the trained YOLOv8 model
MODEL_PATH = "C:/Users/prapt/OneDrive/Desktop/Project/WasteSegAI/runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

def predict_image(image_path: Path) -> PredictionResponse:
    results = model(image_path)  # Run inference
    detections = []
    
    for r in results:
        for box in r.boxes:
            detections.append(DetectedObject(
                class_id=int(box.cls.item()),
                confidence=float(box.conf.item()),
                bbox=box.xyxy.tolist()[0]  # Convert tensor to list
            ))

    return PredictionResponse(detections=detections)
