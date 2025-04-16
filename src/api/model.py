from ultralytics import YOLO
from api.schemas import PredictionResponse, DetectedObject
from pathlib import Path
import cv2

model = YOLO("C:/Users/prapt/OneDrive/Desktop/Project/WasteSegAI/runs/detect/train/weights/best.pt")  

def predict_image(image_path: Path) -> PredictionResponse:
    results = model(image_path)
    detections = []
    output_path = Path("runs/detect/web_output") / image_path.name

    for r in results:
        for box in r.boxes:
            detections.append(DetectedObject(
                class_id=int(box.cls.item()),
                confidence=float(box.conf.item()),
                bbox=box.xyxy.tolist()[0]
            ))

        
        annotated = r.plot()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)

    return PredictionResponse(
        detections=detections,
        image_path=f"/runs/detect/web_output/{image_path.name}"  
    )
