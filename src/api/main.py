from fastapi import FastAPI, File, UploadFile
import shutil
import uuid
from pathlib import Path
from api.model import predict_image
from api.schemas import PredictionResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Class ID to Class Name Mapping
CLASS_NAMES = {
    0: "Apple-Bio",
    1: "Banana-Bio",
    2: "CardBoard-Bio",
    3: "Date-Bio",
    4: "Eraser-NonBio",
    5: "Leaf-Bio",
    6: "Pen-NonBio",
    7: "Pencil-Bio",
    8: "Plastic-NonBio",
    9: "Spectacles-NonBio",
    10: "Steel-NonBio",
    11: "Straw-NonBio",
    12: "Tissue Paper-Bio",
    13: "Wooden Spoon-Bio"
}

@app.get("/")
def home():
    return {"message": "WasteSegAI FastAPI is running!"}

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    results = predict_image(file_path)

    # Ensure results is a dictionary
    results_dict = results.dict() if hasattr(results, "dict") else results

    # Convert class_id to class_name
    for detection in results_dict["detections"]:
        detection["class_name"] = CLASS_NAMES.get(detection["class_id"], "Unknown")

    return results_dict
