from fastapi import FastAPI, File, UploadFile
import shutil
import uuid
from pathlib import Path
from api.model import predict_image
from api.schemas import PredictionResponse

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def home():
    return {"message": "WasteSegAI FastAPI is running!"}

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    results = predict_image(file_path)
    return results
