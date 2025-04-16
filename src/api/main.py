from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uuid
from pathlib import Path
import shutil

from api.model import predict_image
from api.schemas import PredictionResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/runs", StaticFiles(directory="runs"), name="runs")

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

    return predict_image(file_path)
