import io
import torch

from PIL import Image
from contextlib import asynccontextmanager

from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor
)

from fastapi import (
    FastAPI, 
    UploadFile, 
    File
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "./resnet"
LABELS = ['Meningioma', 'Glioma', 'Pituitary']

model = None
image_processor = None

@asynccontextmanager
async def lifespan(app : FastAPI):
    global model, image_processor

    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

    model.eval()
    yield

    image_processor = None
    model = None

app = FastAPI(title = "Brain Cancer Classification", description = "Microservizio per l'inferenza della classe tumorale", lifespan = lifespan)

@app.post("/predict")
async def predict(file : UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    inputs = image_processor(image, return_tensors = 'pt')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    results = torch.nn.functional.softmax(logits, dim = -1).squeeze().tolist()

    predictions = {label: results[i] for i, label in enumerate(LABELS)}

    top_class = max(predictions, key = predictions.get)

    return {
        "predicted_class": top_class,
        "confidence": predictions[top_class],
        "all_predictions": predictions,
        "status": "success"
    }