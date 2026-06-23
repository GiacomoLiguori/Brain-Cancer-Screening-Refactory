import os
import json

from datetime import datetime
from pydantic import BaseModel
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, 
    HTTPException
)

DATA_PATH = "/app/data/history.json"

class DiagnosisLog(BaseModel):
    predicted_class : str
    confidence : float

@asynccontextmanager
async def lifespan(app : FastAPI):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok = True)

    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'w') as f:
            json.dump([], f)

    yield

app = FastAPI(title = "Classification Historic", description = "Microservizio per il salvataggio delle diagnosi", lifespan = lifespan)

@app.post("/logs")
async def save(log : DiagnosisLog):
    try:
        with open(DATA_PATH) as f:
            history = json.load(f)

            record = {
                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_class": log.predicted_class,
                "confidence": f"{log.confidence * 100:.2f}%"
            }

            history.append(record)
            with open(DATA_PATH, 'w') as f:
                json.dump(history, f, indent = 4)

            return {
                "status": "success",
                "message": "diagnose successfully saved"
            }
    
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Writing error: {str(e)}")
    
@app.get("/logs")
async def get():
    try:
        with open(DATA_PATH, 'r') as f:
            history = json.load(f)

        return history
    
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Reading error: {str(e)}") 
    