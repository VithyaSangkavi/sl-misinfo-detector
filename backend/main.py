from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from inference import predict_headlines

app = FastAPI(
    title="Sri Lanka Misinformation Detection API",
    version="1.0.0"
)

# 🔹 CORS setup – allow your React Vite dev server to call this API
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # add more if needed, e.g. your deployed frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # or ["*"] during local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    headlines: List[str]


class PredictResponseItem(BaseModel):
    headline: str
    label_id: int
    label_name: str
    probabilities: dict


@app.get("/")
def root():
    return {"message": "Misinformation Detection API running"}


@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    results = predict_headlines(req.headlines)
    return results
