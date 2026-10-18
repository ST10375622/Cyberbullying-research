"""REST API: the deployable cyberbullying detection framework (SQ5).

Run:   uvicorn app:app --app-dir deployment --port 8000
Test:  curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
            -d "{\"texts\": [\"you are so dumb\", \"great game last night\"]}"

Moderation flow:  post -> pre-processing -> model -> probability ->
                  >= auto_hide threshold          : hidden automatically
                  >= flag_for_review threshold    : queued for a human moderator
                  otherwise                       : allowed
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel

from predictor import CyberbullyingDetector

detector = CyberbullyingDetector(os.environ.get("CB_DEPLOY_CHOICE"),
                                 os.environ.get("CB_DEPLOY_MODEL", "calibrated"))
app = FastAPI(title="Cyberbullying Detection API",
              description="RPDA8412 research prototype - not for production moderation without review.")


class Posts(BaseModel):
    texts: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": detector.model_name, "family": detector.family,
            "trained_on": detector.train_dataset, "thresholds": detector.thresholds}


@app.post("/predict")
def predict(posts: Posts):
    return {"model": detector.model_name, "results": detector.predict(posts.texts)}
