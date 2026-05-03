"""
Section 2: FastAPI Backend – Success Prediction API
Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ── Load saved artefacts ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

try:
    with open(BASE_DIR / "xgb_model.pkl", "rb") as f:
        MODEL = pickle.load(f)

    with open(BASE_DIR / "preprocessor.pkl", "rb") as f:
        PREPROCESSOR = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(
        "Model artefacts not found. Run ml_notebook.py first to generate "
        "'xgb_model.pkl' and 'preprocessor.pkl'."
    ) from e

CLASS_NAMES   = ["Low", "Moderate", "Viral"]
NUMERIC_COLS  = [
    "reel_length_sec", "posting_time", "hook_strength_score",
    "caption_length",  "hashtags_count",
]

# ── Pydantic schemas ────────────────────────────────────────────────────────────
class ReelFeatures(BaseModel):
    reel_length_sec    : int   = Field(..., ge=3,   le=600,  example=30,
                                       description="Duration of the reel in seconds (3–600).")
    posting_time       : int   = Field(..., ge=0,   le=23,   example=18,
                                       description="Hour of day the reel was posted (0–23).")
    hook_strength_score: float = Field(..., ge=0.0, le=1.0,  example=0.85,
                                       description="AI-rated hook strength (0.00–1.00).")
    caption_length     : int   = Field(..., ge=0,   le=2200, example=120,
                                       description="Number of characters in the caption.")
    hashtags_count     : int   = Field(..., ge=0,   le=30,   example=5,
                                       description="Number of hashtags used.")
    trending_audio     : bool  = Field(...,                   example=True,
                                       description="Whether a trending audio track was used.")


class PredictionResponse(BaseModel):
    virality_score  : int
    predicted_class : str
    probabilities   : Dict[str, float]
    feedback        : str


# ── App & CORS ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title       ="Reel Success Predictor",
    description ="Predicts Low / Moderate / Viral reach class from pre-publish metadata.",
    version     ="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     =["*"],   # tighten to your frontend origin in production
    allow_credentials =True,
    allow_methods     =["*"],
    allow_headers     =["*"],
)


# ── Helper ──────────────────────────────────────────────────────────────────────
def _compute_virality_score(probs: np.ndarray) -> int:
    """
    Score = floor(P(Low)×10 + P(Moderate)×50 + P(Viral)×95)
    Range: [10, 95]
    """
    p_low, p_mod, p_viral = probs[0], probs[1], probs[2]
    raw = (p_low * 10) + (p_mod * 50) + (p_viral * 95)
    return math.floor(raw)


def _generate_feedback(features: ReelFeatures, predicted_class: str,
                        score: int, probs: np.ndarray) -> str:
    """Return a brief, data-driven feedback string."""
    tips: list[str] = []

    if features.hook_strength_score < 0.5:
        tips.append("boost your opening hook (score <0.50 detected)")
    if features.hashtags_count < 3:
        tips.append("add more targeted hashtags (3–10 is optimal)")
    if features.hashtags_count > 15:
        tips.append("trim hashtags — fewer, niche tags outperform spam")
    if features.trending_audio is False:
        tips.append("pair with a trending audio track")
    if features.posting_time not in range(17, 22):
        tips.append("try posting between 17:00–21:00 for peak engagement")
    if features.reel_length_sec > 60:
        tips.append("shorter reels (<30 s) typically achieve higher completion rates")

    if not tips:
        tips.append("your pre-publish signals are well-optimised — focus on thumbnail quality")

    prefix_map = {
        "Low":      f"Score {score}/100 – Low reach predicted. To improve:",
        "Moderate": f"Score {score}/100 – Moderate reach predicted. Fine-tune by:",
        "Viral":    f"Score {score}/100 – Viral potential detected! To maximise reach:",
    }
    return f"{prefix_map[predicted_class]} {tips[0]}."


# ── Routes ──────────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
async def root():
    return {"status": "ok", "message": "Reel Success Predictor API is running."}


@app.post("/predict", response_model=PredictionResponse, summary="Predict reel success")
async def predict(features: ReelFeatures):
    """
    Accepts pre-publish reel metadata and returns a virality score,
    predicted class, class probabilities, and actionable feedback.
    """
    try:
        import pandas as pd

        # Build input frame matching training layout
        numeric_input = pd.DataFrame([{
            "reel_length_sec"    : features.reel_length_sec,
            "posting_time"       : features.posting_time,
            "hook_strength_score": features.hook_strength_score,
            "caption_length"     : features.caption_length,
            "hashtags_count"     : features.hashtags_count,
        }])

        scaled_numeric = PREPROCESSOR.transform(numeric_input[NUMERIC_COLS])
        audio_flag     = np.array([[int(features.trending_audio)]])
        X_input        = np.hstack([scaled_numeric, audio_flag])

        # Predict
        probs          = MODEL.predict_proba(X_input)[0]          # shape (3,)
        pred_class_idx = int(np.argmax(probs))
        predicted_class= CLASS_NAMES[pred_class_idx]
        virality_score = _compute_virality_score(probs)

        prob_dict = {
            "Low"     : round(float(probs[0]), 4),
            "Moderate": round(float(probs[1]), 4),
            "Viral"   : round(float(probs[2]), 4),
        }

        feedback = _generate_feedback(features, predicted_class, virality_score, probs)

        return PredictionResponse(
            virality_score  =virality_score,
            predicted_class =predicted_class,
            probabilities   =prob_dict,
            feedback        =feedback,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
