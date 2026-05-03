from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import yt_dlp
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from yt_dlp.utils import DownloadError, ExtractorError


APP_TITLE = "ReelIQ API"
BUNDLE_PATH = Path("reeliq_model_bundle.pkl")
MODEL_PATH = Path("xgb_model.pkl")
PREPROCESSOR_PATH = Path("preprocessor.pkl")

FEATURES = [
    "reel_length_sec",
    "posting_time",
    "hook_strength_score",
    "caption_length",
    "hashtags_count",
    "trending_audio",
]

CLASS_NAMES = {
    0: "Low",
    1: "Moderate",
    2: "Viral",
}

ALLOWED_IG_DOMAINS = {
    "instagram.com",
    "www.instagram.com",
    "m.instagram.com",
    "instagr.am",
    "www.instagr.am",
}


app = FastAPI(
    title=APP_TITLE,
    description="Short-form video success predictor with Instagram Reel metadata auto-fill.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    reel_length_sec: int = Field(..., ge=3, le=600)
    posting_time: int = Field(..., ge=0, le=23)
    hook_strength_score: float = Field(..., ge=0.0, le=1.0)
    caption_length: int = Field(..., ge=0, le=2200)
    hashtags_count: int = Field(..., ge=0, le=30)
    trending_audio: bool


class PredictResponse(BaseModel):
    virality_score: int
    predicted_class: str
    probabilities: Dict[str, float]
    feedback: str


class MetadataRequest(BaseModel):
    url: str = Field(..., min_length=12, max_length=500)

    @field_validator("url")
    @classmethod
    def validate_instagram_url(cls, value: str) -> str:
        cleaned = value.strip()
        parsed = urlparse(cleaned)

        if parsed.scheme not in {"http", "https"}:
            raise ValueError("URL must start with http or https.")

        domain = parsed.netloc.lower()
        if domain not in ALLOWED_IG_DOMAINS and not domain.endswith(".instagram.com"):
            raise ValueError("Please provide a valid Instagram Reel URL.")

        if not parsed.path:
            raise ValueError("URL path is missing.")

        return cleaned


class MetadataResponse(BaseModel):
    status: str
    reel_length_sec: Optional[int]
    caption_length: int
    hashtags_count: int
    message: str


class QuietYtdlpLogger:
    def debug(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "Reel Success Predictor API is running.",
    }


@lru_cache(maxsize=1)
def load_model_assets() -> Dict[str, Any]:
    """
    Loads either:
    1. Option A bundle: reeliq_model_bundle.pkl
    2. Original artifacts: xgb_model.pkl + preprocessor.pkl
    """
    if BUNDLE_PATH.exists():
        bundle = joblib.load(BUNDLE_PATH)

        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            raise RuntimeError("reeliq_model_bundle.pkl exists but has an invalid format.")

        return {
            "kind": "bundle",
            "pipeline": bundle["pipeline"],
            "features": bundle.get("features", FEATURES),
            "class_names": bundle.get("class_names", CLASS_NAMES),
        }

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run the ML notebook first to generate "
            "reeliq_model_bundle.pkl or xgb_model.pkl + preprocessor.pkl."
        )

    return {
        "kind": "separate",
        "model": joblib.load(MODEL_PATH),
        "preprocessor": joblib.load(PREPROCESSOR_PATH),
        "features": FEATURES,
        "class_names": CLASS_NAMES,
    }


def make_feature_frame(payload: PredictRequest) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reel_length_sec": payload.reel_length_sec,
                "posting_time": payload.posting_time,
                "hook_strength_score": payload.hook_strength_score,
                "caption_length": payload.caption_length,
                "hashtags_count": payload.hashtags_count,
                "trending_audio": int(payload.trending_audio),
            }
        ],
        columns=FEATURES,
    )


def normalize_class_name(raw_class: Any, class_names: Dict[Any, str]) -> str:
    if isinstance(raw_class, str) and raw_class in {"Low", "Moderate", "Viral"}:
        return raw_class

    try:
        return class_names[int(raw_class)]
    except Exception:
        return str(raw_class)


def predict_probabilities(payload: PredictRequest) -> Dict[str, float]:
    assets = load_model_assets()
    X = make_feature_frame(payload)

    if assets["kind"] == "bundle":
        pipeline = assets["pipeline"]

        expected_features = getattr(pipeline, "feature_names_in_", None)
        if expected_features is not None:
            X_for_model = X[list(expected_features)]
        else:
            X_for_model = X

        probabilities = pipeline.predict_proba(X_for_model)[0]

        model_step = getattr(pipeline, "named_steps", {}).get("model", pipeline)
        classes = getattr(model_step, "classes_", np.array([0, 1, 2]))
        class_names = assets.get("class_names", CLASS_NAMES)

    else:
        model = assets["model"]
        preprocessor = assets["preprocessor"]

        expected_preprocessor_features = getattr(preprocessor, "feature_names_in_", None)

        if expected_preprocessor_features is not None:
            X_for_preprocessor = X[list(expected_preprocessor_features)]
        else:
            X_for_preprocessor = X[
                [
                    "reel_length_sec",
                    "posting_time",
                    "hook_strength_score",
                    "caption_length",
                    "hashtags_count",
                ]
            ]

        X_processed = preprocessor.transform(X_for_preprocessor)

        model_n_features = getattr(model, "n_features_in_", None)

        if model_n_features == X_processed.shape[1] + 1:
            trending_audio_array = np.array([[int(payload.trending_audio)]])
            X_model_input = np.hstack([X_processed, trending_audio_array])
        else:
            X_model_input = X_processed

        probabilities = model.predict_proba(X_model_input)[0]

        classes = getattr(model, "classes_", np.array([0, 1, 2]))
        class_names = assets.get("class_names", CLASS_NAMES)

    prob_map = {"Low": 0.0, "Moderate": 0.0, "Viral": 0.0}

    for raw_class, probability in zip(classes, probabilities):
        label = normalize_class_name(raw_class, class_names)
        if label in prob_map:
            prob_map[label] = float(probability)

    total = sum(prob_map.values())
    if total > 0:
        prob_map = {key: value / total for key, value in prob_map.items()}

    return prob_map

def generate_feedback(payload: PredictRequest, probabilities: Dict[str, float]) -> str:
    tips = []

    if payload.hook_strength_score < 0.55:
        tips.append("Strengthen the first 3 seconds with a sharper curiosity gap, problem, or visual disruption.")
    elif payload.hook_strength_score >= 0.80:
        tips.append("Your hook score is strong; keep the opening fast and clear.")

    if not payload.trending_audio:
        tips.append("Consider testing a trending audio track if it fits the content naturally.")

    if payload.reel_length_sec > 90:
        tips.append("The Reel is relatively long; tighten pacing or split it into a shorter series.")
    elif payload.reel_length_sec < 8:
        tips.append("Very short Reels need an instantly clear payoff to avoid feeling incomplete.")

    if payload.caption_length > 450:
        tips.append("The caption is long; make the first line scannable and place the key idea early.")

    if payload.hashtags_count < 3:
        tips.append("Try adding a few relevant niche hashtags for discoverability.")
    elif payload.hashtags_count > 15:
        tips.append("Too many hashtags can look noisy; keep only the most relevant ones.")

    if payload.posting_time not in range(17, 23):
        tips.append("Consider testing early-evening posting times for your audience.")

    viral_pct = probabilities.get("Viral", 0.0) * 100

    if not tips:
        tips.append("Metadata looks balanced. The biggest remaining lever is creative quality in the opening seconds.")

    return f"Estimated viral probability: {viral_pct:.1f}%. " + " ".join(tips[:3])


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        probabilities = predict_probabilities(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    predicted_class = max(probabilities, key=probabilities.get)

    virality_score = int(
        round(
            100
            * (
                probabilities.get("Viral", 0.0)
                + 0.50 * probabilities.get("Moderate", 0.0)
            )
        )
    )
    virality_score = int(np.clip(virality_score, 0, 100))

    return PredictResponse(
        virality_score=virality_score,
        predicted_class=predicted_class,
        probabilities={
            "Low": round(probabilities.get("Low", 0.0), 6),
            "Moderate": round(probabilities.get("Moderate", 0.0), 6),
            "Viral": round(probabilities.get("Viral", 0.0), 6),
        },
        feedback=generate_feedback(payload, probabilities),
    )


def extract_reel_metadata_with_ytdlp(url: str) -> MetadataResponse:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
        "socket_timeout": 15,
        "retries": 2,
        "fragment_retries": 2,
        "cachedir": False,
        "logger": QuietYtdlpLogger(),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        if info is None:
            raise ValueError("yt-dlp returned no metadata.")

        info = ydl.sanitize_info(info)

    if isinstance(info, dict) and "entries" in info:
        entries = [entry for entry in info.get("entries", []) if entry]
        if entries:
            info = entries[0]

    duration = info.get("duration") if isinstance(info, dict) else None
    description = ""

    if isinstance(info, dict):
        description = (
            info.get("description")
            or info.get("title")
            or ""
        )

    reel_length_sec = None
    if duration is not None:
        try:
            reel_length_sec = int(round(float(duration)))
        except (TypeError, ValueError):
            reel_length_sec = None

    caption_length = len(description)
    hashtags_count = description.count("#")

    return MetadataResponse(
        status="ok",
        reel_length_sec=reel_length_sec,
        caption_length=caption_length,
        hashtags_count=hashtags_count,
        message="Metadata fetched successfully.",
    )


@app.post("/fetch-metadata", response_model=MetadataResponse)
async def fetch_metadata(payload: MetadataRequest) -> MetadataResponse:
    try:
        return await run_in_threadpool(extract_reel_metadata_with_ytdlp, payload.url)

    except (DownloadError, ExtractorError):
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not fetch metadata. The Reel may be private, unavailable, "
                "login-protected, region-limited, or temporarily blocked by Instagram."
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected metadata fetch error: {exc}",
        )


@app.get("/fetch-metadata", response_model=MetadataResponse)
async def fetch_metadata_get(
    url: str = Query(..., min_length=12, max_length=500)
) -> MetadataResponse:
    payload = MetadataRequest(url=url)
    return await fetch_metadata(payload)