"""FastAPI service exposing the fraud detection model.

Wraps FraudPredictor behind a small HTTP API:
- GET  /health      liveness check
- GET  /model/info  model metadata and metrics
- POST /predict     score a single Ethereum address

The predictor is loaded once at startup and reused across requests.
"""

import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.models.predict import FraudPredictor

logger = logging.getLogger(__name__)

ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# Holds the loaded predictor. Populated on startup, reused per request.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the service starts."""
    try:
        state["predictor"] = FraudPredictor()
        logger.info("FraudPredictor loaded")
    except FileNotFoundError:
        # Let the service start so /health works, but /predict will 503.
        state["predictor"] = None
        logger.warning("No trained model found. /predict will be unavailable.")
    yield
    state.clear()


app = FastAPI(
    title="Crypto Fraud Detector",
    description="Score Ethereum wallet addresses for fraud risk using graph-based ML.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    address: str = Field(
        ...,
        description="Ethereum address (0x followed by 40 hex characters).",
        examples=["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"],
    )

    @field_validator("address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        v = v.strip()
        if not ETH_ADDRESS_RE.match(v):
            raise ValueError(
                "Invalid Ethereum address: must be 0x followed by 40 hex characters."
            )
        return v


class Factor(BaseModel):
    feature: str
    value: float
    direction: str
    shap_value: float


class PredictResponse(BaseModel):
    address: str
    fraud_probability: float
    prediction: str
    risk_level: str
    top_factors: list[Factor]
    model_type: str


@app.get("/health")
def health() -> dict:
    """Liveness check. Reports whether the model is loaded."""
    return {"status": "ok", "model_loaded": state.get("predictor") is not None}


@app.get("/model/info")
def model_info() -> dict:
    """Return model metadata and evaluation metrics."""
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "best_model": predictor.meta["best_model"],
        "feature_count": len(predictor.feature_names),
        "feature_names": predictor.feature_names,
        "metrics": predictor.meta.get("metrics", {}),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    """Score a single Ethereum address for fraud risk."""
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        result = predictor.score_address(request.address)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scoring failed for %s", request.address)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to score address: {exc}",
        ) from exc

    return result
