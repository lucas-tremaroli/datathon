from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from datathon.api.models.predict import (
    PredictRequest,
    PredictResponse,
    StudentPrediction,
)
from datathon.modeling.train import TrainedModel

router = APIRouter(prefix="/api", tags=["predict"])

MODEL_PATH = Path(__file__).parents[3] / "models" / "lag_worsening.pkl"

_model: TrainedModel | None = None


def get_model() -> TrainedModel:
    """Load the trained model (cached)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please train the model first.",
            )
        _model = TrainedModel.load(MODEL_PATH)
    return _model


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict whether students' lag will worsen.

    Returns a probability score and binary prediction for each student.
    """
    model = get_model()

    df = pd.DataFrame([s.model_dump() for s in request.students])

    probabilities = model.predict_proba(df)
    predictions = model.predict(df)

    return PredictResponse(
        predictions=[
            StudentPrediction(will_worsen=bool(pred), probability=float(prob))
            for pred, prob in zip(predictions, probabilities)
        ]
    )
