import logging

import pandas as pd
from fastapi import APIRouter

from datathon.api.models.predict import (
    PredictRequest,
    PredictResponse,
    StudentFeatures,
    StudentPrediction,
)
from datathon.api.util.metrics import (
    FEATURE_VALUE,
    PREDICTION_PROBABILITY,
    PREDICTIONS_TOTAL,
    drift_monitor,
)
from datathon.api.util.model import get_model

logger = logging.getLogger("datathon.api.predict")

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict whether students' lag will worsen.

    Returns a probability score and binary prediction for each student.
    """
    model = get_model()

    df = pd.DataFrame([s.model_dump() for s in request.students])
    logger.info("Batch prediction request: %d students", len(df))

    probabilities = model.predict_proba(df)
    predictions = model.predict(df)

    worsen_count = int(predictions.sum())
    logger.info(
        "Batch prediction result: %d/%d predicted to worsen",
        worsen_count,
        len(predictions),
    )

    _observe_predictions(df, probabilities, predictions)

    return PredictResponse(
        predictions=[
            StudentPrediction(will_worsen=bool(pred), probability=float(prob))
            for pred, prob in zip(predictions, probabilities)
        ]
    )


@router.post("/predict/single", response_model=StudentPrediction)
async def predict_single(student: StudentFeatures) -> StudentPrediction:
    """
    Predict whether a single student's lag will worsen.
    """
    model = get_model()

    df = pd.DataFrame([student.model_dump()])
    logger.info("Single prediction request: %s", student.model_dump())

    probability = float(model.predict_proba(df)[0])
    will_worsen = bool(model.predict(df)[0])

    logger.info(
        "Single prediction result: will_worsen=%s, probability=%.4f",
        will_worsen,
        probability,
    )

    import numpy as np
    _observe_predictions(df, np.array([probability]), np.array([int(will_worsen)]))

    return StudentPrediction(will_worsen=will_worsen, probability=probability)


def _observe_predictions(
    df: pd.DataFrame,
    probabilities: "np.ndarray",
    predictions: "np.ndarray",
) -> None:
    """Record Prometheus metrics and feed drift monitor for a batch of predictions."""
    import datathon.api.util.metrics as m

    for prob in probabilities:
        PREDICTION_PROBABILITY.observe(float(prob))

    for pred in predictions:
        outcome = "worsen" if pred else "stable"
        PREDICTIONS_TOTAL.labels(outcome=outcome).inc()

    for _, row in df.iterrows():
        features = {}
        for col in df.columns:
            val = float(row[col])
            FEATURE_VALUE.labels(feature=col).observe(val)
            features[col] = val
        if m.drift_monitor is not None:
            m.drift_monitor.observe(features)
