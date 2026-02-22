import pandas as pd
from fastapi import APIRouter

from datathon.api.util.model import get_model
from datathon.api.models.predict import (
    PredictRequest,
    PredictResponse,
    StudentFeatures,
    StudentPrediction,
)

router = APIRouter(prefix="/api", tags=["predict"])


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


@router.post("/predict/single", response_model=StudentPrediction)
async def predict_single(student: StudentFeatures) -> StudentPrediction:
    """
    Predict whether a single student's lag will worsen.
    """
    model = get_model()

    df = pd.DataFrame([student.model_dump()])

    probability = float(model.predict_proba(df)[0])
    will_worsen = bool(model.predict(df)[0])

    return StudentPrediction(will_worsen=will_worsen, probability=probability)
