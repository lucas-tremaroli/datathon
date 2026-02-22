from pathlib import Path
from fastapi import HTTPException

from datathon.modeling.train import TrainedModel

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
