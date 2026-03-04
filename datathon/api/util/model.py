import logging
from pathlib import Path

from fastapi import HTTPException

from datathon.modeling.train import TrainedModel

logger = logging.getLogger("datathon.api.model")

MODEL_PATH = Path(__file__).parents[3] / "models" / "lag_worsening.pkl"

_model: TrainedModel | None = None


def get_model() -> TrainedModel:
    """Load the trained model (cached)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            logger.error("Model file not found at %s", MODEL_PATH)
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please train the model first.",
            )
        logger.info("Loading model from %s", MODEL_PATH)
        _model = TrainedModel.load(MODEL_PATH)
        logger.info(
            "Model loaded successfully (features=%s, f1=%.4f)",
            len(_model.feature_columns),
            _model.metrics.f1,
        )
        _init_metrics(_model)
    return _model


def _init_metrics(model: TrainedModel) -> None:
    """Set Prometheus model info and initialize drift monitor."""
    import datathon.api.util.metrics as m

    m.MODEL_INFO.info({
        "f1": f"{model.metrics.f1:.4f}",
        "accuracy": f"{model.metrics.accuracy:.4f}",
        "precision": f"{model.metrics.precision:.4f}",
        "recall": f"{model.metrics.recall:.4f}",
        "auc_roc": f"{model.metrics.auc_roc:.4f}",
        "features": str(len(model.feature_columns)),
    })

    baselines = model.feature_baselines
    if baselines:
        m.drift_monitor = m.DriftMonitor(baselines)
        logger.info("Drift monitor initialized with %d feature baselines", len(baselines))
    else:
        logger.warning("No feature baselines in model artifact; drift monitoring disabled")
