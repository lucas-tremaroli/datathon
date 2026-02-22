from fastapi import APIRouter

from datathon.api.util.model import get_model
from datathon.api.models.model import (
    FeatureImportance,
    ModelInfoResponse,
    ModelMetricsResponse,
)
from datathon.modeling.train import get_feature_importance

router = APIRouter(prefix="/api", tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """
    Get model information including metrics and feature importance.
    """
    model = get_model()

    importance_df = get_feature_importance(model)
    feature_importance = [
        FeatureImportance(feature=row["feature"], importance=float(row["importance"]))
        for _, row in importance_df.iterrows()
    ]

    metrics = ModelMetricsResponse(
        accuracy=model.metrics.accuracy,
        precision=model.metrics.precision,
        recall=model.metrics.recall,
        f1=model.metrics.f1,
        auc_roc=model.metrics.auc_roc,
        cv_f1_mean=model.metrics.cv_f1_mean,
        cv_f1_std=model.metrics.cv_f1_std,
    )

    return ModelInfoResponse(
        feature_columns=model.feature_columns,
        feature_importance=feature_importance,
        metrics=metrics,
    )
