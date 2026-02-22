from pydantic import BaseModel


class FeatureImportance(BaseModel):
    """
    Feature importance entry.
    """
    feature: str
    importance: float


class ModelMetricsResponse(BaseModel):
    """
    Model performance metrics.
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    cv_f1_mean: float
    cv_f1_std: float


class ModelInfoResponse(BaseModel):
    """
    Response model for the model info endpoint.
    """
    feature_columns: list[str]
    feature_importance: list[FeatureImportance]
    metrics: ModelMetricsResponse
