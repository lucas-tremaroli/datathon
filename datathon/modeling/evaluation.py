import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from datathon.modeling.types import ModelMetrics, TrainedModel


def evaluate_model(
    model: RandomForestClassifier,
    X_scaled: np.ndarray,
    y: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    random_state: int,
) -> ModelMetrics:
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba: np.ndarray = model.predict_proba(X_test)
    y_proba = y_proba[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')

    return ModelMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred)),
        recall=float(recall_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred)),
        auc_roc=float(roc_auc_score(y_test, y_proba)),
        cv_f1_mean=float(cv_scores.mean()),
        cv_f1_std=float(cv_scores.std()),
    )


def get_feature_importance(trained: TrainedModel) -> pd.DataFrame:
    """Get feature importance ranking."""
    return pd.DataFrame({
        'feature': trained.feature_columns,
        'importance': trained.model.feature_importances_,
    }).sort_values('importance', ascending=False)
