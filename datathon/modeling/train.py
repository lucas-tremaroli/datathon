import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datathon.modeling.baseline import compute_feature_baselines
from datathon.modeling.config import FEATURE_COLUMNS
from datathon.modeling.evaluation import evaluate_model, get_feature_importance
from datathon.modeling.types import FeatureBaseline, ModelMetrics, TrainedModel

# Re-export all public names for backward compatibility
__all__ = [
    "FEATURE_COLUMNS",
    "FeatureBaseline",
    "ModelMetrics",
    "TrainedModel",
    "train",
    "get_feature_importance",
]


def train(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> TrainedModel:
    """
    Train classification model.

    Arguments:
        df: DataFrame with student data.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        TrainedModel with metrics.
    """
    # Prepare data
    X = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())
    y = (df['lag_next'] > df['lag_current']).astype(int)

    # Compute baseline distributions before scaling
    feature_baselines = compute_feature_baselines(X, FEATURE_COLUMNS)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train (regularized to reduce overfitting)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_scaled, y, X_test, y_test, random_state)

    return TrainedModel(
        model=model,
        scaler=scaler,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
        feature_baselines=feature_baselines,
    )
