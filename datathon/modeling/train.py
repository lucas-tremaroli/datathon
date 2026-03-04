from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


# Features based on PEDE framework
FEATURE_COLUMNS = [
    'ieg',   # Engagement (leading indicator)
    'iaa',   # Self-assessment (leading indicator)
    'ips',   # Psychosocial (leading indicator)
    'ida',   # Academic performance
    'ian',   # Level adequacy
    'ipv',   # Turning point
    'inde',  # Overall development index
    'stone', # Current stone category
    'age',   # Student age
]


@dataclass
class FeatureBaseline:
    """Baseline distribution statistics for a single feature."""

    mean: float
    std: float
    min: float
    max: float
    q25: float
    q50: float
    q75: float
    bin_edges: np.ndarray
    bin_counts: np.ndarray


@dataclass
class ModelMetrics:
    """Evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    cv_f1_mean: float
    cv_f1_std: float


@dataclass
class TrainedModel:
    """Trained model container."""

    model: RandomForestClassifier
    scaler: StandardScaler
    feature_columns: list[str]
    metrics: ModelMetrics
    feature_baselines: dict[str, FeatureBaseline] = field(default_factory=dict)

    def __getattr__(self, name: str):
        if name == "feature_baselines":
            return {}
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of lag worsening."""
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        X_scaled = self.scaler.transform(X)
        proba: np.ndarray = self.model.predict_proba(X_scaled)
        return proba[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary outcome."""
        return (self.predict_proba(df) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> 'TrainedModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


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
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )

    # Prepare data
    X = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())
    y = (df['lag_next'] > df['lag_current']).astype(int)

    # Compute baseline distributions before scaling
    feature_baselines: dict[str, FeatureBaseline] = {}
    for col in FEATURE_COLUMNS:
        values = X[col].dropna().values
        counts, edges = np.histogram(values, bins=10)
        feature_baselines[col] = FeatureBaseline(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            q25=float(np.percentile(values, 25)),
            q50=float(np.percentile(values, 50)),
            q75=float(np.percentile(values, 75)),
            bin_edges=edges,
            bin_counts=counts,
        )

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
    y_pred = model.predict(X_test)
    y_proba: np.ndarray = model.predict_proba(X_test)
    y_proba = y_proba[:, 1]

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')

    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred)),
        recall=float(recall_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred)),
        auc_roc=float(roc_auc_score(y_test, y_proba)),
        cv_f1_mean=float(cv_scores.mean()),
        cv_f1_std=float(cv_scores.std()),
    )

    return TrainedModel(
        model=model,
        scaler=scaler,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
        feature_baselines=feature_baselines,
    )


def get_feature_importance(trained: TrainedModel) -> pd.DataFrame:
    """Get feature importance ranking."""
    return pd.DataFrame({
        'feature': trained.feature_columns,
        'importance': trained.model.feature_importances_,
    }).sort_values('importance', ascending=False)
