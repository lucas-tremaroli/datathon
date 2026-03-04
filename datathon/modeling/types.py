from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


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
