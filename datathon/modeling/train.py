from dataclasses import dataclass
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

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of lag worsening."""
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

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
    y_proba = model.predict_proba(X_test)[:, 1]

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')

    metrics = ModelMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        auc_roc=roc_auc_score(y_test, y_proba),
        cv_f1_mean=cv_scores.mean(),
        cv_f1_std=cv_scores.std(),
    )

    return TrainedModel(
        model=model,
        scaler=scaler,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
    )


def get_feature_importance(trained: TrainedModel) -> pd.DataFrame:
    """Get feature importance ranking."""
    return pd.DataFrame({
        'feature': trained.feature_columns,
        'importance': trained.model.feature_importances_,
    }).sort_values('importance', ascending=False)
