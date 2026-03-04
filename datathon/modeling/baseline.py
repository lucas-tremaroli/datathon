import numpy as np
import pandas as pd

from datathon.modeling.types import FeatureBaseline


def compute_feature_baselines(
    X: pd.DataFrame, feature_columns: list[str]
) -> dict[str, FeatureBaseline]:
    """Compute baseline distribution statistics for each feature."""
    baselines: dict[str, FeatureBaseline] = {}
    for col in feature_columns:
        values = X[col].dropna().values
        counts, edges = np.histogram(values, bins=10)
        baselines[col] = FeatureBaseline(
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
    return baselines
