import logging
from threading import Lock

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Info

from datathon.modeling.train import FeatureBaseline

logger = logging.getLogger("datathon.api.metrics")

# --- Prometheus metrics ---

REQUEST_DURATION = Histogram(
    "request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=["method", "path", "status"],
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of predictions",
    labelnames=["outcome"],
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of prediction probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

FEATURE_VALUE = Histogram(
    "feature_value",
    "Distribution of input feature values",
    labelnames=["feature"],
)

MODEL_INFO = Info(
    "model",
    "Model metadata and performance",
)

FEATURE_DRIFT_PSI = Gauge(
    "feature_drift_psi",
    "Population Stability Index per feature",
    labelnames=["feature"],
)

# --- PSI Calculation ---

PSI_WARNING_THRESHOLD = 0.2
PSI_ALERT_THRESHOLD = 0.25


def calculate_psi(
    baseline_counts: np.ndarray,
    observed_counts: np.ndarray,
) -> float:
    """Calculate Population Stability Index between two distributions.

    Both inputs are histogram bin counts (same number of bins).
    """
    # Normalize to proportions
    baseline_prop = baseline_counts / baseline_counts.sum()
    observed_prop = observed_counts / observed_counts.sum()

    # Replace zeros to avoid log(0)
    eps = 1e-6
    baseline_prop = np.clip(baseline_prop, eps, None)
    observed_prop = np.clip(observed_prop, eps, None)

    psi = np.sum((observed_prop - baseline_prop) * np.log(observed_prop / baseline_prop))
    return float(psi)


# --- Drift Monitor ---


class DriftMonitor:
    """Accumulates incoming feature values and computes PSI against baselines."""

    def __init__(self, baselines: dict[str, FeatureBaseline]) -> None:
        self._baselines = baselines
        self._buffers: dict[str, list[float]] = {name: [] for name in baselines}
        self._lock = Lock()

    def observe(self, features: dict[str, float]) -> None:
        """Record a single observation of feature values."""
        with self._lock:
            for name, value in features.items():
                if name in self._buffers:
                    self._buffers[name].append(value)

    def compute_drift(self) -> dict:
        """Compute PSI for each feature against baseline.

        Returns dict with per-feature PSI, overall status, and sample count.
        """
        with self._lock:
            buffers_snapshot = {k: list(v) for k, v in self._buffers.items()}

        if not buffers_snapshot or not any(buffers_snapshot.values()):
            return {
                "status": "no_data",
                "sample_count": 0,
                "features": {},
            }

        sample_count = max(len(v) for v in buffers_snapshot.values())
        feature_psi: dict[str, float] = {}
        max_psi = 0.0

        for name, baseline in self._baselines.items():
            values = buffers_snapshot.get(name, [])
            if len(values) < 10:
                continue

            observed_counts, _ = np.histogram(values, bins=baseline.bin_edges)
            if observed_counts.sum() == 0:
                continue

            psi = calculate_psi(baseline.bin_counts, observed_counts)
            feature_psi[name] = round(psi, 6)
            max_psi = max(max_psi, psi)

            FEATURE_DRIFT_PSI.labels(feature=name).set(psi)

            if psi > PSI_WARNING_THRESHOLD:
                logger.warning(
                    "Drift detected for feature '%s': PSI=%.4f (threshold=%.2f)",
                    name,
                    psi,
                    PSI_WARNING_THRESHOLD,
                )

        if max_psi > PSI_ALERT_THRESHOLD:
            status = "alert"
        elif max_psi > PSI_WARNING_THRESHOLD:
            status = "warning"
        else:
            status = "no_drift"

        return {
            "status": status,
            "sample_count": sample_count,
            "features": feature_psi,
        }


# Global drift monitor (initialized on model load)
drift_monitor: DriftMonitor | None = None
