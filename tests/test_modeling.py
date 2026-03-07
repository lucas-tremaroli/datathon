import numpy as np
import pandas as pd
import pytest

from datathon.modeling.baseline import compute_feature_baselines
from datathon.modeling.config import FEATURE_COLUMNS
from datathon.modeling.evaluation import get_feature_importance
from datathon.modeling.train import train
from datathon.modeling.types import FeatureBaseline, ModelMetrics, TrainedModel


# ── baseline ──────────────────────────────────────────────────────────────


class TestComputeFeatureBaselines:
    def test_returns_baselines_for_all_columns(self):
        df = pd.DataFrame({
            "a": np.random.normal(5, 1, 50),
            "b": np.random.normal(10, 2, 50),
        })
        baselines = compute_feature_baselines(df, ["a", "b"])
        assert set(baselines.keys()) == {"a", "b"}

    def test_baseline_values_correct(self):
        values = np.arange(1.0, 101.0)
        df = pd.DataFrame({"x": values})
        baselines = compute_feature_baselines(df, ["x"])
        bl = baselines["x"]
        assert bl.mean == pytest.approx(50.5)
        assert bl.min == 1.0
        assert bl.max == 100.0
        assert bl.q50 == pytest.approx(50.5)
        assert len(bl.bin_edges) == 11
        assert len(bl.bin_counts) == 10

    def test_handles_nan(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
        baselines = compute_feature_baselines(df, ["x"])
        assert baselines["x"].mean == pytest.approx(7.0 / 3)


# ── evaluation ────────────────────────────────────────────────────────────


class TestEvaluateModel:
    def test_returns_model_metrics(self, training_df):
        trained = train(training_df)
        assert isinstance(trained.metrics, ModelMetrics)
        assert 0.0 <= trained.metrics.accuracy <= 1.0
        assert 0.0 <= trained.metrics.f1 <= 1.0
        assert 0.0 <= trained.metrics.auc_roc <= 1.0
        assert trained.metrics.cv_f1_std >= 0.0


class TestGetFeatureImportance:
    def test_returns_dataframe_with_all_features(self, training_df):
        trained = train(training_df)
        importance = get_feature_importance(trained)
        assert isinstance(importance, pd.DataFrame)
        assert set(importance.columns) == {"feature", "importance"}
        assert len(importance) == len(FEATURE_COLUMNS)
        assert importance["importance"].sum() == pytest.approx(1.0)


# ── train ─────────────────────────────────────────────────────────────────


class TestTrain:
    def test_returns_trained_model(self, training_df):
        result = train(training_df)
        assert isinstance(result, TrainedModel)
        assert result.feature_columns == FEATURE_COLUMNS
        assert result.scaler is not None
        assert result.model is not None

    def test_baselines_populated(self, training_df):
        result = train(training_df)
        assert len(result.feature_baselines) == len(FEATURE_COLUMNS)
        for col in FEATURE_COLUMNS:
            assert isinstance(result.feature_baselines[col], FeatureBaseline)


# ── types ─────────────────────────────────────────────────────────────────


class TestTrainedModel:
    def test_predict_proba(self, training_df):
        trained = train(training_df)
        proba = trained.predict_proba(training_df)
        assert len(proba) == len(training_df)
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_predict_binary(self, training_df):
        trained = train(training_df)
        preds = trained.predict(training_df)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_custom_threshold(self, training_df):
        trained = train(training_df)
        preds_low = trained.predict(training_df, threshold=0.1)
        preds_high = trained.predict(training_df, threshold=0.9)
        assert preds_low.sum() >= preds_high.sum()

    def test_save_and_load(self, training_df, tmp_path):
        trained = train(training_df)
        model_path = tmp_path / "model.pkl"
        trained.save(model_path)
        assert model_path.exists()

        loaded = TrainedModel.load(model_path)
        assert isinstance(loaded, TrainedModel)
        assert loaded.feature_columns == trained.feature_columns
        np.testing.assert_array_almost_equal(
            loaded.predict_proba(training_df),
            trained.predict_proba(training_df),
        )

    def test_save_creates_parent_dirs(self, training_df, tmp_path):
        trained = train(training_df)
        nested_path = tmp_path / "a" / "b" / "model.pkl"
        trained.save(nested_path)
        assert nested_path.exists()
