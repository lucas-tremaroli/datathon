"""Retrain the model and save to disk."""

from datathon.database.client import DuckDBClient
from datathon.modeling.train import train

MODEL_PATH = "models/lag_worsening.pkl"
DB_PATH = "data/duckdb/datathon.db"


def main() -> None:
    with DuckDBClient(DB_PATH, read_only=True) as db:
        df = db.fetch_table("refined.students")

    print(f"Training on {len(df)} rows...")
    model = train(df)

    print(f"Metrics: F1={model.metrics.f1:.4f}, AUC={model.metrics.auc_roc:.4f}")
    print(f"Baselines computed for {len(model.feature_baselines)} features")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
