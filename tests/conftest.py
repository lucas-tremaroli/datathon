import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from datathon.api.main import app
from datathon.database.client import DuckDBClient
from datathon.modeling.config import FEATURE_COLUMNS


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_student():
    return {
        "ieg": 7.5,
        "iaa": 8.0,
        "ips": 6.5,
        "ida": 7.2,
        "ian": 8.5,
        "ipv": 6.0,
        "inde": 7.3,
        "stone": 3,
        "age": 14,
    }


@pytest.fixture
def outlier_df():
    """DataFrame with known outliers."""
    np.random.seed(42)
    normal_data = np.random.normal(50, 5, 100).tolist()
    normal_data.append(200.0)  # extreme outlier
    normal_data.append(-100.0)  # extreme outlier
    return pd.DataFrame({"age": normal_data})


@pytest.fixture
def training_df():
    """Synthetic DataFrame with enough rows for stratified split."""
    np.random.seed(42)
    n = 200
    data = {col: np.random.uniform(0, 10, n) for col in FEATURE_COLUMNS}
    data["lag_current"] = np.random.uniform(0, 3, n)
    data["lag_next"] = data["lag_current"] + np.random.choice([-1, 0, 1], n, p=[0.3, 0.3, 0.4])
    return pd.DataFrame(data)


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    with DuckDBClient(db_path) as client:
        client.execute_query("CREATE TABLE test_table (id INT, name VARCHAR)")
        client.execute_query("INSERT INTO test_table VALUES (1, 'alice'), (2, 'bob')")
        yield client
