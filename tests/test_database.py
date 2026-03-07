import pandas as pd
import pytest

from datathon.database.client import DuckDBClient


class TestDuckDBClient:
    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx.db")
        with DuckDBClient(db_path) as client:
            assert client.conn is not None

    def test_fetch_table(self, db):
        result = db.fetch_table("test_table")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "name"]

    def test_execute_query_select(self, db):
        result = db.execute_query("SELECT * FROM test_table WHERE id = 1")
        assert len(result) == 1
        assert result.iloc[0]["name"] == "alice"

    def test_execute_query_with_dataframe(self, db):
        df = pd.DataFrame({"x": [10, 20], "y": ["a", "b"]})
        result = db.execute_query("SELECT * FROM temp_table WHERE x > 15", df)
        assert len(result) == 1
        assert result.iloc[0]["y"] == "b"

    def test_temp_table_unregistered_after_use(self, db):
        df = pd.DataFrame({"x": [1]})
        db.execute_query("SELECT * FROM temp_table", df)
        with pytest.raises(Exception):
            db.execute_query("SELECT * FROM temp_table")
