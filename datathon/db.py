import duckdb
from pandas import DataFrame


class Database:
    def __init__(self, db_path: str, read_only: bool = False):
        self.conn = duckdb.connect(database=db_path, read_only=read_only)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def fetch_table(self, table_name: str) -> DataFrame:
        query = f"SELECT * FROM {table_name};"
        return self.conn.execute(query).df()

    def execute_query(self, query: str) -> DataFrame:
        return self.conn.execute(query).df()
