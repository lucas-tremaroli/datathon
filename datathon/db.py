import duckdb
from pandas import DataFrame


class Database:
    def __init__(self, db_path: str):
        self.conn = duckdb.connect(database=db_path)

    def fetch_table(self, table_name: str) -> DataFrame:
        query = f"SELECT * FROM {table_name};"
        return self.conn.execute(query).df()

    def execute_query(self, query: str) -> DataFrame:
        return self.conn.execute(query).df()
