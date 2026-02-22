import duckdb
from pandas import DataFrame


class DuckDBClient:
    def __init__(self, db_path: str, read_only: bool = False):
        self.conn = duckdb.connect(database=db_path, read_only=read_only)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def fetch_table(self, table_name: str) -> DataFrame:
        """
        Fetches an entire table from the database and returns it as a DataFrame.

        Arguments:
            table_name: The name of the table to fetch.

        Returns:
            A DataFrame containing the contents of the specified table.
        """
        query = f"SELECT * FROM {table_name};"
        return self.conn.execute(query).df()

    def execute_query(self, query: str, df: DataFrame = None) -> DataFrame:
        """
        Executes a SQL query against the database.

        Arguments:
            query: The SQL query to execute.
            df: An optional DataFrame to register as a temporary table for the query.

        Returns:
            A DataFrame containing the results of the query.
        """
        if df is not None:
            self.conn.register('temp_table', df)
            try:
                result = self.conn.execute(query).df()
            finally:
                self.conn.unregister('temp_table')
        else:
            result = self.conn.execute(query).df()
        return result
