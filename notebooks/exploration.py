# %%
import pandas as pd

from datathon.db import DuckDBClient

with DuckDBClient("../data/duckdb/datathon.db") as db:
      students = db.fetch_table('refined.students')
      nulls = pd.DataFrame({
            'null_count': students.isnull().sum(),
            'variance': students.var(numeric_only=True)
      })
      for col in students.columns:
            print(f"Column: {col}")
            print(f"Null count: {nulls.loc[col, 'null_count']}")
            print(f"Variance: {nulls.loc[col, 'variance']}")
            print("-" * 40)
