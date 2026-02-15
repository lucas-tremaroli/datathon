# %%

from datathon.db import DuckDBClient

with DuckDBClient("../data/duckdb/datathon.db") as db:
      students = db.fetch_table('refined.students')
      x = students['age_22'].count()
      print(x)
