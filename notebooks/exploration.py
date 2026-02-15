# %%

from datathon.db import Database
from datathon.preprocessing.columns import COLUMN_MAPPING

with Database("../data/duckdb/datathon.db") as db:
      raw_data_2024 = db.fetch_table('raw_data_2024')

refined_data_2024 = raw_data_2024.copy()

# %%

# Rename columns
refined_data_2024.rename(
    columns=COLUMN_MAPPING,
    inplace=True
)

refined_data_2024.columns
