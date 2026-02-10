# %%

from datathon.db import Database
from datathon.preprocessing.columns import COLUMN_MAPPING

with Database("../data/raw/datathon.db") as db:
      raw_data_2022 = db.fetch_table('raw_data_2022')

refined_data_2022 = raw_data_2022.copy()

# %%

# Rename columns
refined_data_2022.rename(
    columns=COLUMN_MAPPING,
    inplace=True
)
