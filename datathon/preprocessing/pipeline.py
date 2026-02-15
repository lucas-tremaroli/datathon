from datathon.db import Database
from datathon.preprocessing.transformations import rename_columns

def clean_and_store(year: int, db: Database):
    # Fetch the raw data for the given year
    raw_data = db.fetch_table(f'raw.data_{year}')

    # Clean the data (e.g., rename columns)
    cleaned_data = rename_columns(raw_data)

    # Store the cleaned data back to the database
    db.conn.execute(f"CREATE OR REPLACE TABLE refined.data_{year} AS SELECT * FROM cleaned_data;")

if __name__ == "__main__":
    with Database('data/duckdb/datathon.db') as db:
        for year in range(2022, 2025):
            clean_and_store(year, db)
