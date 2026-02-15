from datathon.db import Database
from datathon.preprocessing.transformations import rename_columns

def clean_and_store_refined_table(year: int, db: Database) -> None:
    """
    Cleans the raw data for a given year and stores it as a refined table in the database.

    Arguments:
        year: The year for which to clean and store the data.
        db: An instance of the Database class to interact with the database.
    """
    # Fetch the raw data for the specified year
    raw_data = db.fetch_table(f'raw.data_{year}')

    # Clean the data (e.g., rename columns)
    cleaned_data = rename_columns(raw_data)

    # Store the cleaned data back to the database
    with open('data/queries/create_refined_table.sql', 'r') as f:
        create_table_query = f.read().format(year=year)
        db.execute_query(create_table_query, cleaned_data)

def merge_refined_tables(db: Database) -> None:
    """
    Merges all refined tables into a single table for analysis.

    Arguments:
        db: An instance of the Database class to interact with the database.
    """
    with open('data/queries/merge_refined_tables.sql', 'r') as f:
        merge_query = f.read()
        db.execute_query(merge_query)

def run_pipeline() -> None:
    """
    Runs the entire data preprocessing pipeline:
    """
    with Database('data/duckdb/datathon.db') as db:
        # Clean and store refined tables for each year
        for year in range(2022, 2025):
            clean_and_store_refined_table(year, db)
        # Merge all refined tables into a single table
        merge_refined_tables(db)

if __name__ == "__main__":
    run_pipeline()
