from datathon.db import DuckDBClient
from datathon.preprocessing.transformations import (
    detect_outliers_iqr,
    drop_columns,
    impute_nulls,
    rename_columns,
    render_outlier_boxplots,
    round_numeric_columns,
    standardize_dtypes,
    standardize_education_institution,
    standardize_gender,
    treat_outliers_iqr,
)

def clean_and_store_refined_table(year: int, db: DuckDBClient) -> None:
    """
    Cleans the raw data for a given year and stores it as a refined table in the database.

    Arguments:
        year: The year for which to clean and store the data.
        db: An instance of the Database class to interact with the database.
    """
    # Fetch the raw data for the specified year
    raw_data = db.fetch_table(f'raw.data_{year}')

    # Clean the data
    cleaned_data = rename_columns(year, raw_data)
    cleaned_data = standardize_gender(year, cleaned_data)
    cleaned_data = standardize_education_institution(year, cleaned_data)
    cleaned_data = drop_columns(year, cleaned_data)

    # Store the cleaned data back to the database
    with open('data/queries/create_refined_table.sql', 'r') as f:
        create_table_query = f.read().format(year=year)
        db.execute_query(create_table_query, cleaned_data)

def merge_refined_tables(db: DuckDBClient) -> None:
    """
    Merges all refined tables into a single table for analysis.

    Arguments:
        db: An instance of the Database class to interact with the database.
    """
    with open('data/queries/merge_refined_tables.sql', 'r') as f:
        merge_query = f.read()
        db.execute_query(merge_query)


def prepare_students_for_training(db: DuckDBClient) -> None:
    """
    Prepares the refined.students table for ML training:
    1. Standardize data types (convert to numeric, encode categoricals)
    2. Detect and report outliers using IQR method (before imputation)
    3. Treat outliers using winsorization (before imputation)
    4. Impute null values (median for numeric, mode for categorical)
    5. Round numeric columns to 2 decimal places

    Arguments:
        db: An instance of the Database class to interact with the database.
    """
    students = db.fetch_table('refined.students')
    students = standardize_dtypes(students)

    # Outlier detection, reporting, and treatment (BEFORE imputation)
    # This ensures we analyze actual data distribution, not imputed values
    outlier_report = detect_outliers_iqr(students)
    print(outlier_report)
    render_outlier_boxplots(students, outlier_report, output_dir="reports")
    students = treat_outliers_iqr(students)

    students = impute_nulls(students)
    students = round_numeric_columns(students)
    db.execute_query(
        "CREATE OR REPLACE TABLE refined.students AS SELECT * FROM temp_table",
        students
    )

def run_pipeline() -> None:
    """
    Runs the entire data preprocessing pipeline:
    1. Clean and store refined tables for each year
    2. Merge all refined tables into a single students table
    3. Standardize data types and impute null values
    """
    with DuckDBClient('data/duckdb/datathon.db') as db:
        # Clean and store refined tables for each year
        for year in range(2022, 2025):
            clean_and_store_refined_table(year, db)
        # Merge all refined tables into a single table
        merge_refined_tables(db)
        # Standardize types and impute nulls
        prepare_students_for_training(db)

if __name__ == "__main__":
    run_pipeline()
