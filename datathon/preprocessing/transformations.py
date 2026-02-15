import pandas as pd

from datathon.preprocessing.mapping import (
    COLUMN_MAPPING_2022,
    COLUMN_MAPPING_2023,
    COLUMN_MAPPING_2024,
)

def rename_columns(year: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to a consistent format.

    Arguments:
        df: The DataFrame with original column names.

    Returns:
        A DataFrame with renamed columns according to COLUMN_MAPPING.
    """
    if year == 2022:
        return df.rename(columns=COLUMN_MAPPING_2022)
    if year == 2023:
        return df.rename(columns=COLUMN_MAPPING_2023)
    return df.rename(columns=COLUMN_MAPPING_2024)

def normalize_fase(series: pd.Series) -> pd.Series:
    """Convert all Fase formats to int."""
    return (
        series.astype(str)
        .str.replace('ALFA', '0', regex=False)
        .str.replace('FASE ', '', regex=False)
        .str.extract(r'^(\d+)')[0]
        .astype(int)
    )
