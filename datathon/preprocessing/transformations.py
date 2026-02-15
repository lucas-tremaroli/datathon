import pandas as pd

from datathon.preprocessing.columns import COLUMN_MAPPING

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to a consistent format.

    Arguments:
        df: The DataFrame with original column names.

    Returns:
        A DataFrame with renamed columns according to COLUMN_MAPPING.
    """
    return df.rename(columns=COLUMN_MAPPING)

def normalize_fase(series: pd.Series) -> pd.Series:
    """Convert all Fase formats to int."""
    return (
        series.astype(str)
        .str.replace('ALFA', '0', regex=False)
        .str.replace('FASE ', '', regex=False)
        .str.extract(r'^(\d+)')[0]
        .astype(int)
    )
