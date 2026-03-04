import pandas as pd

from datathon.preprocessing.encoding import ENCODED_CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def round_numeric_columns(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """
    Round all numeric columns to a consistent number of decimal places.

    Arguments:
        df: The DataFrame with numeric columns.
        decimals: Number of decimal places to round to.

    Returns:
        A DataFrame with rounded numeric values.
    """
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round(decimals)

    return df


def impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute null values in the DataFrame.

    - Encoded categorical columns: impute with mode (most frequent)
    - Numeric columns: impute with median

    Arguments:
        df: The DataFrame with null values.

    Returns:
        A DataFrame with null values imputed.
    """
    df = df.copy()

    # Impute encoded categorical columns with mode
    for col in ENCODED_CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

    # Impute numeric columns with median
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df
