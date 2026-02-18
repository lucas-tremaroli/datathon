import pandas as pd

from datathon.preprocessing.mapping import (
    COLUMN_MAPPING_2022,
    COLUMN_MAPPING_2023,
    COLUMN_MAPPING_2024,
    COLUMNS_TO_DROP_2022,
    COLUMNS_TO_DROP_2023,
    COLUMNS_TO_DROP_2024,
)

def rename_columns(year: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to a consistent format.

    Arguments:
        year: The year for which to rename columns.
        df: The DataFrame with original column names.

    Returns:
        A DataFrame with renamed columns according to COLUMN_MAPPING.
    """
    if year == 2022:
        return df.rename(columns=COLUMN_MAPPING_2022)
    if year == 2023:
        return df.rename(columns=COLUMN_MAPPING_2023)
    return df.rename(columns=COLUMN_MAPPING_2024)


def drop_columns(year: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not needed for analysis.

    Arguments:
        year: The year for which to drop columns.
        df: The DataFrame with columns to drop.

    Returns:
        A DataFrame with specified columns removed.
    """
    if year == 2022:
        columns_to_drop = COLUMNS_TO_DROP_2022
    elif year == 2023:
        columns_to_drop = COLUMNS_TO_DROP_2023
    else:
        columns_to_drop = COLUMNS_TO_DROP_2024

    existing_columns = columns_to_drop & set(df.columns)
    return df.drop(columns=existing_columns)

def normalize_fase(series: pd.Series) -> pd.Series:
    """Convert all Fase formats to int."""
    return (
        series.astype(str)
        .str.replace('ALFA', '0', regex=False)
        .str.replace('FASE ', '', regex=False)
        .str.extract(r'^(\d+)')[0]
        .astype(int)
    )


def standardize_gender(year: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize gender values to 'Feminino' and 'Masculino'.

    Arguments:
        year: The year for which to standardize gender.
        df: The DataFrame with gender column.

    Returns:
        A DataFrame with standardized gender values.
    """
    gender_col = f"gender_{year}"
    if gender_col in df.columns:
        df[gender_col] = df[gender_col].replace({
            "Menina": "Feminino",
            "Menino": "Masculino",
        })
    return df


def standardize_education_institution(year: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize education_institution values.

    Arguments:
        year: The year for which to standardize education institution.
        df: The DataFrame with education_institution column.

    Returns:
        A DataFrame with standardized education_institution values.
    """
    col = f"education_institution_{year}"
    if col in df.columns:
        df[col] = df[col].replace({
            "Escola Pública": "Pública",
            "Privada - Programa de apadrinhamento": "Privada - Programa de Apadrinhamento",
        })
    return df
