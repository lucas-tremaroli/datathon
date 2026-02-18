import pandas as pd

from datathon.preprocessing.mapping import (
    COLUMN_MAPPING_2022,
    COLUMN_MAPPING_2023,
    COLUMN_MAPPING_2024,
    COLUMNS_TO_DROP_2022,
    COLUMNS_TO_DROP_2023,
    COLUMNS_TO_DROP_2024,
)

# Encoding mappings for categorical columns
STONE_ENCODING = {
    'Quartzo': 1,
    'Ágata': 2,
    'Agata': 2,  # Handle missing accent
    'Ametista': 3,
    'Topázio': 4,
}

GENDER_ENCODING = {
    'Feminino': 0,
    'Masculino': 1,
}

EDUCATION_INSTITUTION_ENCODING = {
    'Pública': 0,
    'Privada': 1,
    'Privada - Programa de Apadrinhamento': 2,
    'Privada *Parcerias com Bolsa 100%': 3,
    'Privada - Pagamento por *Empresa Parceira': 4,
    'Escola JP II': 5,
    'Rede Decisão': 6,
    'Bolsista Universitário *Formado (a)': 7,
    'Concluiu o 3º EM': 8,
    'Nenhuma das opções acima': 9,
}

# Column lists for type standardization
NUMERIC_COLUMNS = [
    # 2022
    'age_22_2022', 'inde_22_2022',
    'iaa_2022', 'ieg_2022', 'ips_2022', 'ida_2022',
    'math_2022', 'portuguese_2022', 'english_2022',
    'ipv_2022', 'ian_2022', 'lag_2022',
    # 2023
    'age_2023', 'inde_2023', 'ipp_2023',
    'iaa_2023', 'ieg_2023', 'ips_2023', 'ida_2023',
    'math_2023', 'portuguese_2023', 'english_2023',
    'ipv_2023', 'ian_2023', 'lag_2023',
    # 2024
    'age_2024', 'inde_2024', 'ipp_2024',
    'iaa_2024', 'ieg_2024', 'ips_2024', 'ida_2024',
    'math_2024', 'portuguese_2024', 'english_2024',
    'ipv_2024', 'ian_2024', 'lag_2024',
]

STONE_COLUMNS = ['stone_22_2022', 'stone_2023', 'stone_2024']

ENCODED_CATEGORICAL_COLUMNS = [
    'gender',
    'education_institution',
    'stone_22_2022',
    'stone_2023',
    'stone_2024',
]

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


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data types across all columns.

    - Convert numeric columns to float
    - Encode stone columns as ordinal integers (Quartzo=1 < Ágata=2 < Ametista=3 < Topázio=4)
    - Encode gender as binary (Feminino=0, Masculino=1)
    - Encode education_institution as categorical integers

    Arguments:
        df: The DataFrame with inconsistent data types.

    Returns:
        A DataFrame with standardized data types.
    """
    df = df.copy()

    # Convert numeric columns to float
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Encode stone columns as ordinal
    for col in STONE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(STONE_ENCODING)

    # Encode categorical columns
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(GENDER_ENCODING)

    if 'education_institution' in df.columns:
        df['education_institution'] = df['education_institution'].map(EDUCATION_INSTITUTION_ENCODING)

    return df


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
