import pandas as pd

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

# Column lists for type standardization (generic names for generalization)
NUMERIC_COLUMNS = [
    'age', 'inde',
    'iaa', 'ieg', 'ips', 'ida',
    'math', 'portuguese',
    'ipv', 'ian',
    'lag_current', 'lag_next',
]

STONE_COLUMNS = ['stone']

ENCODED_CATEGORICAL_COLUMNS = [
    'gender',
    'education_institution',
    'stone',
]


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
