import pandas as pd

def normalize_fase(series: pd.Series) -> pd.Series:
    """Convert all Fase formats to int."""
    return (
        series.astype(str)
        .str.replace('ALFA', '0', regex=False)
        .str.replace('FASE ', '', regex=False)
        .str.extract(r'^(\d+)')[0]
        .astype(int)
    )

def load_and_clean(db) -> pd.DataFrame:
    """Load all years and harmonize schema."""
    dfs = []
    for year in [2022, 2023, 2024]:
        df = db.fetch_table(f'raw_data_{year}')
        df['fase'] = normalize_fase(df['Fase'])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
