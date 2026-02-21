from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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


@dataclass
class OutlierStats:
    """Statistics for outlier detection on a single column."""
    column: str
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    outlier_count: int
    outlier_percentage: float
    total_count: int
    null_count: int
    null_percentage: float


@dataclass
class OutlierReport:
    """Complete outlier analysis report for a DataFrame."""
    column_stats: list[OutlierStats]
    total_records: int
    columns_analyzed: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 90,
            "OUTLIER ANALYSIS REPORT (IQR Method)",
            "=" * 90,
            f"Total records: {self.total_records}",
            f"Columns analyzed: {len(self.columns_analyzed)}",
            "",
            f"{'Column':<16} {'Valid':>7} {'Null%':>7} {'Outliers':>9} {'Out%':>7} {'Lower':>10} {'Upper':>10}",
            "-" * 90,
        ]
        for stat in self.column_stats:
            lines.append(
                f"{stat.column:<16} {stat.total_count:>7} {stat.null_percentage:>6.1f}% "
                f"{stat.outlier_count:>9} {stat.outlier_percentage:>6.1f}% "
                f"{stat.lower_bound:>10.2f} {stat.upper_bound:>10.2f}"
            )
        lines.append("=" * 90)
        return "\n".join(lines)


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    multiplier: float = 1.5,
) -> OutlierReport:
    """
    Detect outliers using the IQR (Interquartile Range) method.

    Outliers are values below Q1 - multiplier*IQR or above Q3 + multiplier*IQR.

    Arguments:
        df: The DataFrame to analyze.
        columns: List of columns to analyze. Defaults to NUMERIC_COLUMNS.
        multiplier: IQR multiplier for bounds (default 1.5).

    Returns:
        OutlierReport containing statistics for each column.
    """
    if columns is None:
        columns = NUMERIC_COLUMNS

    available_columns = [col for col in columns if col in df.columns]
    column_stats = []

    for col in available_columns:
        null_count = df[col].isna().sum()
        data = df[col].dropna()
        if len(data) == 0:
            continue

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_count = outlier_mask.sum()

        column_stats.append(OutlierStats(
            column=col,
            q1=q1,
            q3=q3,
            iqr=iqr,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            outlier_count=outlier_count,
            outlier_percentage=(outlier_count / len(data)) * 100,
            total_count=len(data),
            null_count=null_count,
            null_percentage=(null_count / len(df)) * 100,
        ))

    return OutlierReport(
        column_stats=column_stats,
        total_records=len(df),
        columns_analyzed=available_columns,
    )


def treat_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Treat outliers using winsorization (capping to IQR bounds).

    Values below Q1 - multiplier*IQR are set to the lower bound.
    Values above Q3 + multiplier*IQR are set to the upper bound.

    Arguments:
        df: The DataFrame to treat.
        columns: List of columns to treat. Defaults to NUMERIC_COLUMNS.
        multiplier: IQR multiplier for bounds (default 1.5).

    Returns:
        A DataFrame with outliers capped to IQR bounds.
    """
    if columns is None:
        columns = NUMERIC_COLUMNS

    df = df.copy()
    available_columns = [col for col in columns if col in df.columns]

    for col in available_columns:
        data = df[col].dropna()
        if len(data) == 0:
            continue

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        df[col] = np.clip(df[col], lower_bound, upper_bound)

    return df


def render_outlier_boxplots(
    df: pd.DataFrame,
    report: OutlierReport,
    output_dir: str | Path = "reports",
) -> Path:
    """
    Render boxplots for outlier analysis and save to the output directory.

    Creates a grid of boxplots showing the distribution of each numeric column
    with outlier bounds indicated.

    Arguments:
        df: The DataFrame containing the data.
        report: The OutlierReport from detect_outliers_iqr.
        output_dir: Directory to save the boxplot images.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = report.columns_analyzed
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for idx, stat in enumerate(report.column_stats):
        ax = axes[idx]
        col = stat.column
        data = df[col].dropna()

        bp = ax.boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][0].set_alpha(0.7)

        for element in ['whiskers', 'caps']:
            for item in bp[element]:
                item.set_color('#2c3e50')
                item.set_linewidth(1.5)
        bp['medians'][0].set_color('#e74c3c')
        bp['medians'][0].set_linewidth(2)

        ax.axhline(y=stat.lower_bound, color='#e74c3c', linestyle='--',
                   linewidth=1.5, label=f'Lower: {stat.lower_bound:.2f}')
        ax.axhline(y=stat.upper_bound, color='#e74c3c', linestyle='--',
                   linewidth=1.5, label=f'Upper: {stat.upper_bound:.2f}')

        null_info = f', {stat.null_percentage:.0f}% null' if stat.null_percentage > 0 else ''
        ax.set_title(f'{col}\n({stat.outlier_count} outliers, {stat.outlier_percentage:.1f}%{null_info})',
                     fontsize=10, fontweight='bold')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Outlier Analysis - Boxplots (IQR Method)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'outlier_boxplots.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Boxplot report saved to: {output_path}")
    return output_path
