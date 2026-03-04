from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datathon.preprocessing.encoding import NUMERIC_COLUMNS


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
