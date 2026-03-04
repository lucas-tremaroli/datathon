"""Re-export shim — all symbols moved to focused submodules."""

from datathon.preprocessing.cleaning import (
    drop_columns,
    rename_columns,
    standardize_education_institution,
    standardize_gender,
)
from datathon.preprocessing.encoding import (
    EDUCATION_INSTITUTION_ENCODING,
    ENCODED_CATEGORICAL_COLUMNS,
    GENDER_ENCODING,
    NUMERIC_COLUMNS,
    STONE_COLUMNS,
    STONE_ENCODING,
    standardize_dtypes,
)
from datathon.preprocessing.imputation import (
    impute_nulls,
    round_numeric_columns,
)
from datathon.preprocessing.outliers import (
    OutlierReport,
    OutlierStats,
    detect_outliers_iqr,
    render_outlier_boxplots,
    treat_outliers_iqr,
)

__all__ = [
    "EDUCATION_INSTITUTION_ENCODING",
    "ENCODED_CATEGORICAL_COLUMNS",
    "GENDER_ENCODING",
    "NUMERIC_COLUMNS",
    "STONE_COLUMNS",
    "STONE_ENCODING",
    "OutlierReport",
    "OutlierStats",
    "detect_outliers_iqr",
    "drop_columns",
    "impute_nulls",
    "rename_columns",
    "render_outlier_boxplots",
    "round_numeric_columns",
    "standardize_dtypes",
    "standardize_education_institution",
    "standardize_gender",
    "treat_outliers_iqr",
]
