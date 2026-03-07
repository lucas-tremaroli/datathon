import numpy as np
import pandas as pd

from datathon.preprocessing.cleaning import (
    drop_columns,
    rename_columns,
    standardize_education_institution,
    standardize_gender,
)
from datathon.preprocessing.encoding import (
    EDUCATION_INSTITUTION_ENCODING,
    GENDER_ENCODING,
    NUMERIC_COLUMNS,
    STONE_ENCODING,
    standardize_dtypes,
)
from datathon.preprocessing.imputation import impute_nulls, round_numeric_columns
from datathon.preprocessing.outliers import (
    OutlierReport,
    detect_outliers_iqr,
    treat_outliers_iqr,
)


# ── cleaning ──────────────────────────────────────────────────────────────


class TestRenameColumns:
    def test_rename_2022(self):
        df = pd.DataFrame({"RA": [1], "Gênero": ["M"], "IAA": [5.0]})
        result = rename_columns(2022, df)
        assert "ra_2022" in result.columns
        assert "gender_2022" in result.columns
        assert "iaa_2022" in result.columns

    def test_rename_2023(self):
        df = pd.DataFrame({"RA": [1], "Gênero": ["F"], "IAA": [6.0]})
        result = rename_columns(2023, df)
        assert "ra_2023" in result.columns
        assert "gender_2023" in result.columns

    def test_rename_2024(self):
        df = pd.DataFrame({"RA": [1], "Gênero": ["F"]})
        result = rename_columns(2024, df)
        assert "ra_2024" in result.columns

    def test_rename_unknown_year_uses_2024_mapping(self):
        df = pd.DataFrame({"RA": [1]})
        result = rename_columns(2025, df)
        assert "ra_2024" in result.columns


class TestDropColumns:
    def test_drop_2022(self):
        df = pd.DataFrame({"name_2022": ["x"], "ra_2022": [1], "iaa_2022": [5.0]})
        result = drop_columns(2022, df)
        assert "name_2022" not in result.columns
        assert "ra_2022" in result.columns

    def test_drop_2023(self):
        df = pd.DataFrame({"anonymized_name_2023": ["x"], "ra_2023": [1]})
        result = drop_columns(2023, df)
        assert "anonymized_name_2023" not in result.columns

    def test_drop_2024(self):
        df = pd.DataFrame({"anonymized_name_2024": ["x"], "ra_2024": [1]})
        result = drop_columns(2024, df)
        assert "anonymized_name_2024" not in result.columns

    def test_drop_nonexistent_columns_ignored(self):
        df = pd.DataFrame({"ra_2022": [1], "iaa_2022": [5.0]})
        result = drop_columns(2022, df)
        assert list(result.columns) == ["ra_2022", "iaa_2022"]


class TestStandardizeGender:
    def test_replaces_menina_menino(self):
        df = pd.DataFrame({"gender_2022": ["Menina", "Menino", "Feminino"]})
        result = standardize_gender(2022, df)
        assert list(result["gender_2022"]) == ["Feminino", "Masculino", "Feminino"]

    def test_no_gender_column_is_noop(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        result = standardize_gender(2022, df)
        pd.testing.assert_frame_equal(result, df)


class TestStandardizeEducationInstitution:
    def test_replaces_variants(self):
        df = pd.DataFrame({
            "education_institution_2023": [
                "Escola Pública",
                "Privada - Programa de apadrinhamento",
                "Privada",
            ]
        })
        result = standardize_education_institution(2023, df)
        expected = ["Pública", "Privada - Programa de Apadrinhamento", "Privada"]
        assert list(result["education_institution_2023"]) == expected

    def test_no_column_is_noop(self):
        df = pd.DataFrame({"other": [1]})
        result = standardize_education_institution(2023, df)
        pd.testing.assert_frame_equal(result, df)


# ── encoding ──────────────────────────────────────────────────────────────


class TestStandardizeDtypes:
    def test_numeric_columns_converted(self):
        df = pd.DataFrame({"age": ["14", "15"], "inde": ["7.5", "bad"]})
        result = standardize_dtypes(df)
        assert np.issubdtype(result["age"].dtype, np.number)
        assert result["age"].iloc[0] == 14
        assert result["inde"].iloc[0] == 7.5
        assert pd.isna(result["inde"].iloc[1])

    def test_stone_encoded(self):
        df = pd.DataFrame({"stone": ["Quartzo", "Ágata", "Ametista", "Topázio"]})
        result = standardize_dtypes(df)
        assert list(result["stone"]) == [1, 2, 3, 4]

    def test_stone_agata_without_accent(self):
        df = pd.DataFrame({"stone": ["Agata"]})
        result = standardize_dtypes(df)
        assert result["stone"].iloc[0] == 2

    def test_gender_encoded(self):
        df = pd.DataFrame({"gender": ["Feminino", "Masculino"]})
        result = standardize_dtypes(df)
        assert list(result["gender"]) == [0, 1]

    def test_education_institution_encoded(self):
        df = pd.DataFrame({"education_institution": ["Pública", "Privada"]})
        result = standardize_dtypes(df)
        assert result["education_institution"].iloc[0] == 0
        assert result["education_institution"].iloc[1] == 1

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"gender": ["Feminino"]})
        standardize_dtypes(df)
        assert df["gender"].iloc[0] == "Feminino"


# ── imputation ────────────────────────────────────────────────────────────


class TestImputeNulls:
    def test_numeric_imputed_with_median(self):
        df = pd.DataFrame({"age": [10.0, 20.0, np.nan], "inde": [5.0, np.nan, 7.0]})
        result = impute_nulls(df)
        assert result["age"].iloc[2] == 15.0  # median of 10, 20
        assert result["inde"].iloc[1] == 6.0  # median of 5, 7

    def test_categorical_imputed_with_mode(self):
        df = pd.DataFrame({"gender": [0, 0, 1, np.nan]})
        result = impute_nulls(df)
        assert result["gender"].iloc[3] == 0  # mode is 0

    def test_no_nulls_unchanged(self):
        df = pd.DataFrame({"age": [10.0, 20.0], "inde": [5.0, 7.0]})
        result = impute_nulls(df)
        pd.testing.assert_frame_equal(result, df)

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"age": [10.0, np.nan]})
        impute_nulls(df)
        assert pd.isna(df["age"].iloc[1])


class TestRoundNumericColumns:
    def test_rounds_to_2_decimals(self):
        df = pd.DataFrame({"age": [14.567, 15.123], "inde": [7.5555, 8.9999]})
        result = round_numeric_columns(df)
        assert result["age"].iloc[0] == 14.57
        assert result["inde"].iloc[1] == 9.0

    def test_custom_decimals(self):
        df = pd.DataFrame({"age": [14.567]})
        result = round_numeric_columns(df, decimals=1)
        assert result["age"].iloc[0] == 14.6

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"age": [14.567]})
        round_numeric_columns(df)
        assert df["age"].iloc[0] == 14.567


# ── outliers ──────────────────────────────────────────────────────────────


class TestDetectOutliersIqr:
    def test_detects_outliers(self, outlier_df):
        report = detect_outliers_iqr(outlier_df, columns=["age"])
        assert isinstance(report, OutlierReport)
        assert len(report.column_stats) == 1
        stat = report.column_stats[0]
        assert stat.outlier_count >= 2
        assert stat.column == "age"

    def test_empty_column_skipped(self):
        df = pd.DataFrame({"age": [np.nan, np.nan, np.nan]})
        report = detect_outliers_iqr(df, columns=["age"])
        assert len(report.column_stats) == 0

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        report = detect_outliers_iqr(df, columns=["age"])
        assert len(report.column_stats) == 0
        assert report.columns_analyzed == []

    def test_custom_multiplier(self, outlier_df):
        report_tight = detect_outliers_iqr(outlier_df, columns=["age"], multiplier=1.0)
        report_loose = detect_outliers_iqr(outlier_df, columns=["age"], multiplier=3.0)
        assert report_tight.column_stats[0].outlier_count >= report_loose.column_stats[0].outlier_count

    def test_report_str(self, outlier_df):
        report = detect_outliers_iqr(outlier_df, columns=["age"])
        text = str(report)
        assert "OUTLIER ANALYSIS REPORT" in text
        assert "age" in text

    def test_null_percentage_tracked(self):
        df = pd.DataFrame({"age": [1.0, 2.0, 3.0, np.nan, np.nan]})
        report = detect_outliers_iqr(df, columns=["age"])
        stat = report.column_stats[0]
        assert stat.null_count == 2
        assert stat.null_percentage == 40.0


class TestTreatOutliersIqr:
    def test_caps_outliers(self, outlier_df):
        result = treat_outliers_iqr(outlier_df, columns=["age"])
        assert result["age"].max() < 200.0
        assert result["age"].min() > -100.0

    def test_does_not_modify_original(self, outlier_df):
        original_max = outlier_df["age"].max()
        treat_outliers_iqr(outlier_df, columns=["age"])
        assert outlier_df["age"].max() == original_max

    def test_empty_column_unchanged(self):
        df = pd.DataFrame({"age": [np.nan, np.nan]})
        result = treat_outliers_iqr(df, columns=["age"])
        assert result["age"].isna().all()

    def test_missing_column_unchanged(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = treat_outliers_iqr(df, columns=["age"])
        pd.testing.assert_frame_equal(result, df)
