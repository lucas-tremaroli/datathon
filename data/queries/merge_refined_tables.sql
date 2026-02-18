CREATE OR REPLACE TABLE refined.students AS
SELECT
    -- Unified identifier
    COALESCE(d24.ra_2024, d23.ra_2023, d22.ra_2022) AS ra,

    -- Unified static columns (picking most recent non-null value)
    COALESCE(d24.gender_2024, d23.gender_2023, d22.gender_2022) AS gender,
    COALESCE(d24.education_institution_2024, d23.education_institution_2023, d22.education_institution_2022) AS education_institution,

    -- 2022 features
    d22.age_22_2022,
    d22.stone_22_2022,
    d22.inde_22_2022,
    d22.iaa_2022,
    d22.ieg_2022,
    d22.ips_2022,
    d22.ida_2022,
    d22.math_2022,
    d22.portuguese_2022,
    d22.english_2022,
    d22.ipv_2022,
    d22.ian_2022,
    d22.lag_2022,

    -- 2023 features
    d23.age_2023,
    d23.stone_2023,
    d23.inde_2023,
    d23.ipp_2023,
    d23.iaa_2023,
    d23.ieg_2023,
    d23.ips_2023,
    d23.ida_2023,
    d23.math_2023,
    d23.portuguese_2023,
    d23.english_2023,
    d23.ipv_2023,
    d23.ian_2023,
    d23.lag_2023,

    -- 2024 features
    d24.age_2024,
    d24.stone_2024,
    d24.inde_2024,
    d24.ipp_2024,
    d24.iaa_2024,
    d24.ieg_2024,
    d24.ips_2024,
    d24.ida_2024,
    d24.math_2024,
    d24.portuguese_2024,
    d24.english_2024,
    d24.ipv_2024,
    d24.ian_2024,
    d24.lag_2024

FROM refined.data_2022 d22
FULL OUTER JOIN refined.data_2023 d23 ON d22.ra_2022 = d23.ra_2023
FULL OUTER JOIN refined.data_2024 d24 ON COALESCE(d22.ra_2022, d23.ra_2023) = d24.ra_2024;
