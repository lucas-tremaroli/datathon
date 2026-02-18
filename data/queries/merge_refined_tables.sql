CREATE OR REPLACE TABLE refined.students AS

-- 2022 indicators → 2023 lag
SELECT
    d22.ra_2022 AS ra,
    d22.gender_2022 AS gender,
    d22.education_institution_2022 AS education_institution,
    d22.age_22_2022 AS age,
    d22.stone_22_2022 AS stone,
    d22.inde_22_2022 AS inde,
    d22.iaa_2022 AS iaa,
    d22.ieg_2022 AS ieg,
    d22.ips_2022 AS ips,
    d22.ida_2022 AS ida,
    d22.math_2022 AS math,
    d22.portuguese_2022 AS portuguese,
    d22.english_2022 AS english,
    d22.ipv_2022 AS ipv,
    d22.ian_2022 AS ian,
    NULL AS ipp,  -- IPP not available in 2022
    d22.lag_2022 AS lag_current,
    d23.lag_2023 AS lag_next
FROM refined.data_2022 d22
INNER JOIN refined.data_2023 d23 ON d22.ra_2022 = d23.ra_2023

UNION ALL

-- 2023 indicators → 2024 lag
SELECT
    d23.ra_2023 AS ra,
    d23.gender_2023 AS gender,
    d23.education_institution_2023 AS education_institution,
    d23.age_2023 AS age,
    d23.stone_2023 AS stone,
    d23.inde_2023 AS inde,
    d23.iaa_2023 AS iaa,
    d23.ieg_2023 AS ieg,
    d23.ips_2023 AS ips,
    d23.ida_2023 AS ida,
    d23.math_2023 AS math,
    d23.portuguese_2023 AS portuguese,
    d23.english_2023 AS english,
    d23.ipv_2023 AS ipv,
    d23.ian_2023 AS ian,
    d23.ipp_2023 AS ipp,
    d23.lag_2023 AS lag_current,
    d24.lag_2024 AS lag_next
FROM refined.data_2023 d23
INNER JOIN refined.data_2024 d24 ON d23.ra_2023 = d24.ra_2024;
