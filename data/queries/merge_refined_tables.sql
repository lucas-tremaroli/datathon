CREATE OR REPLACE TABLE refined.students AS
SELECT *
FROM refined.data_2022 d22
FULL OUTER JOIN refined.data_2023 d23 ON d22.ra_2022 = d23.ra_2023
FULL OUTER JOIN refined.data_2024 d24 ON COALESCE(d22.ra_2022, d23.ra_2023) = d24.ra_2024;
