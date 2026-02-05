SELECT "Instituição de ensino" AS name, COUNT(*) AS students
FROM raw_data_2022
GROUP BY 1;
