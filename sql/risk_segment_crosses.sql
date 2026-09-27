-- Default rate for two-way combinations of admissible features (config.SEGMENT_CROSSES),
-- for all loans and for loans outside credit_type = EQUI (D-017).
-- A cross shows whether one feature's effect differs between levels of another.
-- Combinations with fewer than $min_segment_size loans are not shown (D-021).
-- Parameters: $equi (config.EQUI_LEVEL), $min_segment_size (config.SQL_MIN_SEGMENT_SIZE).
WITH base AS (
    SELECT
        Status,
        credit_type,
        coalesce(loan_type, '<missing>')        AS loan_type,
        coalesce(loan_purpose, '<missing>')     AS loan_purpose,
        coalesce(occupancy_type, '<missing>')   AS occupancy_type,
        coalesce(Region, '<missing>')           AS Region
    FROM loans_clean
),
populations AS (
    SELECT 'all' AS population, * FROM base
    UNION ALL
    SELECT 'outside_EQUI' AS population, * FROM base
    WHERE credit_type IS DISTINCT FROM $equi
),
all_crosses AS (
SELECT population, 'loan_type' AS feature_1, loan_type AS level_1,
       'loan_purpose' AS feature_2, loan_purpose AS level_2,
       count(*) AS n_loans, avg(Status) AS default_rate
FROM populations GROUP BY ALL
UNION ALL
SELECT population, 'loan_type', loan_type, 'occupancy_type', occupancy_type,
       count(*), avg(Status)
FROM populations GROUP BY ALL
UNION ALL
SELECT population, 'Region', Region, 'loan_type', loan_type,
       count(*), avg(Status)
FROM populations GROUP BY ALL
)
SELECT *
FROM all_crosses
WHERE n_loans >= $min_segment_size
ORDER BY population, feature_1, feature_2, level_1, level_2;
