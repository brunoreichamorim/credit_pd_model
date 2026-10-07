-- Missing count and default rate with / without a value, for every column that has
-- missing values (decision log D-011, D-017). This is the SQL counterpart of
-- data_processing.missingness_table and is reconciled with it in the tests.
--
-- Step 1 turns every column into a TRUE / FALSE "is missing" flag (names are kept).
-- Step 2 UNPIVOTs the flags into long format: one row per (loan, column).
-- Step 3 aggregates per column.
WITH missing_flags AS (
    SELECT Status, COLUMNS(* EXCLUDE (Status)) IS NULL
    FROM loans_clean
),
long_flags AS (
    UNPIVOT missing_flags
    ON COLUMNS(* EXCLUDE (Status))
    INTO NAME column_name VALUE is_missing
)
SELECT
    column_name,
    count(*) FILTER (WHERE is_missing)            AS n_missing,
    avg(is_missing::INTEGER)                      AS share_missing,
    avg(Status) FILTER (WHERE is_missing)         AS default_rate_if_missing,
    avg(Status) FILTER (WHERE NOT is_missing)     AS default_rate_if_present
FROM long_flags
GROUP BY column_name
HAVING count(*) FILTER (WHERE is_missing) > 0
ORDER BY n_missing DESC, column_name;
