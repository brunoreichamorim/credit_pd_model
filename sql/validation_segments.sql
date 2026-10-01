-- Observed default rate vs mean predicted PD by loan_type and loan_purpose, on the
-- hold-out sample. Segments with fewer than $min_size loans are left out (D-021).
-- Missing levels are not shown (UNPIVOT drops NULLs).
-- Parameters: $sample (config.SAMPLE_HOLDOUT), $min_size (config.SQL_MIN_SEGMENT_SIZE).
WITH long_values AS (
    UNPIVOT (
        SELECT s.pd, l.Status, l.loan_type, l.loan_purpose
        FROM pd_scores AS s
        JOIN loans_clean AS l USING (ID)
        WHERE s.sample = $sample
    )
    ON loan_type, loan_purpose
    INTO NAME variable VALUE level
)
SELECT
    variable,
    coalesce(level, '<missing>')  AS level,
    count(*)                      AS n_loans,
    avg(pd)                       AS mean_pd,
    avg(Status)                   AS observed_rate,
    avg(Status) - avg(pd)         AS gap
FROM long_values
GROUP BY variable, level
HAVING count(*) >= $min_size
ORDER BY variable, level;
