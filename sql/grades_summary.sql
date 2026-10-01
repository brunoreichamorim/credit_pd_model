-- Loans, defaults, mean PD and observed default rate per illustrative risk grade, for one
-- sample. Stage 7 runs it on the development sample only: the hold-out's outcomes are
-- not used for the grades (D-025, D-027).
-- Shares are computed over all loans of the sample BEFORE the small-cell filter (D-021).
-- Parameters: $sample (config.SAMPLE_DEVELOPMENT), $min_size (config.SQL_MIN_SEGMENT_SIZE).
WITH graded AS (
    SELECT g.grade, g.grade_rank, g.grade_pd, g.pd, l.Status
    FROM pd_grades AS g
    JOIN loans_clean AS l USING (ID)
    WHERE g.sample = $sample
),
per_grade AS (
    SELECT
        grade,
        grade_rank,
        any_value(grade_pd)       AS grade_pd,
        count(*)                  AS n_loans,
        sum(Status)               AS n_defaults,
        avg(pd)                   AS mean_pd,
        avg(Status)               AS observed_rate
    FROM graded
    GROUP BY grade, grade_rank
)
SELECT
    grade,
    grade_rank,
    grade_pd,
    n_loans,
    n_defaults,
    n_loans / sum(n_loans) OVER ()   AS share,
    mean_pd,
    observed_rate,
    observed_rate - mean_pd          AS gap
FROM per_grade
QUALIFY n_loans >= $min_size
ORDER BY grade_rank;
