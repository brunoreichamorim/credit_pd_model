-- Observed default rate vs mean predicted PD by equal-count bins of loan_amount and
-- income_clean, on the hold-out sample. Answers the D-023 trigger: does the frozen model
-- miscalibrate in the top bins of these features? Missing income_clean is not binned.
-- ORDER BY value, ID breaks ties deterministically.
-- Parameters: $sample (config.SAMPLE_HOLDOUT), $n_bins (config.CALIBRATION_N_BINS).
WITH long_values AS (
    UNPIVOT (
        SELECT s.ID, s.pd, l.Status, l.loan_amount, l.income_clean
        FROM pd_scores AS s
        JOIN loans_clean AS l USING (ID)
        WHERE s.sample = $sample
    )
    ON loan_amount, income_clean
    INTO NAME variable VALUE value
),
binned AS (
    SELECT *, NTILE($n_bins) OVER (PARTITION BY variable ORDER BY value, ID) AS bin
    FROM long_values
)
SELECT
    variable,
    bin,
    min(value)                AS bin_min,
    max(value)                AS bin_max,
    count(*)                  AS n_loans,
    avg(pd)                   AS mean_pd,
    avg(Status)               AS observed_rate,
    avg(Status) - avg(pd)     AS gap
FROM binned
GROUP BY variable, bin
ORDER BY variable, bin;
