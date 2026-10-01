-- Predicted vs observed default rate by equal-count PD bins (NTILE of the frozen model's
-- PD), on the in-scope hold-out sample (pd_scores holds in-scope loans only, D-026).
-- Feeds the decile-gap criterion and the reliability plot (D-025).
-- ORDER BY pd, ID breaks ties deterministically.
-- Parameters: $sample (config.SAMPLE_HOLDOUT), $n_bins (config.CALIBRATION_N_BINS).
WITH scored AS (
    SELECT s.ID, s.pd, l.Status
    FROM pd_scores AS s
    JOIN loans_clean AS l USING (ID)
    WHERE s.sample = $sample
),
binned AS (
    SELECT *, NTILE($n_bins) OVER (ORDER BY pd, ID) AS bin
    FROM scored
)
SELECT
    bin,
    count(*)                  AS n_loans,
    sum(Status)               AS n_defaults,
    min(pd)                   AS pd_min,
    max(pd)                   AS pd_max,
    avg(pd)                   AS mean_pd,
    avg(Status)               AS observed_rate,
    avg(Status) - avg(pd)     AS gap
FROM binned
GROUP BY bin
ORDER BY bin;
