-- Stage 8 monitoring (decision D-028): Population Stability Index contribution of every
-- bin, comparing a monitored sample with the baseline sample. The PSI of a variable is
-- the sum of its bins' contributions:
--     (monitored share - baseline share) * ln(monitored share / baseline share),
-- with each share floored at $eps (config.PSI_EPSILON) so an empty bin stays finite.
-- Counts only, from the view monitoring_bin_counts: no outcome is read (D-025).
-- Parameters: $baseline (config.SAMPLE_DEVELOPMENT), $monitored (config.SAMPLE_HOLDOUT
-- in this project; a new period's sample in real monitoring), $eps.
WITH paired AS (
    SELECT
        b.variable,
        b.bin_order,
        b.bin_label,
        b.n_loans                     AS n_baseline,
        m.n_loans                     AS n_monitored,
        b.share                       AS share_baseline,
        m.share                       AS share_monitored,
        greatest(b.share, $eps)       AS floored_baseline,
        greatest(m.share, $eps)       AS floored_monitored
    FROM monitoring_bin_counts AS b
    JOIN monitoring_bin_counts AS m
        ON m.variable = b.variable AND m.bin_label = b.bin_label
    WHERE b.sample = $baseline
      AND m.sample = $monitored
)
SELECT
    variable,
    bin_order,
    bin_label,
    n_baseline,
    n_monitored,
    share_baseline,
    share_monitored,
    (floored_monitored - floored_baseline) * ln(floored_monitored / floored_baseline) AS psi_contribution
FROM paired
ORDER BY variable, bin_order;
