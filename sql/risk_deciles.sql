-- Default rate by equal-count bins (NTILE) of loan_amount and income_clean, for two
-- populations: all loans, and loans outside credit_type = EQUI. EQUI is 99.99% default
-- (D-017), so the second population shows the ranking that survives the leakage pattern.
--
-- Only non-missing values are binned (UNPIVOT drops NULLs). NTILE makes bins of equal
-- size, so when many loans share one value (loan amounts are rounded), that value can
-- appear in two neighbouring bins. bin_min / bin_max show the range of each bin.
-- Parameters: $n_bins (config.SQL_N_BINS), $equi (config.EQUI_LEVEL).
WITH long_values AS (
    UNPIVOT (SELECT Status, credit_type, loan_amount, income_clean FROM loans_clean)
    ON loan_amount, income_clean
    INTO NAME variable VALUE value
),
populations AS (
    SELECT 'all' AS population, * FROM long_values
    UNION ALL
    SELECT 'outside_EQUI' AS population, * FROM long_values
    WHERE credit_type IS DISTINCT FROM $equi
),
binned AS (
    SELECT
        *,
        NTILE($n_bins) OVER (PARTITION BY population, variable ORDER BY value) AS bin
    FROM populations
)
SELECT
    population,
    variable,
    bin,
    min(value)    AS bin_min,
    max(value)    AS bin_max,
    count(*)      AS n_loans,
    avg(Status)   AS default_rate
FROM binned
GROUP BY population, variable, bin
ORDER BY population, variable, bin;
