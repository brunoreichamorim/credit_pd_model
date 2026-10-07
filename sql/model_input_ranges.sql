-- Input ranges of the main model's numeric features on the loans it was fitted on:
-- in-scope development loans (model_scope_dataset, D-026). The dashboard warns when a
-- scored input lies outside [lower, upper] (decision D-029; quantile levels are the
-- HEURISTIC config.INPUT_RANGE_QUANTILES). Missing values are left out.
-- Only quantiles are written, never a minimum or maximum, which would be one borrower's value.
WITH dev AS (
    SELECT loan_amount, income_clean
    FROM model_scope_dataset
    WHERE sample = 'development'
),
long AS (
    SELECT 'loan_amount' AS variable, loan_amount AS value FROM dev
    UNION ALL
    SELECT 'income_clean' AS variable, income_clean AS value FROM dev
)
SELECT
    variable,
    count(value)                       AS n_loans,
    quantile_cont(value, $q_lower)     AS lower,
    median(value)                      AS median,
    quantile_cont(value, $q_upper)     AS upper
FROM long
WHERE value IS NOT NULL
GROUP BY variable
ORDER BY variable;
