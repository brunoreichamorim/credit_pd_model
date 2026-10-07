-- Stage 8 monitoring (decision D-028): number of in-scope loans in each frozen
-- development bin, for every sample. The bins are the table monitoring_baseline,
-- built from development loans by src/monitoring.py.
--
-- Step 1 gathers each scored loan's PD ('score'), grade and the six model features.
-- Step 2 UNPIVOTs them into long format: one row per (loan, variable), keeping NULLs.
-- Step 3 puts each value in its bin:
--   * numeric: a range join, lower <= value < upper (the same rule as
--     sql/grade_assignment.sql); a missing value goes to '<missing>';
--   * categorical: the level's own bin; missing -> '<missing>'; a level never seen in
--     development -> '<unseen>' (the model's encoder would reject it, D-023).
-- Step 4 counts loans per bin and sample. Every bin appears for every sample, with 0
-- loans if it is empty, so the PSI always compares the same bins.
--
-- Only inputs and scores are read, never Status: the hold-out's outcomes are not used
-- (D-025, D-028). EQUI loans are not scored (D-026), so they are not counted here.
-- The variable names must match config.MONITORING_*_VARIABLES and the labels must match
-- config.MISSING_BIN_LABEL / UNSEEN_BIN_LABEL; tests/test_monitoring.py checks this.
CREATE OR REPLACE VIEW monitoring_bin_counts AS
WITH inputs AS (
    SELECT
        g.sample,
        CAST(g.pd AS DOUBLE)            AS score,
        -- a NaN would sort above every edge in DuckDB, so it is treated as missing
        CASE WHEN isnan(CAST(m.loan_amount AS DOUBLE)) THEN NULL
             ELSE CAST(m.loan_amount AS DOUBLE) END                   AS loan_amount,
        CASE WHEN isnan(CAST(m.income_clean AS DOUBLE)) THEN NULL
             ELSE CAST(m.income_clean AS DOUBLE) END                  AS income_clean,
        CAST(g.grade AS VARCHAR)        AS grade,
        CAST(m.lump_sum_payment AS VARCHAR)  AS lump_sum_payment,
        CAST(m.Neg_ammortization AS VARCHAR) AS Neg_ammortization,
        CAST(m.loan_type AS VARCHAR)    AS loan_type,
        CAST(m.loan_purpose AS VARCHAR) AS loan_purpose
    FROM pd_grades AS g
    JOIN model_scope_dataset AS m USING (ID)
),
numeric_long AS (
    SELECT sample, variable, num_value
    FROM inputs
    UNPIVOT INCLUDE NULLS (num_value FOR variable IN (score, loan_amount, income_clean))
),
categorical_long AS (
    SELECT sample, variable, cat_value
    FROM inputs
    UNPIVOT INCLUDE NULLS (cat_value FOR variable IN (grade, lump_sum_payment, Neg_ammortization, loan_type, loan_purpose))
),
numeric_binned AS (
    SELECT v.sample, v.variable, coalesce(b.bin_label, '<missing>') AS bin_label
    FROM numeric_long AS v
    LEFT JOIN monitoring_baseline AS b
        ON b.variable = v.variable
       AND b.kind = 'range'
       AND v.num_value >= b.lower
       AND v.num_value < b.upper
),
categorical_binned AS (
    SELECT
        v.sample,
        v.variable,
        CASE
            WHEN v.cat_value IS NULL THEN '<missing>'
            WHEN b.bin_label IS NULL THEN '<unseen>'
            ELSE b.bin_label
        END AS bin_label
    FROM categorical_long AS v
    LEFT JOIN monitoring_baseline AS b
        ON b.variable = v.variable
       AND b.kind = 'level'
       AND b.level = v.cat_value
),
counts AS (
    SELECT sample, variable, bin_label, count(*) AS n_loans
    FROM (SELECT * FROM numeric_binned UNION ALL SELECT * FROM categorical_binned)
    GROUP BY sample, variable, bin_label
),
samples AS (
    SELECT DISTINCT sample FROM inputs
)
SELECT
    s.sample,
    b.variable,
    b.bin_order,
    b.bin_label,
    coalesce(c.n_loans, 0)                                                    AS n_loans,
    coalesce(c.n_loans, 0) / sum(coalesce(c.n_loans, 0)) OVER (PARTITION BY s.sample, b.variable) AS share
FROM monitoring_baseline AS b
CROSS JOIN samples AS s
LEFT JOIN counts AS c
    ON c.sample = s.sample AND c.variable = b.variable AND c.bin_label = b.bin_label;
