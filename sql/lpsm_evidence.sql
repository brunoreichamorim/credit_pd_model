-- Evidence behind keeping lump_sum_payment = lpsm in the main model (decision D-017,
-- corrected 2026-10-06). Development sample only; never the hold-out. Run from
-- notebooks/02_model_training.ipynb and tests/test_model.py; not a committed table.
-- "pricing_present" and "all_six_present" use the raw columns named in D-011 / D-017.
WITH lpsm AS (
    SELECT l.*
    FROM loans_clean AS l
    JOIN sample_split AS s USING (ID)
    WHERE s.sample = $sample
      AND l.lump_sum_payment = $level
)
SELECT 'development' AS label, count(*) AS n_loans, sum(Status)::BIGINT AS n_defaults, avg(Status) AS default_rate
FROM lpsm
UNION ALL
SELECT 'outside_equi', count(*), sum(Status)::BIGINT, avg(Status)
FROM lpsm WHERE credit_type <> $equi
UNION ALL
SELECT 'outside_equi_' || loan_type, count(*), sum(Status)::BIGINT, avg(Status)
FROM lpsm WHERE credit_type <> $equi
GROUP BY loan_type
UNION ALL
SELECT 'pricing_present', count(*), sum(Status)::BIGINT, avg(Status)
FROM lpsm
WHERE rate_of_interest IS NOT NULL AND Interest_rate_spread IS NOT NULL AND Upfront_charges IS NOT NULL
UNION ALL
SELECT 'all_six_present', count(*), sum(Status)::BIGINT, avg(Status)
FROM lpsm
WHERE rate_of_interest IS NOT NULL AND Interest_rate_spread IS NOT NULL AND Upfront_charges IS NOT NULL
  AND dtir1 IS NOT NULL AND property_value IS NOT NULL AND income IS NOT NULL
ORDER BY label;
