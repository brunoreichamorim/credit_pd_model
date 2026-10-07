-- Headline facts of the loaded table, recomputed in SQL.
-- tests/test_db.py reconciles them with the pandas pipeline (artifacts/dq_summary.csv).
SELECT
    count(*)                  AS n_rows,
    count(DISTINCT ID)        AS n_distinct_ids,
    sum(Status)::BIGINT       AS n_defaults,
    avg(Status)               AS default_rate
FROM loans_clean;
