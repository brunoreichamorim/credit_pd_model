-- Loans in and out of the main model's scope (D-026), by sample: the out-of-scope
-- credit_type = EQUI loans are not scored, and are reported here instead.
-- Parameters: $equi (config.EQUI_LEVEL).
SELECT
    s.sample,
    CASE WHEN l.credit_type IS DISTINCT FROM $equi THEN 'in_scope'
         ELSE 'out_of_scope_EQUI' END   AS scope,
    count(*)                            AS n_loans,
    sum(l.Status)                       AS n_defaults,
    avg(l.Status)                       AS observed_rate
FROM loans_clean AS l
JOIN sample_split AS s USING (ID)
GROUP BY ALL
ORDER BY s.sample, scope;
