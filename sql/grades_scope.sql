-- How many loans of each sample received an illustrative risk grade (D-027). Counts only:
-- no outcome is read, so the hold-out row uses nothing but the PD lookup (D-025).
-- EQUI loans are out of the model's scope (D-026) and are not graded; every other loan
-- should be graded, so n_in_scope_not_graded must be 0.
-- Parameter: $equi (config.EQUI_LEVEL).
SELECT
    sp.sample,
    count(*)                                                           AS n_loans,
    count(g.ID)                                                        AS n_graded,
    count(*) FILTER (WHERE l.credit_type = $equi)                      AS n_out_of_scope_equi,
    count(*) FILTER (WHERE g.ID IS NULL
                       AND l.credit_type IS DISTINCT FROM $equi)       AS n_in_scope_not_graded
FROM sample_split AS sp
JOIN loans_clean AS l USING (ID)
LEFT JOIN pd_grades AS g USING (ID)
GROUP BY sp.sample
ORDER BY sp.sample;
