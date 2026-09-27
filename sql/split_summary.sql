-- Size and default rate of each sample in the D-013 split. Stratification on Status
-- should make the default rate (almost) equal in both samples.
SELECT
    s.sample,
    count(*)                        AS n_loans,
    count(*) / sum(count(*)) OVER () AS share_loans,
    sum(l.Status)::BIGINT           AS n_defaults,
    avg(l.Status)                   AS default_rate
FROM loans_clean AS l
JOIN sample_split AS s USING (ID)
GROUP BY s.sample
ORDER BY s.sample;
