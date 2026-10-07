-- Portfolio composition by segment: number of loans, exposure (sum of loan_amount),
-- and two default rates:
--   * count-based:     share of loans that defaulted;
--   * amount-weighted: share of the lent amount that sits in defaulted loans.
-- If the two differ, large and small loans in the segment default at different rates.
-- The segment columns match config.PORTFOLIO_SEGMENT_COLUMNS. Missing values form
-- their own level, '<missing>'.
--
-- Segments with fewer than $min_segment_size loans are not shown (D-021), because such
-- a row describes (almost) a single borrower. The shares are computed over ALL loans
-- before that filter, so the rows shown are unchanged by it; the hidden rows are why
-- the shown shares of a column can add up to slightly less than 1.
WITH segments AS (
    SELECT
        Status,
        loan_amount,
        coalesce(loan_type, '<missing>')              AS loan_type,
        coalesce(loan_purpose, '<missing>')           AS loan_purpose,
        coalesce(Region, '<missing>')                 AS Region,
        coalesce(occupancy_type, '<missing>')         AS occupancy_type,
        coalesce(CAST(term AS VARCHAR), '<missing>')  AS term
    FROM loans_clean
),
long_segments AS (
    UNPIVOT segments
    ON loan_type, loan_purpose, Region, occupancy_type, term
    INTO NAME segment_column VALUE segment_level
),
all_segments AS (
SELECT
    segment_column,
    segment_level,
    count(*)                                            AS n_loans,
    count(*) / sum(count(*)) OVER (PARTITION BY segment_column)                    AS share_loans,
    sum(loan_amount)                                    AS exposure,
    sum(loan_amount) / sum(sum(loan_amount)) OVER (PARTITION BY segment_column)    AS share_exposure,
    avg(Status)                                         AS default_rate,
    sum(loan_amount * Status) / sum(loan_amount)        AS default_rate_amount_weighted
FROM long_segments
GROUP BY segment_column, segment_level
)
SELECT *
FROM all_segments
WHERE n_loans >= $min_segment_size
ORDER BY segment_column, n_loans DESC;
