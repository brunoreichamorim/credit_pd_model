-- The model-ready dataset for Stage 5 onwards: the row key, the sample flag, the target
-- and ONLY the 16 admissible candidate features (config.MAIN_MODEL_CANDIDATE_FEATURES,
-- decision D-017). Columns excluded in config.EXCLUDED_FROM_MAIN_MODEL cannot reach the
-- model through this view. No values are changed here: missing values stay missing and
-- are handled inside the scikit-learn pipeline (fitted on development data only).
CREATE OR REPLACE VIEW model_dataset AS
SELECT
    l.ID,
    s.sample,
    l.Status,
    l.loan_amount,
    l.term,
    l.income_clean,
    l.loan_limit,
    l.approv_in_adv,
    l.loan_type,
    l.loan_purpose,
    l.Credit_Worthiness,
    l.open_credit,
    l.Neg_ammortization,
    l.interest_only,
    l.lump_sum_payment,
    l.occupancy_type,
    l.total_units,
    l."co-applicant_credit_type",
    l.Region
FROM loans_clean AS l
JOIN sample_split AS s USING (ID);
