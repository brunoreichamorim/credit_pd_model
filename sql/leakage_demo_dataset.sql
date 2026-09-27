-- The leakage-demonstration dataset (option C, decision D-024). Unlike model_dataset,
-- this view DELIBERATELY includes the fields excluded under D-017 (in their clean
-- versions where one exists), plus term and co-applicant_credit_type (D-022).
-- It is used only to show how much of a model's apparent skill comes from
-- target leakage. It is NEVER used on the hold-out sample, for risk grades,
-- monitoring or the dashboard -- see src/model.py and docs/decision_log.md D-024.
CREATE OR REPLACE VIEW leakage_demo_dataset AS
SELECT
    l.ID,
    s.sample,
    l.Status,
    -- the 6 admissible main-model features (config.MAIN_MODEL_FEATURES)
    l.income_clean,
    l.loan_amount,
    l.lump_sum_payment,
    l.Neg_ammortization,
    l.loan_type,
    l.loan_purpose,
    -- D-017-excluded fields, added back only for this demonstration.
    -- Numeric (config.LEAKAGE_DEMO_NUMERIC_FEATURES):
    l.rate_of_interest_clean,
    l.Interest_rate_spread,
    l.Upfront_charges,
    l.property_value_clean,
    l.LTV_clean,
    l.dtir1,
    -- Categorical (config.LEAKAGE_DEMO_CATEGORICAL_FEATURES):
    l.credit_type,
    l.age,
    l.submission_of_application,
    l.term,
    l."co-applicant_credit_type"
FROM loans_clean AS l
JOIN sample_split AS s USING (ID);
