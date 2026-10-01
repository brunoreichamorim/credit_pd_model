-- The main model's population (decision D-026): model_dataset without the loans whose
-- credit_type is EQUI. EQUI is 99.99% default (D-017) and is treated as a dataset-
-- construction artefact, not a borrower population, so it is out of the model's scope.
-- credit_type is used here only to decide scope; it is still never a model feature, and
-- the EQUI rows stay in loans_clean (nothing is deleted).
-- 'EQUI' must equal config.EQUI_LEVEL; tests/test_db.py checks the view against it.
CREATE OR REPLACE VIEW model_scope_dataset AS
SELECT m.*
FROM model_dataset AS m
JOIN loans_clean AS l USING (ID)
WHERE l.credit_type IS DISTINCT FROM 'EQUI';
