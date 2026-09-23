# Decision Log

This log records every important methodological or data decision in the project, **why** it was made, and **what kind of statement it is**. The aim is that anybody reviewing the project (a validator, an interviewer, or me in three months) can separate what is known from what was assumed or chosen.

## Decision types

| Type | Meaning | Example |
|---|---|---|
| **FACT** | Directly verifiable from the data or the source, and reproduced by code in this repository. | "ID is unique across all rows." |
| **ASSUMPTION** | Believed to be true but cannot be verified with the available data. It could be wrong. | "Upfront_charges = 0 means 'no fee', not 'missing'." |
| **HEURISTIC** | A practical rule of thumb or threshold chosen from evidence, but not derived from theory or an official standard. | "property_value < 10,000 is invalid." |
| **MODELLING CHOICE** | A deliberate design decision among several defensible options. | "Use logistic regression as the primary model." |

## Status values

- **Agreed**: decided; implemented, or to be implemented in the stage shown.
- **To verify**: decided in principle, but the supporting evidence must first be reproduced by this project's own code (the stage shown).
- **Open**: not yet decided; the options are listed.

Findings from the earlier, pre-project investigation are treated as **To verify** until our own pipeline reproduces them. Nothing is marked FACT until code in this repository has shown it.

---

## Decisions

### D-001 Dataset source
- **Type:** FACT
- **Status:** Agreed (Stage 1)
- **Decision:** The data is the *Loan Default Dataset* by Yasser H. on Kaggle (<https://www.kaggle.com/datasets/yasserh/loan-default-dataset>), file `Loan_Default.csv`, about 148,670 rows × 34 columns.
- **Why it matters:** A PD model is only as credible as its documented data lineage. A similar Kaggle dataset (`nikhil1e9/loan-default`) has a different schema and must not be confused with this one.

### D-002 The raw data is not committed to Git
- **Type:** ASSUMPTION
- **Status:** Agreed (Stage 1)
- **Decision:** `data/raw/`, `data/processed/`, `*.csv`, `*.parquet` and `*.duckdb` are gitignored. Users download the file themselves and place it at `data/raw/Loan_Default.csv`.
- **Why:** We assume the licence does not clearly permit redistribution. Committing only code keeps the public repository safe either way. If the licence is later confirmed to allow redistribution, this can be revisited.

### D-003 `ID` is excluded from modelling
- **Type:** MODELLING CHOICE (uniqueness to be confirmed as FACT)
- **Status:** To verify (Stage 2)
- **Decision:** `ID` is kept as a row key for joins and scoring output but never used as a feature.
- **Why:** An identifier carries no economic information about the borrower's risk. If it were used as a feature, any correlation with the target (for example from the order in which records were assembled) would be spurious and would not generalise.

### D-004 `year` and out-of-time validation
- **Type:** FACT, once confirmed
- **Status:** To verify (Stage 2/3)
- **Decision:** If `year` is constant (expected: 2019 for every row), it is dropped, and the project states explicitly: *"The available dataset does not provide sufficient temporal variation for meaningful out-of-time validation."* No time-based split is invented.
- **Why:** Out-of-time validation tests whether a model survives changes in the economic environment and the population. A random hold-out cannot test that, so claiming otherwise would overstate the validation.

### D-005 `property_value < 10,000` is treated as invalid (set to missing, rows kept)
- **Type:** HEURISTIC
- **Status:** To verify (Stage 2/3)
- **Decision:** Values below 10,000 are set to missing. The rows are **not** deleted.
- **Why:** The earlier investigation found property values such as 8,000 attached to loans of several hundred thousand, which produce LTVs around 7,800%. These look like data-entry or unit errors rather than real collateral values. Values such as 28,000 to 48,000 can be legitimate depending on the loan amount, so the threshold is set low on purpose.
- **Rejected alternative:** "LTV > 200% → remove". That rule is arbitrary, deletes information about the borrower, and would remove genuinely high-LTV (high-risk) loans, which biases the default rate downwards.
- **Revisit if:** Stage 3 shows a clear gap in the property_value distribution at a different point.

### D-006 LTV is recomputed as `LTV_clean`
- **Type:** MODELLING CHOICE
- **Status:** Agreed (Stage 2)
- **Decision:** `LTV_clean = 100 × loan_amount / property_value_clean`. The raw `LTV` column is not used as a feature.
- **Why:** The raw LTV inherits the invalid property values. Recomputing it from its components makes the variable reproducible and consistent with the cleaning rule. Where `property_value_clean` is missing, `LTV_clean` is missing as well rather than extreme.

### D-007 No automatic capping or removal of high but plausible LTV values
- **Type:** MODELLING CHOICE
- **Status:** Agreed (Stage 2)
- **Why:** High LTV is a genuine risk driver. After D-005, the remaining high values are treated as information, not noise.

### D-008 `income = 0` is kept as recorded
- **Type:** ASSUMPTION
- **Status:** Agreed; to be reassessed (Stage 3)
- **Decision:** About 1,260 rows with `income = 0` are kept unchanged.
- **Why:** Zero could mean genuine zero income, missing data coded as 0, or another convention. With no data dictionary, we cannot tell which. Keeping the value is the least invasive choice. This is documented as a limitation.
- **Revisit if:** the default rate or other characteristics of the zero-income rows differ strongly from their neighbours.

### D-009 `rate_of_interest <= 0` is treated as invalid (set to missing)
- **Type:** ASSUMPTION
- **Status:** To verify (Stage 2)
- **Why:** A contractual interest rate of zero or below is implausible for a commercial mortgage. About one row is affected.

### D-010 `Upfront_charges = 0` is kept as a legitimate "no fee" value
- **Type:** ASSUMPTION
- **Status:** Agreed; to be reassessed (Stage 3)
- **Why:** About 20,770 zeros is too frequent to be a random error, and "no upfront fee" is a normal product feature.

### D-011 Mandatory investigation of missingness in the pricing variables
- **Type:** Investigation (the result will be FACT; the treatment will be a MODELLING CHOICE)
- **Status:** Open (Stage 3)
- **Scope:** `rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`.
- **Plan:**
  1. Measure the missing rate of each variable and whether the three are missing on the same rows.
  2. Compare the default rate for missing vs. non-missing rows.
  3. Only then assess whether these variables would realistically be available at the intended prediction point (the application / origination decision).
- **No conclusion is assumed in advance.** Possible outcomes range from "harmless missingness" to "the missingness carries information about the outcome that would not be available at application time (target leakage)". The treatment (a missing indicator, imputation, or excluding the variable) is decided only after the evidence is in.

### D-012 Logistic regression is the primary model
- **Type:** MODELLING CHOICE
- **Status:** Agreed (Stage 5)
- **Why:** It outputs a probability directly. Each coefficient is readable as a change in log-odds, so its direction can be checked against credit intuition. It is stable, easy to validate and well established for PD modelling. A challenger model is optional and only comes after the baseline is complete.

### D-013 Stratified random 70/30 hold-out with a fixed seed
- **Type:** MODELLING CHOICE
- **Status:** Agreed (Stage 5)
- **Decision:** A 70/30 split, stratified on `Status`, with `RANDOM_SEED = 42`. All learned preprocessing (imputation, scaling, encoding, binning) is fitted on the training set only, inside a scikit-learn Pipeline.
- **Why:** Stratification keeps the default rate equal in both samples. Fitting only on the training set prevents information from the test set leaking into the model. This is **out-of-sample**, not out-of-time, validation (see D-004).

### D-014 Risk grades are illustrative internal grades for this project
- **Type:** MODELLING CHOICE
- **Status:** Open; boundaries decided in Stage 7
- **Decision:** Predicted PD is mapped to 10 grades (A to J). They are described as **illustrative internal risk grades for this project**. They are not an official banking methodology, not a regulatory master scale, and not an underwriting decision.
- **Checks planned:** default rates rise from grade to grade, each grade has enough observations, and the construction of the boundaries is documented transparently.

### D-015 The definition of `Status` (default) is undocumented
- **Type:** ASSUMPTION
- **Status:** Agreed; documented as a limitation
- **Decision:** `Status = 1` is treated as "default" as labelled. The dataset gives no default definition (for example days past due), observation window or performance horizon.
- **Consequence:** The predicted PDs are meaningful **relative to this dataset only**. They are not comparable to a regulatory 12-month PD. The portfolio default rate of about 24.6% is far above typical mortgage-portfolio levels, which reinforces this caveat.

### D-016 Use of `Gender` (and possibly `age`) as a model feature
- **Type:** MODELLING CHOICE
- **Status:** Open (Stage 5)
- **Issue:** Using protected characteristics such as sex in credit decisions is legally and ethically sensitive in the EU. Even when a variable is predictive, a bank would need a strong justification and a legal review.
- **Options:**
  - (a) Exclude `Gender` from the model and keep it only for descriptive analysis.
  - (b) Include it and document the concern.
- **Leaning:** (a). The decision will be made explicitly in Stage 5.

---

## Future enhancements (explicitly out of scope for the one-week MVP)

- Weight of Evidence (WoE) binning and Information Value (IV) analysis; a WoE-based scorecard version of the model.
- A challenger model (for example gradient boosting) compared against the logistic baseline.
- Simulated stress / drift scenarios for monitoring. The MVP covers basic PSI / stability analysis only.
- Out-of-time validation, if a dataset with real temporal variation becomes available.
