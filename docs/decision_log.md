# Decision Log

This log records every important methodological or data decision in the project, **why** it was made, and **what kind of statement it is**. The aim is that anybody reviewing the project can separate what is known from what was assumed or chosen.

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
- **Type:** FACT (file size and schema, verified by code). The Kaggle source is recorded from where the file was downloaded; code cannot verify where a file came from.
- **Status:** Agreed (verified in Stage 2)
- **Decision:** The data is the *Loan Default Dataset* by Yasser H. on Kaggle (<https://www.kaggle.com/datasets/yasserh/loan-default-dataset>), file `Loan_Default.csv`, 148,670 rows × 34 columns.
- **Why it matters:** A PD model is only as credible as its documented data lineage. A similar Kaggle dataset (`nikhil1e9/loan-default`) has a different schema and must not be confused with this one.
- **Stage 2 evidence:** `src/data_processing.validate_schema` and `tests/test_data_processing.py` confirm 148,670 rows, 34 columns and exactly the column names and order in `config.EXPECTED_RAW_COLUMNS`. The verified file has SHA-256 `4234b122f463ff4d563de600ade5ec347a9ab8f02cfc204535f7c0d4929bfe70`.

### D-002 The raw data is not committed to Git
- **Type:** ASSUMPTION
- **Status:** Agreed (Stage 1)
- **Decision:** `data/raw/`, `data/processed/`, `*.csv`, `*.parquet` and `*.duckdb` are gitignored. Users download the file themselves and place it at `data/raw/Loan_Default.csv`.
- **Why:** We assume the licence does not clearly permit redistribution. Committing only code keeps the public repository safe either way. If the licence is later confirmed to allow redistribution, this can be revisited.

### D-003 `ID` is excluded from modelling
- **Type:** MODELLING CHOICE (uniqueness is FACT)
- **Status:** Agreed (uniqueness verified in Stage 2)
- **Decision:** `ID` is kept as a row key for joins and scoring output but never used as a feature.
- **Why:** An identifier carries no economic information about the borrower's risk. If it were used as a feature, any correlation with the target (for example from the order in which records were assembled) would be spurious and would not generalise.
- **Stage 2 evidence (FACT):** `ID` has no missing values and is unique. It is a contiguous integer range from 24,890 to 173,559, and the file is sorted by it. The default rate is flat across `ID` deciles (24.2% to 25.5%, from exploration; not part of the pipeline). There are no duplicate rows, including when `ID` is ignored.

### D-004 `year` and out-of-time validation
- **Type:** FACT
- **Status:** Agreed (verified and implemented in Stage 2)
- **Decision:** If `year` is constant (expected: 2019 for every row), it is dropped, and the project states explicitly: *"The available dataset does not provide sufficient temporal variation for meaningful out-of-time validation."* No time-based split is invented.
- **Why:** Out-of-time validation tests whether a model survives changes in the economic environment and the population. A random hold-out cannot test that, so claiming otherwise would overstate the validation.
- **Stage 2 evidence:** `year` = 2019 in all 148,670 rows. The dataset has no genuine multi-period time dimension, so true out-of-time validation is not possible later. `apply_cleaning_rules` drops `year` from the processed file and **stops with an error** if a future version of the file is not constant.

### D-005 `property_value < 10,000` is treated as invalid (set to missing, rows kept)
- **Type:** HEURISTIC (an investigation threshold for this dataset, not an industry or regulatory rule)
- **Status:** Agreed; implemented in Stage 2 as `property_value_clean` (raw `property_value` kept unchanged); reassessed in Stage 3 with no change
- **Decision:** Values below 10,000 are set to missing. The rows are **not** deleted.
- **Why:** The earlier investigation found property values such as 8,000 attached to loans of several hundred thousand, which produce LTVs around 7,800%. These look like data-entry or unit errors rather than real collateral values. Values such as 28,000 to 48,000 can be legitimate depending on the loan amount, so the threshold is set low on purpose.
- **Rejected alternative:** "LTV > 200% → remove". That rule is arbitrary, deletes information about the borrower, and would remove genuinely high-LTV (high-risk) loans, which biases the default rate downwards.
- **Revisit if:** Stage 3 shows a clear gap in the property_value distribution at a different point.
- **Stage 2 evidence (FACT):**
  - **Rows affected:** 6. All of them have `property_value = 8,000`, the lowest value in the file. They are actual numeric values, not missing values.
  - **What they look like:** loan amounts from 186,500 to 626,500 and raw LTVs from 2,331% to 7,831%. All 6 also have `income` missing. Five have `Status = 0`, one has `Status = 1`.
  - **Isolated, not systematic:** the next distinct values are 18,000 (1 row, loan 16,500, LTV 91.7%, which is plausible) and 28,000 (9 rows). Above the 8,000 rows, the next-highest raw LTV is 263.5%, which leaves a large gap. The threshold therefore isolates exactly the 8,000 value.
  - **Rounding pattern:** every `property_value` ends in 8,000 (…8,000, 18,000, 28,000, …) and every `loan_amount` ends in 6,500. This suggests the values were rounded or binned before publication (ASSUMPTION about why), so 8,000 is best read as "the lowest bucket", not an exact valuation.
- **Stage 3 reassessment (FACT; `notebooks/01_eda.ipynb`, section 10):** the "revisit" condition is **not** met.
  - The 30 lowest distinct values are spaced exactly 10,000 apart, and the row counts rise smoothly from 18,000 upwards (1, 9, 35, 71, 141 rows, …). There is no break at any other point.
  - Only the 8,000 value produces implausible LTVs (2,331% to 7,831%). The highest raw LTV among all other rows is 263.5%, which is also the maximum of `LTV_clean`.
  - **Result:** the threshold isolates exactly that cluster, so it stays unchanged.
- **Scope after Stage 3:** `property_value` and LTV are excluded from the main model (D-017), so this rule now affects only the leakage demonstration model and the descriptive analysis.

### D-006 LTV is recomputed as `LTV_clean`
- **Type:** MODELLING CHOICE
- **Status:** Agreed (implemented in Stage 2)
- **Decision:** `LTV_clean = 100 × loan_amount / property_value_clean`. The raw `LTV` column is not used as a feature.
- **Why:** The raw LTV inherits the invalid property values. Recomputing it from its components makes the variable reproducible and consistent with the cleaning rule. Where `property_value_clean` is missing, `LTV_clean` is missing as well rather than extreme.
- **Stage 2 evidence (FACT):** The raw `LTV` equals `100 × loan_amount / property_value` on every row (max difference about 5e-8), and `LTV` is missing exactly where `property_value` is missing (15,098 rows). So `LTV_clean` differs from the raw `LTV` only on the 6 rows affected by D-005.

### D-007 No automatic capping or removal of high but plausible LTV values
- **Type:** MODELLING CHOICE
- **Status:** Agreed (Stage 2)
- **Why:** High LTV is a genuine risk driver. After D-005, the remaining high values are treated as information, not noise.
- **Stage 2 evidence (FACT):**
  - **Raw LTV distribution:** 0.97% to 7,831%, with a median of 75.1%, a 99th percentile of 102.8% and a 99.9th percentile of 123.7%.
  - **After D-005:** 1,793 rows have `LTV_clean` above 100% and 11 above 200%. The maximum is 263.5%. The most extreme cases are mostly small loans on low-value properties: the 27 rows above 150% have a median loan of 106,500 and a median property value of 68,000.
  - **Why they are kept:** these values are unusual but potentially legitimate, so none are capped or removed. Very low LTVs (minimum 0.97%, a 106,500 loan on an 11,008,000 property) are also unusual but potentially legitimate, and are kept.

### D-008 `income = 0` is treated as invalid (set to missing in `income_clean`)
- **Type:** MODELLING CHOICE (the Stage 3 evidence is FACT)
- **Status:** Agreed (reassessed in Stage 3; implemented as `income_clean`, raw `income` kept unchanged)
- **Original decision (Stage 2):** 1,260 rows with `income = 0` (FACT, Stage 2) were kept unchanged. Separately, 9,150 rows (6.15%) have `income` missing.
- **Original reasoning:** Zero could mean genuine zero income, missing data coded as 0, or another convention. With no data dictionary, we cannot tell which. Keeping the value was the least invasive choice.
- **Revisit if:** the default rate or other characteristics of the zero-income rows differ strongly from their neighbours.
- **Stage 3 evidence (FACT; `notebooks/01_eda.ipynb`, section 6):** the "revisit" condition is met.
  - The 1,260 zero-income rows have a 99.4% default rate. That is far above the lowest positive-income decile and the portfolio rate of 24.64%.
  - 912 of them are `credit_type = EQUI`. The 348 zero-income rows outside EQUI still default at 97.7%. So the zero is a leakage indicator in its own right (see D-017), not a genuine low-income effect.
  - By contrast, a *missing* income looks like ordinary missingness: those rows default at 13.5%, below the 25.4% for rows with income present.
- **Decision (Stage 3):** `income_clean` sets `income <= 0` to missing (`config.INCOME_MIN_EXCLUSIVE`). The combined missing group (10,410 rows) then has a 23.9% default rate, close to the portfolio rate, so a missing `income_clean` no longer reveals the target. Missing values are imputed inside the model pipeline in Stage 5.
- **Why:** a mortgage with zero recorded income is implausible without a special product, and here the value behaves like the other outcome-driven indicators. Setting it to missing follows the same pattern as D-005 and D-009. Rows are kept, and the raw column is unchanged.

### D-009 `rate_of_interest <= 0` is treated as invalid (set to missing)
- **Type:** ASSUMPTION
- **Status:** Agreed (implemented in Stage 2 as `rate_of_interest_clean`; raw `rate_of_interest` kept unchanged)
- **Why:** A contractual interest rate of zero or below is implausible for a mortgage loan.
- **Stage 2 evidence (FACT):** Exactly one row is affected: `ID` 61162, `rate_of_interest = 0.0`, `Status = 0`. The next-lowest rate is 2.125%. Its `Interest_rate_spread` is -3.638, the minimum in the file. That is consistent with the spread being computed from the same zero rate, so the spread on that row is probably wrong too (ASSUMPTION). The spread is **not** changed in Stage 2; see D-017.

### D-010 `Upfront_charges = 0` is kept as a legitimate "no fee" value
- **Type:** ASSUMPTION
- **Status:** Agreed (reassessed in Stage 3; no change)
- **Why:** 20,770 zeros (FACT, Stage 2) is too frequent to be a random error, and "no upfront fee" is a normal product feature.
- **Stage 3 reassessment (FACT; `notebooks/01_eda.ipynb`, section 7):**
  - Zero fees have a 0.26% default rate, against 0.11% for positive fees. Both are far below the portfolio rate, because a present `Upfront_charges` value almost never occurs on a default (D-011).
  - Zeros are more common in `type2` loans (22.8% of present values) than in `type1` (11.7%).
  - Nothing contradicts the "no fee" reading.
- **Scope:** `Upfront_charges` is excluded from the main model under D-017, so this assumption only affects the leakage demonstration model.

### D-011 Mandatory investigation of missingness in the pricing variables
- **Type:** Investigation (the result is FACT; the treatment is a MODELLING CHOICE)
- **Status:** Agreed (investigation completed in Stage 3; treatment in D-017)
- **Scope:** `rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`.
- **Plan:**
  1. Measure the missing rate of each variable and whether the three are missing on the same rows.
  2. Compare the default rate for missing vs. non-missing rows.
  3. Only then assess whether these variables would realistically be available at the intended prediction point (the application / origination decision).
- **No conclusion is assumed in advance.** Possible outcomes range from "harmless missingness" to "the missingness carries information about the outcome that would not be available at application time (target leakage)". The treatment (a missing indicator, imputation, or excluding the variable) is decided only after the evidence is in.
- **Stage 2 evidence (FACT; steps 1 and 2 of the plan).** Source: `artifacts/dq_missingness.csv`.

  | Variable | Missing | % | Default rate if missing | Default rate if present |
  |---|---:|---:|---:|---:|
  | `Upfront_charges` | 39,642 | 26.66 | 92.0% | 0.14% |
  | `Interest_rate_spread` | 36,639 | 24.64 | **100%** | **0%** |
  | `rate_of_interest` | 36,439 | 24.51 | 100% | 0.18% |

  - `Interest_rate_spread` is missing **if and only if** `Status = 1`. Its missingness alone separates the target perfectly.
  - The three variables are missing together on 36,439 rows, all of which have `Status = 1`. Every row with `rate_of_interest` missing also has the other two missing.
  - The pattern extends beyond the pricing variables; see D-017.
- **Not concluded (Stage 2):** whether this is target leakage, an artefact of how the dataset was assembled, or something else. Nothing was removed, imputed or transformed because of this finding.
- **Stage 3, step 3 (availability at the prediction point):**
  - **FACT** (`notebooks/01_eda.ipynb`, sections 3.1 and 5): when the three pricing variables are present, the default rate is close to 0% in every decile. Their predictive power comes almost entirely from *whether* they are recorded, not from their values.
  - **ASSUMPTION:** in a real lending process, the rate, spread and fees are set at or before origination, so they would exist for every approved loan. A record where they are missing only for loans that later defaulted does not reflect the application process. It more likely reflects how the dataset was assembled, or data recorded after the outcome. The cause cannot be verified without documentation.
  - **Treatment:** the three variables are excluded from the main model (D-017).

### D-012 Logistic regression is the primary model
- **Type:** MODELLING CHOICE
- **Status:** Agreed (Stage 5)
- **Why:** It outputs a probability directly. Each coefficient is readable as a change in log-odds, so its direction can be checked against credit intuition. It is stable, easy to validate and well established for PD modelling. A challenger model is optional and only comes after the baseline is complete.
- **Implementation (Stage 5):** `src/model.py` fits an **unpenalised** logistic regression (`LogisticRegression(C=np.inf)`, scikit-learn's documented way to fit without a penalty), so the coefficients are plain maximum-likelihood estimates rather than shrunk ones, and match a `statsmodels.Logit` fit on the same design matrix (`coefficient_table`, both used for cross-checking). `class_weight` is left at its default, so predicted probabilities stay calibrated to the development sample's 24.64% default rate, rather than being rebalanced.

### D-013 Stratified 70/30 split with 5-fold cross-validation and a fixed seed
- **Type:** MODELLING CHOICE
- **Status:** Agreed (Stage 5)
- **Decision:** 70% development/training set, with 5-fold cross-validation performed only within the development/training set, and a 30% final hold-out test set used once for final evaluation. The split is stratified on `Status`, with `RANDOM_SEED = 42` (`TEST_SIZE = 0.30`, `CV_FOLDS = 5` in `src/config.py`). All learned preprocessing (imputation, scaling, encoding, binning) is fitted on the development/training data only, inside a scikit-learn Pipeline.
- **Why:** Stratification keeps the default rate equal in both samples. Cross-validation inside the development/training set is used for model development without touching the hold-out test set, so the final hold-out result remains an unbiased estimate. Fitting only on development/training data prevents information from the hold-out test set leaking into the model. This is **out-of-sample**, not out-of-time, validation (see D-004).
- **Implementation (Stage 4, D-020):** the split is made once by `src/db.py` and stored in DuckDB as the table `sample_split` (`ID` → `development` / `holdout`). Later stages read it from there instead of re-splitting.

### D-014 Risk grades are illustrative internal grades for this project
- **Type:** MODELLING CHOICE
- **Status:** Open; boundaries decided in Stage 7
- **Decision:** Predicted PD is mapped to 10 grades (A to J). They are described as **illustrative internal risk grades for this project**. They are not an official banking methodology, not a regulatory master scale, and not an underwriting decision.
- **Checks planned:** default rates rise from grade to grade, each grade has enough observations, and the construction of the boundaries is documented transparently.

### D-015 The definition of `Status` (default) is undocumented
- **Type:** ASSUMPTION
- **Status:** Agreed; documented as a limitation
- **Decision:** `Status = 1` is treated as "default" as labelled. No default definition has been found in the dataset description yet; this must be confirmed against the original Kaggle dataset documentation before treating the target definition as established.
- **Consequence:** The predicted PDs are meaningful **relative to this dataset only**. They are not comparable to a regulatory 12-month PD. The portfolio default rate of 24.64% is far above typical mortgage-portfolio levels, which reinforces this caveat.
- **Stage 2 evidence (FACT):** `Status` is present in every row and takes only the values 0 (112,031 rows) and 1 (36,639 rows), a default rate of 24.64%. The data itself cannot establish what `Status = 1` means, so the definition remains unconfirmed.

### D-016 Use of `Gender` (and possibly `age`) as a model feature
- **Type:** MODELLING CHOICE
- **Status:** Agreed (decided after Stage 3)
- **Issue:** Using protected characteristics such as sex in credit decisions is legally and ethically sensitive in the EU. Even when a variable is predictive, a bank would need a strong justification and a legal review.
- **Options:**
  - (a) Exclude `Gender` from the model and keep it only for descriptive analysis.
  - (b) Include it and document the concern.
- **Decision:** (a). `Gender` is excluded from all models and may be used only for descriptive analysis. `age` is also excluded from the main model, for the D-017 reason (its 200 missing values are all defaults), so its sensitivity does not need a separate decision.

### D-017 Variables whose missingness or category almost perfectly separates `Status`
- **Type:** FACT (the pattern); the treatment is a MODELLING CHOICE. The cause remains unknown.
- **Status:** Agreed (treatment decided after the Stage 3 investigation; applied in Stage 5)
- **Finding (Stage 2):** Beyond the pricing variables in D-011, several fields have a missing value or category whose default rate is 100% or close to it. Sources: `artifacts/dq_missingness.csv` and `artifacts/dq_categorical_levels.csv`.

  | Indicator | Rows | Default rate | Default rate otherwise |
  |---|---:|---:|---:|
  | `Interest_rate_spread` missing | 36,639 | 100% | 0% |
  | `rate_of_interest` missing | 36,439 | 100% | 0.18% |
  | `property_value` / `LTV` missing | 15,098 | 99.99% | 16.1% |
  | `credit_type = EQUI` | 15,298 | 99.99% | 16.0% |
  | `Upfront_charges` missing | 39,642 | 92.0% | 0.14% |
  | `dtir1` missing | 24,121 | 67.6% | 16.3% |
  | `age` and `submission_of_application` missing (same 200 rows) | 200 | 100% | 24.5% |
  | `construction_type = mh` / `Secured_by = land` / `Security_Type = Indriect` (same 33 rows) | 33 | 100% | 24.6% |

- **Why it matters:** A model given these fields could look almost perfect without learning anything about borrower risk. If the fields are only filled in (or only left empty) *after* the loan's outcome is known, using them would be **target leakage**. At the moment of the credit decision, the model would not have that information.
- **Possible explanations (none is assumed):** the defaulted and non-defaulted records may come from different source systems or extraction processes; the fields may be recorded after the outcome; or the pattern may be genuine but specific to this dataset.
- **Related point:** the `Interest_rate_spread` on the D-009 row (`ID` 61162) is probably derived from the invalid zero rate. This no longer needs a decision, because the spread is excluded from the main model (below).
- **Stage 2 action:** None. No variable was removed, imputed or recoded because of this finding.
- **Stage 3 evidence (FACT; `notebooks/01_eda.ipynb`, section 5):**
  - Every default has at least one of `rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`, `dtir1`, `property_value` or `income` missing. The 101,333 rows where all six are present contain **no** defaults. So no subset of rows is free of the pattern, and the problem can only be handled by choosing columns.
  - The 200 defaults that do have an interest rate are exactly the rows with `age` and `submission_of_application` missing. All of them are EQUI with `dtir1` missing.
  - Differences in default rate between category levels remain after the EQUI records are set aside (section 3.2; for example `loan_type` ranges from 14.4% to 25.8%). The categorical variables therefore appear to carry signal beyond the leakage pattern.
  - *Correction:* a Stage 3 draft compared category default rates with each level's share of missing `rate_of_interest`. That comparison is circular, because a missing rate is almost identical to `Status = 1`, so it was replaced by the comparison outside EQUI.
- **Decision (options A + C):**
  - **Admissibility rule for the main model:** a field is used only if it would be captured for every applicant at the time of the credit decision, **and** its values or missingness are not driven by the outcome. The rule is applied consistently, whatever the number of rows affected. Imputation cannot repair missingness that is caused by the outcome, so these fields are excluded rather than imputed.
  - **Excluded under this rule:**
    - `rate_of_interest` / `rate_of_interest_clean`, `Interest_rate_spread` and `Upfront_charges` (D-011);
    - `credit_type` (EQUI is 99.99% default);
    - `property_value` / `property_value_clean` / `LTV` / `LTV_clean` (missing: 99.99% default);
    - `dtir1` (missing: 67.6% default vs 16.3%);
    - `age` and `submission_of_application` (missing: 100% default);
    - raw `income` (replaced by `income_clean`, D-008).
  - **Kept:** fields whose missing values default at close to the portfolio rate (`loan_limit`, `approv_in_adv`, `loan_purpose`, `Neg_ammortization`, `term`). Their missing values are handled inside the model pipeline in Stage 5.
  - **Main model candidates (16, before Stage 5 screening):** `config.MAIN_MODEL_CANDIDATE_FEATURES`. Every excluded column and its reason is in `config.EXCLUDED_FROM_MAIN_MODEL`, and `tests/test_config.py` checks that together they cover every processed column exactly once.
  - **Leakage demonstration (option C):** a separate full-feature model, clearly labelled as a leakage demonstration. It is never used for risk grades, the dashboard or reported PDs.
- **Main limitation (ASSUMPTION about real practice):** a mortgage PD model without LTV and DTI lacks its core risk drivers, and would not pass a conceptual-soundness review for production use. In a real bank, this would be raised as a data-quality finding with the data owner, and the missing values would be requested from the source systems. That is not possible here. The main model is therefore a prototype built on data that failed its fitness-for-use check, and it is documented as such.
- **Resolved in Stage 5 (D-022):** `lump_sum_payment = lpsm` (development sample: 76.4%, 64.4% outside EQUI) is kept in the main model. The rate holds up in the subset with every other leakage-pattern field present (64.3%) and in every `loan_type` (61-73%), and the coefficient stays strongly positive in a sensitivity fit that excludes every EQUI row -- so unlike the D-017 fields, it is not explained by the leakage pattern.

- **Update after Stage 6 (D-026):** the EQUI rows are also left out of the main model's **population**, not just its features. They stay in the data and in the leakage demonstration.

### D-018 Categorical values are kept exactly as recorded
- **Type:** MODELLING CHOICE (for Stage 2); the observations below are FACT
- **Status:** Agreed for Stage 2; recoding or grouping categories is decided in Stage 3/5
- **Decision:** Stage 2 does not rename, merge, collapse or drop any category, and does not convert category labels into missing values.
- **Stage 2 observations (FACT; full list in `artifacts/dq_categorical_levels.csv`):**
  - **Duplicated information:**
    - `construction_type = mh`, `Secured_by = land` and `Security_Type = Indriect` mark exactly the same 33 rows, so the three columns carry the same information.
    - `business_or_commercial = b/c` marks exactly the same 20,762 rows as `loan_type = type2`.
  - **Encoding oddities:**
    - `Security_Type = "Indriect"` is a spelling error in the source.
    - `Region` mixes capitalisation (`North`, `North-East`, but `south`, `central`).
    - `Gender = "Sex Not Available"` (37,659 rows) is in effect a missing-value code stored as a category.
  - **No whitespace problems** were found in any text value.
  - **Missing categories:** `loan_limit` (3,344), `approv_in_adv` (908), `age` and `submission_of_application` (200), `loan_purpose` (134) and `Neg_ammortization` (121).
- **Why:** Fixing labels is cosmetic, but merging categories or treating "Sex Not Available" as missing changes the information a model sees. Those are modelling choices and should be made with the Stage 3 evidence. Redundant columns (for example `Secured_by` next to `construction_type`) are a feature-selection question for Stage 5.
- **Redundant columns, decided after Stage 3 (MODELLING CHOICE, Agreed; the duplication is reproduced in `notebooks/01_eda.ipynb`, section 9):**
  - `loan_type` is kept, and `business_or_commercial` (identical to `loan_type = type2`) is excluded.
  - `construction_type`, `Secured_by` and `Security_Type` are all excluded. They are almost constant, and their single rare level (33 rows) is 100% default, which matches the D-017 leakage pattern.
  - `Gender`, including "Sex Not Available", is excluded under D-016, so how that label is coded no longer matters for the model.
  - Label clean-ups (the "Indriect" spelling, `Region` capitalisation) remain unnecessary: none of the affected columns is a model candidate except `Region`, where capitalisation does not change the levels.

### D-019 `Credit_Score` has no ranking power and is excluded from the main model
- **Type:** FACT (the finding); MODELLING CHOICE (the exclusion)
- **Status:** Agreed (decided after Stage 3)
- **Stage 3 evidence (FACT; `notebooks/01_eda.ipynb`, section 8):**
  - `Credit_Score` takes integer values from 500 to 900, spread evenly. A chi-square test of uniformity over the 401 values gives p = 0.86, so uniformity is not rejected.
  - Used directly as a score, it has an AUC of 0.503, and the default rate is flat across its deciles.
  - Its largest absolute Spearman correlation with any other numeric column is below 0.01.
- **Decision:** excluded from the main model, because it fails univariate screening.
- **Not claimed:** why the column behaves like this. Whether it is synthetic or on an unknown scale cannot be established from the data. A real bureau score ranks risk, so in a real bank a score with no signal would trigger a data investigation. It is recorded as a data limitation.

### D-020 DuckDB is the single data source for Stages 5 to 9, with a stored sample split
- **Type:** MODELLING CHOICE (the split sizes and default rates are FACT)
- **Status:** Agreed (implemented in Stage 4)
- **Decision:**
  - `src/db.py` loads `loans_clean.parquet` into `data/processed/credit_risk.duckdb` (gitignored).
  - The D-013 split is made once, with scikit-learn `train_test_split` (stratified on `Status`, `TEST_SIZE`, `RANDOM_SEED`), and stored as the table `sample_split`. The IDs are sorted first, so the split does not depend on row order.
  - The view `model_dataset` (`sql/model_dataset.sql`) contains only `ID`, `sample`, `Status` and the 16 D-017 candidate features. The model reads its data from this view, so an excluded column cannot reach it by accident.
  - The SQL analyses in `sql/` write aggregate tables to `artifacts/sql_*.csv`.
- **Why:** a bank records which loans were used for development and which for validation as a flag on the data, so every later step (validation, grades, monitoring, dashboard) uses exactly the same samples. Recreating the split in each stage would depend on everyone using the same call, seed and row order. The split method itself is unchanged (D-013).
- **Stage 4 evidence (FACT; `tests/test_db.py`):**
  - development 104,069 loans (25,647 defaults, 24.644%); hold-out 44,601 loans (10,992 defaults, 24.645%);
  - rebuilding the database gives an identical split;
  - the SQL missingness table (`sql/dq_missingness.sql`) reproduces the pandas table `artifacts/dq_missingness.csv` for every column, so the D-011 / D-017 evidence is confirmed by a second, independent implementation.
- **Bug fixed in Stage 5:** `sql/risk_deciles.sql`'s `NTILE` ordered only by value, so loans tied on a repeated value (many loan amounts and incomes are rounded) could fall in a different bin on each rebuild. Its `ORDER BY` now breaks ties on `ID`; `tests/test_db.py` checks that a rebuild reproduces the table exactly.

### D-021 Segments with fewer than 10 loans are not written to the SQL segment tables
- **Type:** HEURISTIC (a small-cell rule of thumb for this project, not a regulatory standard)
- **Status:** Agreed (implemented in Stage 4)
- **Issue (FACT, Stage 4):** before this rule, `artifacts/sql_portfolio_by_segment.csv` had 3 rows with a single loan (rare `term` values 165, 280 and 322 months). Such a row shows that one loan's amount and whether it defaulted, which is in effect a row-level record in a committed file.
- **Decision:** `sql/portfolio_by_segment.sql` and `sql/risk_segment_crosses.sql` leave out segments with fewer than `config.SQL_MIN_SEGMENT_SIZE = 10` loans. Shares are computed over all loans **before** the filter, so the rows shown are unchanged. The hidden rows are the reason the shown shares of a column can add up to slightly less than 1. The `risk_deciles` bins are far above the threshold, and a test checks all three tables.
- **Effect (FACT):** 4 `term` levels (1 to 8 loans each) are hidden from the portfolio table, and 6 rows from the crosses table. The data itself is not changed; the rule only affects what is written to `artifacts/`.
- **Why:** hiding small cells is common practice in published aggregate reports. The value 10 is a judgement call, chosen low enough to keep every real segment of interest.

### D-022 Stage 5 feature screening: 6 of the 16 D-017 candidates go into the main model
- **Type:** MODELLING CHOICE (the evidence is FACT); the IV screen threshold is HEURISTIC
- **Status:** Agreed (Stage 5)
- **Method:** Information Value (IV) computed on the development sample, **outside** `credit_type = EQUI` so the leakage category does not inflate it (`src/eda.information_value`, `src/model.screening_table`, `artifacts/model_screening.csv`). Two candidates are dropped for cause before the IV screen is even applied; the rest are screened at `IV_MIN = 0.02` (a common rule-of-thumb cut-off for "not predictive", not a regulatory standard).
- **`term` is dropped (evidence FACT).** `term = 300` defaults at 56.7% overall and 44.4% outside EQUI, far above every other term (~16% outside EQUI). Outside EQUI, the effect is almost entirely one cell: **579 loans with `term = 300` and `Neg_ammortization = neg_amm` default at 91.4%**, in every `loan_type` (90-94%). `term = 300` *without* `neg_amm` defaults at 16.4%, the same as everything else. No credit reason explains a threefold jump for a 25-year negatively-amortising loan over a 30-year one; this is treated as a new D-017-type anomaly (an unexplained, near-deterministic cell) rather than a genuine term effect, and is not modelled. `Neg_ammortization` is kept: outside the cell it still defaults at 29.1% against 13.3%, and its own coefficient partly absorbs the cell (documented, not hidden).
- **`co-applicant_credit_type` is dropped (evidence FACT): it is a proxy for EQUI.** Development EQUI loans are 10,676 `EXP` and 2 `CIB`. Overall, `EXP` defaults at 30.9% against 18.4% for `CIB` -- but **outside EQUI, `EXP` defaults at 13.0%, below `CIB`'s 18.4%: the direction reverses.** A model that kept this column would learn a positive `EXP` effect that comes entirely from the leakage category. This fails the D-017 admissibility rule the same way the excluded fields do, so it is treated the same way: excluded, not imputed or recoded.
- **`lump_sum_payment` is kept** despite being a partial EQUI proxy (34% of `lpsm` loans are EQUI, against 10% overall) -- see the D-017 update above.
- **IV screen (outside EQUI):** kept -- `income_clean` 0.158, `lump_sum_payment` 0.137, `Neg_ammortization` 0.133, `loan_type` 0.070, `loan_amount` 0.061, `loan_purpose` 0.020. Screened out (all < 0.02) -- `Region` 0.019, `loan_limit` 0.014, `occupancy_type` 0.011, `approv_in_adv` 0.010, `total_units` 0.007, `Credit_Worthiness` 0.005, `interest_only` 0.001, `open_credit` 0.001.
- **Result:** `config.MAIN_MODEL_FEATURES` (6): `income_clean`, `loan_amount`, `lump_sum_payment`, `Neg_ammortization`, `loan_type`, `loan_purpose`. `config.SCREENED_OUT_FEATURES` records the other 10, each with its reason. `tests/test_config.py` checks the two sets partition `MAIN_MODEL_CANDIDATE_FEATURES` exactly.
- **Limitation of the screen (known, accepted):** the screen is computed once on the whole development sample, including the rows each CV fold later validates on; it is not repeated inside every fold. The CV metrics (D-023) are therefore slightly optimistic, most for borderline features such as `loan_purpose` (IV 0.0204 against the 0.02 cut-off). This follows common practice (screen on the development sample, validate on data the screen never saw): the Stage 6 hold-out played no part in screening and is the unbiased check.

### D-023 Main model preprocessing: missing values, encoding and numeric transforms
- **Type:** MODELLING CHOICE
- **Status:** Agreed (Stage 5); implemented in `src/model.build_preprocessor`, fitted only on development data / training folds inside a scikit-learn Pipeline (D-013)
- **Missing values:** `loan_purpose` (95 rows) and `Neg_ammortization` (85 rows) are imputed with their most frequent level -- both are under 0.1% of the development sample and close to the majority level's rate, so this is a minor, low-risk choice. `income_clean` is missing on 7,370 development rows (7.1%; those rows default at 24.1%, and at 14.6% outside EQUI against 16.1% with income present). That is too frequent to impute silently: it is imputed with its (log-scale) median, **plus a binary missing-value indicator**, so the model can learn whether missingness itself carries risk, rather than hiding it. (It does not, materially: the indicator's coefficient is small and negative -- see D-024's ablation for the contrast with the D-017 fields, where missingness *is* the risk signal.)
- **Encoding:** one-hot, with the most frequent level as the dropped reference (`type1`, `p3`, `not_neg`, `not_lpsm` -- `config.REFERENCE_LEVELS`), so every coefficient reads as "vs the most common level". **Rare-level rule (HEURISTIC):** a level below 1% of development rows (`config.RARE_LEVEL_MIN_SHARE`) would need grouping before encoding; `src/model.check_rare_levels` raises loudly if one appears rather than merging silently. As of Stage 5 the rarest kept level (`lump_sum_payment = lpsm`, `loan_purpose = p2`) is 2.2%, so the rule has not had to act. **Unseen levels:** the encoder uses `handle_unknown="error"`, so a level never seen in training (for example a new `loan_type` in the hold-out or at scoring) raises an error instead of being scored silently as the reference level. Missing values are different: they are imputed with the most frequent level, as decided above.
- **Numeric transform:** `loan_amount` and `income_clean` are right-skewed (skew 1.5 and 19 on the development sample). Both are `log`-transformed, then standardised; no capping. A univariate check confirmed the log scale fits the log-odds far better than the raw scale (correlation with the binned log-odds around -0.87, against about -0.57 raw). Both variables have a mild uptick in their top ~5%, a known misfit to be checked against calibration by decile in Stage 6.
- **Model:** unpenalised logistic regression (D-012's Stage 5 implementation note).
- **`loan_amount`'s sign reverses in the model; kept as is (decided after Stage 5 review).**
  - **Evidence (FACT; Stage 5 fit on all development rows, before D-026):** the fitted coefficient is **positive** (+0.062, `sm_p_value` < 0.001, the same sign in all 5 CV folds), the opposite of its univariate direction (negative: larger loans default less often on their own). `loan_amount` and `income_clean` are correlated at about 0.63 in log scale. Fitted alone, `log(loan_amount)` has a coefficient of -0.285; adding `log(income_clean)` turns it positive.
  - **Interpretation (ASSUMPTION):** the univariate effect is mostly income in disguise -- larger loans go to higher-income borrowers, who default less. Once income is held fixed, what remains in `loan_amount` is loan size *relative to income*: a larger loan for the same income means higher leverage and a higher repayment burden, which a bank would expect to raise PD. The positive sign therefore has a credit rationale. It stands in for the loan-to-income and debt-to-income information the model otherwise lacks (`dtir1` is excluded under D-017).
  - **Decision:** keep `loan_amount` unchanged. The effect is small (odds ratio 1.06 per standard deviation of log loan amount), stable across folds and interpretable. Dropping it would remove the only leverage signal left in the model.
  - **What stays visible:** `config.EXPECTED_SIGNS` keeps the univariate prior (-1), so `artifacts/model_coefficients.csv` still reports `sign_matches_expected = False` for `loan_amount`. The prior is not rewritten after seeing the result; the mismatch is explained here instead.
  - **Revisit if:** the coefficient changes sign or loses significance on the Stage 6 hold-out, or the Stage 6 calibration by `loan_amount` decile shows a misfit (see the top-5% uptick noted above).
  - **Stage 6 outcome (FACT; D-025 second look, model fitted in scope under D-026):**
    - After the D-026 refit the coefficient is larger: +0.117 in development, and still positive and significant on the hold-out refit (+0.100, p < 0.001). The first trigger did not fire.
    - The top-decile misfit remains:
      - top `loan_amount` decile under-predicted by 3.4 pp (PD 12.1%, observed 15.5%);
      - top `income_clean` decile under-predicted by 3.5 pp (PD 8.5%, observed 12.1%).
      Both are inside the 5 pp amber limit.
    - **Decision (MODELLING CHOICE, Agreed; Bruno, 2026-10-01):** this does **not** count as the second trigger firing. The top-decile misfit is accepted as a documented limitation, and `loan_amount` and `income_clean` stay as single log-linear terms. Figures: `reports/figures/11_calibration_by_feature.png`, `notebooks/03_validation.ipynb`.

### D-024 Leakage-demonstration model (option C)
- **Type:** MODELLING CHOICE
- **Status:** Agreed (Stage 5); implemented in `sql/leakage_demo_dataset.sql` and `src/model.run_leakage_demo`
- **Decision:** two models, both scored with 5-fold CV on the development sample only, **never** on the hold-out, for risk grades, monitoring or the dashboard:
  - **Full model:** the 6 main-model features plus every D-017-excluded field in its clean version where one exists (`rate_of_interest_clean`, `Interest_rate_spread`, `Upfront_charges`, `property_value_clean`, `LTV_clean`, `dtir1`, `credit_type`, `age`, `submission_of_application`), plus `term` and `co-applicant_credit_type` (D-022). It is built on the main model's own preprocessing, so the two stay comparable. The missingness stays visible: each extra numeric field gets a missing-value indicator next to its median-imputed value, and each extra categorical field keeps missing as its own level. Uses the default L2-penalised logistic regression: with near-perfect separation, an unpenalised fit does not converge.
  - **Ablation:** only whether each of `rate_of_interest_clean`, `Interest_rate_spread`, `Upfront_charges`, `property_value_clean`, `dtir1`, `age` is missing, plus `credit_type` (`submission_of_application` is missing on exactly the same rows as `age`, so it is left out to avoid a duplicate indicator).
- **Stage 5 evidence (FACT; `artifacts/model_cv_metrics.csv`):** mean CV AUC -- main model 0.651 (all development rows; 0.675 in scope after D-026), full leakage model **1.000**, indicators-only ablation **1.000**. The ablation shows that missingness alone is enough for a perfect score: `Interest_rate_spread` is missing if and only if `Status = 1` (D-017), so its indicator by itself separates the target. The apparent skill comes from how the dataset was assembled, not from borrower risk.
- **Correction during Stage 5 review:** a first version of the full model median-imputed the extra fields without missing indicators, which hid the very signal it was meant to demonstrate (it scored 0.858, below the ablation). It was fixed before commit.
- **Why:** this shows concretely, on this project's own data, how a model that ignored D-017 would look almost perfect while learning nothing about borrower risk -- the central caution of the project (see README limitations).

- **Note after D-026:** the leakage models still use all development rows (EQUI is part of the leakage they show), while the main model's CV now covers only in-scope rows (AUC 0.675). The contrast (1.000 against 0.675) still makes the point, but the two populations are no longer identical.

### D-025 Stage 6 validation design and pre-set criteria
- **Type:** MODELLING CHOICE; the thresholds are HEURISTIC (judgement calls for this project, not regulatory standards)
- **Status:** Agreed (Stage 6). Written **before** the hold-out was scored.
- **Design:**
  - The frozen Stage 5 model (`artifacts/pd_model.joblib`) scores the 30% hold-out once. It is never refitted or changed in response to hold-out results without Bruno's approval, and any such change is logged here (the hold-out would then no longer be an unbiased estimate).
  - Calibration: mean PD vs observed default rate (two-sided binomial test), calibration intercept and slope, a PD-decile table, and Hosmer-Lemeshow. Hosmer-Lemeshow is reported **without** a pass/fail judgement, because at about 44,000 loans it rejects even trivial misfit.
  - Extras: results for all loans and outside `credit_type = EQUI` (`credit_type` is used only for this split, never as a feature; superseded by D-026, under which only in-scope loans are scored); 95% bootstrap intervals (1,000 resamples, seed 42); performance by `loan_type` and `loan_purpose`; calibration by `loan_amount` and `income_clean` decile.
  - Diagnostic refit: the same features are refitted on the hold-out only to compare coefficient signs and significance with development (the D-023 trigger). The refit is never used for scoring, grades or decisions.
  - `pd_scores` (`ID`, `sample`, `pd`) is stored in DuckDB for Stages 7 and 8. Development scores are in-sample.
- **Pre-set criteria (whole hold-out only; `config.py`):**

  | Criterion | Green | Amber | Red |
  |---|---|---|---|
  | CV mean AUC minus hold-out AUC | ≤ 0.02 | ≤ 0.05 | > 0.05 |
  | Binomial test, observed defaults vs mean PD | p ≥ 0.05 | 0.01 ≤ p < 0.05 | p < 0.01 |
  | Calibration slope | 0.90 to 1.10 | 0.80 to 1.20 | outside |
  | Largest PD-decile gap, observed vs mean PD | ≤ 2 pp | ≤ 5 pp | > 5 pp |
- **Implementation:** `src/holdout.py` (`python -m src.holdout`), metric functions in `src/validation.py`, SQL in `sql/validation_*.sql`, tables in `artifacts/validation_*.csv`. The runner is a separate module from `validation.py` because `model.py` already imports `validation.py`.
- **First look at the hold-out (FACT; superseded by the second look below).** Model fitted on all development rows, scored on all 44,601 hold-out loans:
  - hold-out AUC 0.651, against a CV mean of 0.651;
  - mean PD 24.68% against an observed 24.65%;
  - criteria: 3 green, 1 amber (largest PD-decile gap 3.1 pp).
  - **But outside `credit_type = EQUI`, mean PD was 24.05% against an observed 15.94%, with every PD decile 4 to 11 pp too high.** The overall "green" calibration came from two errors that cancelled: EQUI loans were under-predicted (PD about 30%, observed 100%), and all other loans over-predicted. The error was already visible in development (in-sample, outside EQUI: PD 24.1% against observed 16.0%). This led to D-026.
- **Second look (D-026; model refitted on in-scope development rows, scored on the 39,981 in-scope hold-out loans; `artifacts/validation_*.csv`).** The hold-out has now been looked at twice, so these figures are **not fully unbiased**. The change was structural: it chose which population is modelled, it was based on evidence already visible in development, and it was not tuned to hold-out metrics. Features, preprocessing and the criteria above are unchanged.
  - **Discrimination:**
    - hold-out AUC 0.670 (95% bootstrap interval 0.662 to 0.678), against a CV mean of 0.675;
    - Gini 0.341, KS 0.267, Brier 0.122.
  - **Calibration:** mean PD 15.97% against an observed 15.94% (binomial p = 0.86); slope 1.008; intercept -0.003.
  - **Pre-set criteria: 3 green, 1 amber.**
    - AUC drop 0.004: green.
    - Binomial test: green.
    - Slope: green.
    - Largest PD-decile gap 2.5 pp, in decile 1 (mean PD 6.8%, observed 9.2%): amber. Decile 9 is next at 2.3 pp.
  - **Hosmer-Lemeshow** rejects (statistic 88 on 8 df), as expected at this sample size. Not used for a judgement.
  - **Segments:** AUC 0.64 to 0.69. The largest gap is `loan_purpose = p2` (857 loans), under-predicted by 4.3 pp. Every other level is within 1 pp.
  - **Diagnostic coefficient refit:**
    - 9 of 10 terms keep their sign. The exception is `loan_purpose_p4`, which is not significant in either sample (development -0.003, hold-out +0.027, p = 0.45).
    - `income_clean`'s 95% intervals do not overlap: development -0.415, hold-out -0.350. Same direction, somewhat weaker on the hold-out.
  - **Out of scope:** 4,620 hold-out EQUI loans (100% default) and 10,678 development EQUI loans (99.99%) are counted in `validation_scope.csv`, but not scored.

### D-026 The main model's scope excludes `credit_type = EQUI`
- **Type:** MODELLING CHOICE (the evidence is FACT)
- **Status:** Agreed (decided after the first Stage 6 look at the hold-out)
- **Issue (FACT):**
  - The Stage 5 model was fitted on all development rows, including 10,678 EQUI loans (10.3%, 99.99% default), but the model cannot recognise them because `credit_type` is not a feature (D-017).
  - It therefore under-predicted EQUI (PD about 29.5%) and over-predicted every other loan (PD 24.1% against observed 16.0%, in-sample).
  - The coefficients were distorted as well. Refitted outside EQUI on development rows only, `income_clean` goes from -0.284 to -0.415, `loan_amount` from +0.062 to +0.117, and `loan_purpose_p1` changes sign (+0.091 to -0.172).
  - It was also inconsistent with D-022, which screened features on evidence from outside EQUI precisely because EQUI distorts it.
- **Options considered:**
  - (A) leave the bias documented;
  - (B) recalibrate only the intercept on non-EQUI loans, a patch that leaves the coefficients distorted;
  - (C) take EQUI out of the model's population.
- **Decision:** (C). The view `model_scope_dataset` (`sql/model_scope_dataset.sql`, `config.MODEL_SCOPE_VIEW`) is `model_dataset` without EQUI rows. Stage 5 fits on it and Stage 6 scores it; `pd_scores` holds in-scope loans only.
  - `credit_type` is used only to decide scope; it is still never a feature.
  - EQUI rows are not deleted. They stay in `loans_clean`, the screening tables (D-022) and the leakage demonstration (D-024), and are counted in `artifacts/validation_scope.csv`.
- **Why:** EQUI is treated as a marker of how the dataset was assembled, not as a borrower population (D-017). A PD that is calibrated only on average across two populations it gets very wrong would not pass a bank validation review. Excluding a documented, defective sub-population from the development sample is a normal, documented model-scope choice.
- **Unchanged:** features (D-022, and `model_screening.csv` is identical), preprocessing and reference levels (D-023; the most frequent levels are the same outside EQUI, and the rarest kept level, `lpsm`, is 1.65% of in-scope development rows, above the 1% rule), and the D-025 criteria.
- **Stage 5 result in scope (FACT; `artifacts/model_cv_metrics.csv`, `model_coefficients.csv`):** 93,391 development loans, 16.03% default rate. Mean CV AUC 0.675, Gini 0.349, KS 0.273, Brier 0.123.
- **Cost:** the hold-out was evaluated a second time after this change (D-025), so the hold-out figures are no longer a fully unbiased estimate. This is stated wherever they are reported.
- **Limitation:** the model says nothing about EQUI loans. If EQUI were a real applicant group at decision time, they would need their own treatment. The data cannot tell us (D-017).

---

## Future enhancements (explicitly out of scope for the one-week MVP)

- Weight of Evidence (WoE) binning and Information Value (IV) analysis; a WoE-based scorecard version of the model.
- A challenger model (for example gradient boosting) compared against the logistic baseline.
- Simulated stress / drift scenarios for monitoring. The MVP covers basic PSI / stability analysis only.
- Out-of-time validation, if a dataset with real temporal variation becomes available.
