"""Central project configuration.

Every path, random seed and data-quality threshold used anywhere in the project
(pipeline, SQL layer, notebooks, dashboard, tests) is defined here once, so a
threshold can never silently differ between two parts of the project.

Each data-quality threshold references its entry in docs/decision_log.md, where
the reasoning and the decision type (FACT / ASSUMPTION / HEURISTIC /
MODELLING CHOICE) are documented.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Loan_Default.csv"  # placed manually, never committed
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DATA_PATH = PROCESSED_DIR / "loans_clean.parquet"
DUCKDB_PATH = PROCESSED_DIR / "credit_risk.duckdb"

SQL_DIR = PROJECT_ROOT / "sql"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DQ_REPORT_DIR = ARTIFACTS_DIR  # Stage 2 aggregate data-quality tables (dq_*.csv, committed)
MODEL_PATH = ARTIFACTS_DIR / "pd_model.joblib"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# ---------------------------------------------------------------------------
# Reproducibility and sampling  (decision log D-013)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
# 70% development/training set, with 5-fold cross-validation performed only
# within the development/training set, and a 30% final hold-out test set used
# once by design for final evaluation (it was evaluated a second time after the
# D-026 scope change; see D-025). Stratified random split on Status: this is
# out-of-sample, NOT out-of-time, validation (D-004).
TEST_SIZE = 0.30  # share of rows in the final hold-out test set
CV_FOLDS = 5      # stratified folds, drawn from the development/training set only

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
ID_COL = "ID"
TARGET_COL = "Status"  # 1 = default, 0 = non-default (definition undocumented, D-015)
TARGET_VALUES = (0, 1)
YEAR_COL = "year"

# D-001: size of the documented source file, verified by the Stage 2 pipeline.
EXPECTED_N_ROWS = 148_670

# The 34 columns expected in the raw file (Yasser H., "Loan Default Dataset", Kaggle).
# Stage 2 validates the actual file against this list and fails loudly on a mismatch.
EXPECTED_RAW_COLUMNS = [
    "ID", "year", "loan_limit", "Gender", "approv_in_adv", "loan_type",
    "loan_purpose", "Credit_Worthiness", "open_credit", "business_or_commercial",
    "loan_amount", "rate_of_interest", "Interest_rate_spread", "Upfront_charges",
    "term", "Neg_ammortization", "interest_only", "lump_sum_payment",
    "property_value", "construction_type", "occupancy_type", "Secured_by",
    "total_units", "income", "credit_type", "Credit_Score",
    "co-applicant_credit_type", "age", "submission_of_application", "LTV",
    "Region", "Security_Type", "Status", "dtir1",
]

# ---------------------------------------------------------------------------
# Data-quality rules  (values that are set to missing; rows are NEVER deleted)
# ---------------------------------------------------------------------------
# D-005 (HEURISTIC): property values below this are treated as invalid -> NaN.
# Derived from the earlier investigation (e.g. property_value = 8,000 against
# loans of several hundred thousand). Not a banking standard.
PROPERTY_VALUE_MIN_VALID = 10_000

# D-009 (ASSUMPTION): a contractual interest rate <= 0 is implausible -> NaN.
RATE_OF_INTEREST_MIN_EXCLUSIVE = 0.0

# D-008 (MODELLING CHOICE, Stage 3): income <= 0 is treated as invalid -> NaN.
# The 1,260 zero-income rows default at 99.4%, so the zero behaves like a
# leakage indicator (D-017), not like a genuine low income.
INCOME_MIN_EXCLUSIVE = 0.0

# Cleaned columns are ADDED next to the raw ones; raw columns are never overwritten.
PROPERTY_VALUE_CLEAN_COL = "property_value_clean"  # D-005
RATE_OF_INTEREST_CLEAN_COL = "rate_of_interest_clean"  # D-009
LTV_CLEAN_COL = "LTV_clean"  # D-006
INCOME_CLEAN_COL = "income_clean"  # D-008

# D-011: pricing variables whose missingness pattern must be investigated
# against Status before any modelling decision is made about them.
PRICING_COLUMNS_UNDER_INVESTIGATION = [
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges",
]

# ---------------------------------------------------------------------------
# Stage 3 exploratory analysis  (notebooks/01_eda.ipynb)
# ---------------------------------------------------------------------------
# D-017: fields whose missingness (or, for credit_type, one category) almost
# perfectly separates Status. Investigated in Stage 3; treatment still Open.
MISSINGNESS_COLUMNS_UNDER_INVESTIGATION = [
    RATE_OF_INTEREST_CLEAN_COL,
    "Interest_rate_spread",
    "Upfront_charges",
    "dtir1",
    PROPERTY_VALUE_CLEAN_COL,
    "income",
]

# D-017 (MODELLING CHOICE, agreed after Stage 3): feature scope of the main
# (application-time) model. A field is admissible only if it would be captured
# for every applicant at decision time AND its values or missingness are not
# driven by the outcome. Stage 5 screening may remove further candidates.
MAIN_MODEL_CANDIDATE_FEATURES = [
    "loan_amount", "term", INCOME_CLEAN_COL,
    "loan_limit", "approv_in_adv", "loan_type", "loan_purpose", "Credit_Worthiness",
    "open_credit", "Neg_ammortization", "interest_only", "lump_sum_payment",
    "occupancy_type", "total_units", "co-applicant_credit_type", "Region",
]
# Excluded from the main model, with the decision that excludes each one.
EXCLUDED_FROM_MAIN_MODEL = {
    ID_COL: "D-003 identifier",
    "rate_of_interest": "D-017 missing only for defaults",
    RATE_OF_INTEREST_CLEAN_COL: "D-017 missing only for defaults",
    "Interest_rate_spread": "D-017 missing if and only if default",
    "Upfront_charges": "D-017 missingness driven by the outcome",
    "credit_type": "D-017 EQUI is 99.99% default",
    "property_value": "D-017 missingness driven by the outcome",
    PROPERTY_VALUE_CLEAN_COL: "D-017 missingness driven by the outcome",
    "LTV": "D-006/D-017 derived from property_value",
    LTV_CLEAN_COL: "D-017 derived from property_value",
    "dtir1": "D-017 missingness driven by the outcome",
    "income": "D-008 raw value; income_clean is used instead",
    "age": "D-017 missing only for defaults; D-016 sensitive",
    "submission_of_application": "D-017 missing only for defaults",
    "Gender": "D-016 protected characteristic",
    "Credit_Score": "D-019 no ranking power",
    "business_or_commercial": "D-018 duplicates loan_type",
    "construction_type": "D-018 near-constant; rare level 100% default",
    "Secured_by": "D-018 duplicates construction_type",
    "Security_Type": "D-018 duplicates construction_type",
}

# Numeric columns analysed in the EDA (cleaned versions where they exist).
EDA_NUMERIC_COLUMNS = [
    "loan_amount", RATE_OF_INTEREST_CLEAN_COL, "Interest_rate_spread", "Upfront_charges",
    "term", PROPERTY_VALUE_CLEAN_COL, "income", "Credit_Score", LTV_CLEAN_COL, "dtir1",
]
EDA_N_BINS = 10  # deciles for default-rate-by-bin tables
EDA_MIN_LEVEL_SHARE = 0.01  # HEURISTIC, display only: levels below 1% of rows are left out of range charts

# ---------------------------------------------------------------------------
# Stage 4 SQL / DuckDB layer  (src/db.py, sql/*.sql; decision log D-020)
# ---------------------------------------------------------------------------
# DuckDB objects. The SQL files use these names literally; tests/test_db.py
# checks that the built database matches them.
LOANS_TABLE = "loans_clean"          # the processed Parquet, loaded as a table
SPLIT_TABLE = "sample_split"         # ID -> sample, the D-013 split made once
MODEL_DATASET_VIEW = "model_dataset"  # ID, sample, Status + the D-017 candidate features
# D-026: the main model's population -- model_dataset without credit_type = EQUI
# (EQUI_LEVEL below). Stage 5 fits on it and Stage 6 scores it.
MODEL_SCOPE_VIEW = "model_scope_dataset"

# Sample labels stored in SPLIT_TABLE (D-013).
SAMPLE_DEVELOPMENT = "development"  # 70%: model fitting and 5-fold CV
SAMPLE_HOLDOUT = "holdout"          # 30%: final evaluation; used once by design, evaluated again after D-026 (D-025)

SQL_N_BINS = 10  # equal-count bins (NTILE) in sql/risk_deciles.sql
EQUI_LEVEL = "EQUI"  # credit_type level used only to show results "outside EQUI" (D-017)
SQL_ARTIFACT_PREFIX = "sql_"  # aggregate SQL result tables: artifacts/sql_*.csv
# D-021 (HEURISTIC): segments with fewer loans than this are not written to the
# committed segment tables, because a row that small describes (almost) a single
# borrower. A small-cell rule of thumb, not a regulatory standard.
SQL_MIN_SEGMENT_SIZE = 10

# Segments in sql/portfolio_by_segment.sql and sql/risk_segment_crosses.sql.
PORTFOLIO_SEGMENT_COLUMNS = ["loan_type", "loan_purpose", "Region", "occupancy_type", "term"]
SEGMENT_CROSSES = [
    ("loan_type", "loan_purpose"),
    ("loan_type", "occupancy_type"),
    ("Region", "loan_type"),
]

# ---------------------------------------------------------------------------
# Stage 5 baseline logistic regression model  (src/model.py; decision log D-022 to D-024)
# ---------------------------------------------------------------------------
# Screening thresholds (development sample, evidence outside credit_type = EQUI).
IV_MIN = 0.02              # HEURISTIC: below this, Information Value counts as "not predictive"
RARE_LEVEL_MIN_SHARE = 0.01  # HEURISTIC: a level below this share is a rare-level candidate
SCREENING_N_BINS = 10        # deciles used to compute Information Value for numeric features
# A numeric column is binned for IV only if it has more than this many times
# SCREENING_N_BINS distinct values (rule of thumb); otherwise its values are used as
# levels, so a discrete column like `term` (25 values) keeps small groups visible.
SCREENING_BIN_DISTINCT_FACTOR = 3
LOGIT_MAX_ITER = 2_000  # solver iteration cap for every logistic regression in Stage 5

# D-022: of the 16 D-017 candidates, 6 survive screening for the main model.
# `term` and `co-applicant_credit_type` are dropped for cause (see D-022); the rest
# of MAIN_MODEL_CANDIDATE_FEATURES fall to the IV_MIN screen.
MAIN_MODEL_FEATURES = [
    INCOME_CLEAN_COL, "loan_amount", "lump_sum_payment", "Neg_ammortization",
    "loan_type", "loan_purpose",
]
LOG_NUMERIC_FEATURES = [INCOME_CLEAN_COL, "loan_amount"]  # log, then standardised (D-023)
CATEGORICAL_FEATURES = ["lump_sum_payment", "Neg_ammortization", "loan_type", "loan_purpose"]
# One-hot reference level per categorical feature: the most frequent level (D-023).
REFERENCE_LEVELS = {
    "lump_sum_payment": "not_lpsm",
    "Neg_ammortization": "not_neg",
    "loan_type": "type1",
    "loan_purpose": "p3",
}
SCREENED_OUT_FEATURES = {
    "term": "D-022 signal is almost all the unexplained term=300 x neg_amm cell",
    "co-applicant_credit_type": "D-022 EQUI proxy; direction reverses outside EQUI",
    "loan_limit": "D-022 IV < 0.02 outside EQUI",
    "approv_in_adv": "D-022 IV < 0.02 outside EQUI",
    "Credit_Worthiness": "D-022 IV < 0.02 outside EQUI",
    "open_credit": "D-022 IV < 0.02 outside EQUI",
    "interest_only": "D-022 IV < 0.02 outside EQUI",
    "occupancy_type": "D-022 IV < 0.02 outside EQUI",
    "total_units": "D-022 IV < 0.02 outside EQUI",
    "Region": "D-022 IV < 0.02 outside EQUI",
}

# Expected coefficient sign from credit sense, keyed by the fitted feature name
# (numeric features by name; one-hot categoricals as "<column>_<level>"). None
# means no confident prior; those are reported but not asserted against.
EXPECTED_SIGNS = {
    INCOME_CLEAN_COL: -1,
    "loan_amount": -1,
    f"{INCOME_CLEAN_COL}_missing": None,
    "lump_sum_payment_lpsm": +1,
    "Neg_ammortization_neg_amm": +1,
    "loan_type_type2": +1,
    "loan_type_type3": None,
    "loan_purpose_p1": None,
    "loan_purpose_p2": +1,
    "loan_purpose_p4": None,
}

# D-024: the leakage-demonstration model (option C). Never used on the hold-out,
# for risk grades, monitoring or the dashboard.
LEAKAGE_DEMO_VIEW = "leakage_demo_dataset"
# The D-017-excluded fields (clean versions where they exist), plus term and
# co-applicant_credit_type (D-022), added to MAIN_MODEL_FEATURES for the full demo model.
# Numeric ones get a missing-value indicator; categorical ones keep missing as a level,
# so the demo model sees the missingness that carries the leakage (D-024).
LEAKAGE_DEMO_NUMERIC_FEATURES = [
    RATE_OF_INTEREST_CLEAN_COL, "Interest_rate_spread", "Upfront_charges",
    PROPERTY_VALUE_CLEAN_COL, LTV_CLEAN_COL, "dtir1",
]
LEAKAGE_DEMO_CATEGORICAL_FEATURES = [
    "credit_type", "age", "submission_of_application", "term", "co-applicant_credit_type",
]
LEAKAGE_DEMO_EXTRA_FEATURES = LEAKAGE_DEMO_NUMERIC_FEATURES + LEAKAGE_DEMO_CATEGORICAL_FEATURES
# The ablation model: only whether each of these is missing, plus credit_type.
# submission_of_application is missing on exactly the same rows as age (D-017),
# so only age is used to avoid a duplicated indicator.
LEAKAGE_ABLATION_MISSINGNESS_FEATURES = [
    RATE_OF_INTEREST_CLEAN_COL, "Interest_rate_spread", "Upfront_charges",
    PROPERTY_VALUE_CLEAN_COL, "dtir1", "age",
]

MODEL_SCREENING_PATH = ARTIFACTS_DIR / "model_screening.csv"
MODEL_CV_METRICS_PATH = ARTIFACTS_DIR / "model_cv_metrics.csv"
MODEL_COEFFICIENTS_PATH = ARTIFACTS_DIR / "model_coefficients.csv"
# D-022: development-only CV of the main model without these features, reported only
# and never acted on (the lpsm question in D-017). The hold-out is not used.
SENSITIVITY_DROPPED_FEATURES = ["lump_sum_payment"]
LPSM_LEVEL = "lpsm"  # the lump_sum_payment level whose evidence sql/lpsm_evidence.sql reproduces (D-017)
MODEL_SENSITIVITY_PATH = ARTIFACTS_DIR / "model_sensitivity.csv"

# ---------------------------------------------------------------------------
# Stage 6 validation and calibration  (src/validation.py; decision log D-025)
# ---------------------------------------------------------------------------
# HEURISTIC pass/fail thresholds, fixed BEFORE the hold-out was scored. Judgement
# calls for this project, not regulatory standards.
# Each is (green limit, amber limit): within green -> green, within amber -> amber, else red.
CRITERION_AUC_DROP = (0.02, 0.05)          # CV mean AUC minus hold-out AUC
CRITERION_BINOMIAL_P = (0.05, 0.01)        # p-value: >= 0.05 green, >= 0.01 amber, else red
CRITERION_SLOPE_GREEN = (0.90, 1.10)       # calibration slope range
CRITERION_SLOPE_AMBER = (0.80, 1.20)
CRITERION_DECILE_GAP = (0.02, 0.05)        # largest |observed - mean PD| in a PD decile
BOOTSTRAP_N_RESAMPLES = 1_000
BOOTSTRAP_CI_LEVEL = 0.95
CALIBRATION_N_BINS = 10                    # PD deciles (also the Hosmer-Lemeshow groups)
PD_SCORES_TABLE = "pd_scores"              # ID, sample, pd: scores from the frozen model
VALIDATION_SEGMENT_COLUMNS = ["loan_type", "loan_purpose"]
VALIDATION_FEATURE_DECILE_COLUMNS = ["loan_amount", INCOME_CLEAN_COL]
VALIDATION_ARTIFACT_PREFIX = "validation_"  # artifacts/validation_*.csv

# ---------------------------------------------------------------------------
# Stage 7 illustrative risk grades  (src/grades.py; decision log D-014, D-027)
# ---------------------------------------------------------------------------
# HEURISTIC grade-construction thresholds, fixed BEFORE the final grade run (D-027).
# Built from development loans only; the hold-out is never used (D-025).
GRADE_START_BINS = 20      # equal-count PD bins the merging starts from
GRADE_MERGE_ALPHA = 0.05   # each grade must default significantly more than the one below (one-sided)
GRADE_MIN_SHARE = 0.05     # each grade holds at least this share of development loans
GRADE_MAX_GRADES = 10      # upper limit on the number of grades (D-014)
GRADE_EDGE_DECIMALS = 6    # boundaries are rounded once, so the published scale is the one applied
GRADE_LABELS = "ABCDEFGHIJ"  # A = lowest PD
PD_MIN, PD_MAX = 0.0, 1.0    # a PD outside [PD_MIN, PD_MAX] or NaN is an error, never graded or clipped
GRADE_SCALE_TABLE = "grade_scale"  # grade, grade_rank, pd_lower, pd_upper, grade_pd
PD_GRADES_VIEW = "pd_grades"       # pd_scores + grade (sql/grade_assignment.sql)
GRADES_ARTIFACT_PREFIX = "grades_"  # artifacts/grades_*.csv

# ---------------------------------------------------------------------------
# Stage 8 monitoring  (src/monitoring.py; decision log D-028)
# ---------------------------------------------------------------------------
# Baseline = in-scope development loans. The hold-out stands in for a "next period":
# only its inputs and scores are read, never its outcomes (D-025, D-028).
MONITORING_N_BINS = 10           # equal-count development bins for the score and numeric features
MONITORING_EDGE_DECIMALS = 6     # bin edges are rounded once, like the grade boundaries
MONITORING_SCORE_VARIABLE = "score"  # name of the PD in the monitoring tables
MONITORING_GRADE_VARIABLE = "grade"
MONITORING_NUMERIC_VARIABLES = [MONITORING_SCORE_VARIABLE, "loan_amount", INCOME_CLEAN_COL]
MONITORING_CATEGORICAL_VARIABLES = [MONITORING_GRADE_VARIABLE, *CATEGORICAL_FEATURES]
MISSING_BIN_LABEL = "<missing>"  # every variable has one
UNSEEN_BIN_LABEL = "<unseen>"    # categorical level never seen in development (D-023)
# HEURISTIC thresholds, fixed BEFORE the first monitoring run (D-028). Judgement calls for
# this project, not regulatory standards. (green limit, amber limit), both inclusive.
PSI_EPSILON = 1e-4               # floor on each bin share, so an empty bin gives a finite PSI
PSI_THRESHOLDS = (0.10, 0.25)    # conventional credit-scoring rule of thumb for PSI / CSI
WATCH_TOP_DECILE_FEATURES = ["loan_amount", INCOME_CLEAN_COL]  # D-023 top-decile misfit
WATCH_TOP_DECILE_SHARE = (0.15, 0.20)      # share of loans in the top development decile bin
WATCH_GRADE_A_SHARE_CHANGE = (0.05, 0.10)  # |change| in the best grade's share (D-027)
WATCH_OUT_OF_SCOPE_SHARE = (0.15, 0.20)    # EQUI share of all loans (D-026)
WATCH_RARE_LEVEL = ("lump_sum_payment", "lpsm")  # amber if its share falls below RARE_LEVEL_MIN_SHARE
MONITORING_BASELINE_TABLE = "monitoring_baseline"   # frozen development bins and counts
MONITORING_BIN_COUNTS_VIEW = "monitoring_bin_counts"  # loans per bin and sample (sql/)
MONITORING_ARTIFACT_PREFIX = "monitoring_"  # artifacts/monitoring_*.csv

# ---------------------------------------------------------------------------
# Stage 9 dashboard  (app.py, src/dashboard.py; decision log D-029)
# ---------------------------------------------------------------------------
# A read-only view of the committed tables: nothing is refitted or recomputed. Each
# CSV the dashboard reads (in ARTIFACTS_DIR) -> the columns it uses. The loader checks
# them, and tests/test_dashboard.py checks the committed files against this list.
DASHBOARD_ARTIFACT_COLUMNS = {
    "dq_summary.csv": ["check", "value"],
    "dq_missingness.csv": ["column", "n_missing", "pct_missing",
                           "default_rate_if_missing", "default_rate_if_present"],
    "dq_categorical_levels.csv": ["column", "level", "n"],
    "sql_split_summary.csv": ["sample", "n_loans", "share_loans", "n_defaults", "default_rate"],
    "sql_dq_missingness.csv": ["column_name", "n_missing", "share_missing",
                               "default_rate_if_missing", "default_rate_if_present"],
    "model_cv_metrics.csv": ["model", "fold", "auc", "gini", "ks", "brier"],
    "model_coefficients.csv": ["feature", "coefficient", "odds_ratio", "sm_std_error", "sm_p_value",
                               "expected_sign", "sign_matches_expected", "cv_sign_share"],
    "model_screening.csv": ["feature", "iv_all", "iv_outside_equi", "kept", "reason"],
    "model_sensitivity.csv": ["model", "fold", "auc", "gini", "ks", "brier"],
    "validation_scope.csv": ["sample", "scope", "n_loans", "n_defaults", "observed_rate"],
    "validation_metrics.csv": ["population", "n_loans", "auc", "gini", "ks", "brier"],
    "validation_confidence_intervals.csv": ["metric", "estimate", "ci_lower", "ci_upper", "population"],
    "validation_criteria.csv": ["criterion", "value", "status"],
    "validation_calibration.csv": ["population", "n_loans", "mean_pd", "observed_rate", "binomial_p",
                                   "calibration_intercept", "calibration_slope", "hl_p_value"],
    "validation_calibration_deciles.csv": ["bin", "n_loans", "mean_pd", "observed_rate", "gap"],
    "validation_segments.csv": ["variable", "level", "n_loans", "mean_pd", "observed_rate", "gap", "auc"],
    "validation_feature_deciles.csv": ["variable", "bin", "bin_min", "bin_max", "n_loans", "mean_pd",
                                       "observed_rate", "gap"],
    "validation_coefficient_stability.csv": ["feature", "dev_coefficient", "holdout_coefficient",
                                             "same_sign", "ci_overlap"],
    "grades_scale.csv": ["grade", "pd_lower", "pd_upper", "grade_pd", "n_loans", "share",
                         "observed_rate", "gap", "step_p_value"],
    "grades_checks.csv": ["check", "value", "rule", "status"],
    "grades_oof_check.csv": ["sample", "grade", "n_loans", "mean_pd", "observed_rate"],
    "monitoring_psi.csv": ["variable", "role", "n_bins_used", "psi", "psi_expected_no_shift",
                           "chi2_p_value", "light"],
    "monitoring_psi_bins.csv": ["variable", "bin_order", "bin_label", "share_baseline",
                                "share_monitored", "psi_contribution"],
    "monitoring_watch_list.csv": ["kpi", "decision", "baseline_value", "monitored_value", "rule", "light"],
    "monitoring_characteristic.csv": ["feature", "baseline_contribution", "monitored_contribution",
                                      "change_in_log_odds"],
    "monitoring_grade_backtest.csv": ["sample", "grade", "grade_pd", "n_loans", "observed_rate", "gap",
                                      "binomial_p", "gap_light", "binomial_light"],
    "monitoring_scope.csv": ["sample", "n_loans", "n_graded", "n_out_of_scope_equi", "out_of_scope_share"],
}
# Presentation (D-030). Page key -> sidebar name, in reading order.
DASHBOARD_PAGES = {
    "overview": "1. Overview",
    "leakage": "2. Data leakage",
    "model": "3. Model",
    "validation": "4. Validation",
    "grades": "5. Risk grades",
    "monitoring": "6. Monitoring",
}
# Decision-log entries linked at the end of each page ("Further reading").
DASHBOARD_PAGE_DECISIONS = {
    "overview": ["D-001", "D-004", "D-013", "D-015", "D-026", "D-029", "D-030"],
    "leakage": ["D-011", "D-017", "D-024"],
    "model": ["D-012", "D-022", "D-023"],
    "validation": ["D-004", "D-025", "D-026"],
    "grades": ["D-014", "D-027"],
    "monitoring": ["D-028"],
}
# Links point to the decision log on GitHub's main branch (Bruno, D-030); they resolve
# once this work is merged into main.
REPO_URL = "https://github.com/brunoreichamorim/credit_pd_model"
DOCS_GIT_REF = "main"
DECISION_LOG_PATH = PROJECT_ROOT / "docs" / "decision_log.md"
# Model names as written in model_cv_metrics.csv / model_sensitivity.csv (src/model.py).
CV_MODEL_MAIN = "main"
CV_MODEL_LEAKAGE_FULL = "leakage_full"
CV_MODEL_LEAKAGE_ABLATION = "leakage_ablation_indicators_only"
CV_MEAN_ROW = "mean"
CV_MODEL_LABELS = {
    CV_MODEL_MAIN: "Main model",
    CV_MODEL_LEAKAGE_FULL: "Leakage model",
    CV_MODEL_LEAKAGE_ABLATION: "Missing-value flags only",
}

# Display formats (display only; the CSVs keep full precision).
DISPLAY_METRIC_DECIMALS = 3   # AUC, Gini, KS, Brier
DISPLAY_RATE_DECIMALS = 1     # rates and shares, in %
DISPLAY_SPLIT_DECIMALS = 0    # development / hold-out shares on the Overview pipeline
DISPLAY_PSI_DECIMALS = 4
DISPLAY_TABLE_DECIMALS = 4    # other numbers in detail tables
DISPLAY_P_VALUE_FLOOR = 0.001  # smaller p-values are shown as "< 0.001"
DISPLAY_P_VALUE_DECIMALS = 3
# Columns shown as a percentage (stored as a fraction) and as a whole count.
PERCENT_COLUMNS = {
    "share_missing", "default_rate_if_missing", "default_rate_if_present", "share_loans",
    "default_rate", "observed_rate", "mean_pd", "gap", "pd_min", "pd_max", "pd_lower",
    "pd_upper", "grade_pd", "share", "share_baseline", "share_monitored", "out_of_scope_share",
    "cv_sign_share",
}
COUNT_COLUMNS = {
    "n", "n_missing", "n_loans", "n_defaults", "n_baseline", "n_monitored", "n_graded",
    "n_out_of_scope_equi", "n_in_scope_not_graded", "n_bins_used", "hl_df", "chi2_df",
}
P_VALUE_COLUMNS = {
    "sm_p_value", "binomial_p", "hl_p_value", "step_p_value", "chi2_p_value", "holdout_p_value",
}
# Plain column labels for every table the dashboard shows.
COLUMN_LABELS = {
    "check": "Check", "value": "Value", "rule": "Rule", "status": "Status", "light": "Light",
    "column": "Field", "column_name": "Field", "field": "Field", "feature": "Field",
    "variable": "Variable", "level": "Level", "reason": "Reason", "decision": "Decision",
    "n": "Loans", "pct": "% of loans", "default_rate": "Default rate",
    "n_missing": "Loans missing", "pct_missing": "% missing", "share_missing": "Share missing",
    "default_rate_if_missing": "Default rate if missing",
    "default_rate_if_present": "Default rate if present",
    "sample": "Sample", "scope": "Scope", "population": "Population", "model": "Model",
    "fold": "Fold", "n_loans": "Loans", "n_defaults": "Defaults", "share_loans": "Share of loans",
    "lower": "Lower bound", "median": "Median", "upper": "Upper bound",
    "auc": "AUC", "gini": "Gini", "ks": "KS", "brier": "Brier score",
    "coefficient": "Coefficient", "odds_ratio": "Odds ratio",
    "sm_coefficient": "Coefficient (statsmodels)", "sm_std_error": "Standard error",
    "sm_p_value": "p-value", "expected_sign": "Expected sign",
    "sign_matches_expected": "Sign as expected", "cv_sign_share": "Same sign in CV folds",
    "iv_all": "Information value, all loans", "iv_outside_equi": "Information value, in scope",
    "kept": "Kept",
    "metric": "Metric", "estimate": "Estimate", "ci_lower": "CI lower", "ci_upper": "CI upper",
    "criterion": "Criterion", "mean_pd": "Mean PD", "observed_rate": "Observed default rate",
    "gap": "Gap (observed − PD)", "binomial_p": "Binomial test p-value",
    "calibration_intercept": "Calibration intercept", "calibration_slope": "Calibration slope",
    "hl_statistic": "Hosmer-Lemeshow statistic", "hl_df": "Hosmer-Lemeshow df",
    "hl_p_value": "Hosmer-Lemeshow p-value",
    "bin": "Decile", "pd_min": "Lowest PD", "pd_max": "Highest PD",
    "bin_min": "Lowest value", "bin_max": "Highest value",
    "holdout_coefficient": "Hold-out coefficient", "holdout_std_error": "Hold-out standard error",
    "holdout_p_value": "Hold-out p-value", "dev_coefficient": "Development coefficient",
    "dev_std_error": "Development standard error", "same_sign": "Same sign",
    "ci_overlap": "Confidence intervals overlap",
    "grade": "Grade", "grade_rank": "Grade rank", "pd_lower": "PD from", "pd_upper": "PD up to",
    "grade_pd": "Grade PD", "share": "Share of loans", "step_p_value": "Step test p-value",
    "role": "Role", "n_bins_used": "Bins", "n_baseline": "Loans, development",
    "n_monitored": "Loans, hold-out", "psi": "PSI",
    "psi_expected_no_shift": "PSI expected with no shift", "chi2_statistic": "Chi-square statistic",
    "chi2_df": "Chi-square df", "chi2_p_value": "Chi-square p-value",
    "bin_order": "Bin number", "bin_label": "Bin", "share_baseline": "Share, development",
    "share_monitored": "Share, hold-out", "psi_contribution": "PSI contribution",
    "kpi": "Indicator", "baseline_value": "Development value", "monitored_value": "Hold-out value",
    "baseline_contribution": "Mean contribution, development",
    "monitored_contribution": "Mean contribution, hold-out",
    "change_in_log_odds": "Change in log-odds",
    "n_graded": "Loans graded", "n_out_of_scope_equi": "Out of scope (EQUI)",
    "n_in_scope_not_graded": "In scope, not graded", "out_of_scope_share": "Share out of scope",
    "gap_light": "Gap light", "binomial_light": "Binomial test light",
}
# Traffic lights in tables and tiles: an icon with its word, never colour alone.
LIGHT_LABELS = {"green": "🟢 green", "amber": "🟠 amber", "red": "🔴 red",
                "pass": "🟢 pass", "report": "⚪ reported"}
LIGHT_COLUMNS = {"status", "light", "gap_light", "binomial_light"}
# Headline tiles: id -> short label ({placeholders} are filled from the CSVs).
# Short labels: each side panel has one caption with the shared context
# (dashboard_text.PANEL_CAPTIONS), and each tile's "?" help gives the full meaning.
TILE_LABELS = {
    "cv_auc_main": "Main model",
    "cv_auc_leakage_full": "Leakage model",
    "cv_auc_leakage_ablation": "Missing-value flags",
    "n_features": "Fields",
    "cv_auc": "CV AUC",
    "cv_auc_without": "Without {fields}",
    "holdout_auc": "AUC",
    "holdout_gini": "Gini",
    "holdout_ks": "KS",
    "holdout_brier": "Brier score",
    "holdout_mean_pd": "Mean PD",
    "holdout_observed": "Observed rate",
    "criteria_green": "Criteria green",
    "n_grades": "Grades",
    "first_grade_share": "Share in grade {grade}",
    "top_grade_rate": "Observed rate, grade {grade}",
    "largest_psi": "Largest PSI ({variable})",
    "watch_green": "Watch list green",
    "equi_share_holdout": "EQUI share, hold-out",
}
# Readable names on charts and tiles (display only; tables keep COLUMN_LABELS).
VARIABLE_LABELS = {
    "score": "model score (PD)", "grade": "risk grade",
    "loan_amount": "loan amount", INCOME_CLEAN_COL: "income",
    "lump_sum_payment": "lump-sum payment", "Neg_ammortization": "negative amortisation",
    "loan_type": "loan type", "loan_purpose": "loan purpose",
    "Interest_rate_spread": "interest rate spread", "rate_of_interest": "interest rate",
    "property_value": "property value", "Upfront_charges": "upfront charges",
    "dtir1": "debt-to-income (dtir1)", "age": "age", "credit_type": "credit type",
}
# Model coefficients (model_coefficients.csv `feature`) -> readable name.
COEFFICIENT_LABELS = {
    INCOME_CLEAN_COL: "income (log, standardised)",
    f"{INCOME_CLEAN_COL}_missing": "income missing",
    "loan_amount": "loan amount (log, standardised)",
    "lump_sum_payment_lpsm": "lump-sum payment",
    "Neg_ammortization_neg_amm": "negative amortisation",
    "loan_type_type2": "loan type: type2", "loan_type_type3": "loan type: type3",
    "loan_purpose_p1": "loan purpose: p1", "loan_purpose_p2": "loan purpose: p2",
    "loan_purpose_p3": "loan purpose: p3", "loan_purpose_p4": "loan purpose: p4",
}
# Light as an icon only (pipeline strip) and as a badge colour (Monitoring side panel).
LIGHT_ICONS = {"green": "🟢", "amber": "🟠", "red": "🔴"}
GRADE_OOF_POOLED = "pooled"  # the pooled-folds rows of grades_oof_check.csv (src/grades.py)
LIGHT_BADGE_COLORS = {"green": "green", "amber": "orange", "red": "red"}
LIGHT_BADGE_ICONS = {"green": ":material/check_circle:", "amber": ":material/warning:",
                     "red": ":material/error:"}
# D-029 (HEURISTIC): in-scope development input ranges written by sql/model_input_ranges.sql
# to sql_model_input_ranges.csv. A judgement call for this project, not a standard: about 1%
# of development loans lie beyond each end. The table stays in the pipeline; since the
# scoring page was removed (D-030) the dashboard no longer reads it.
INPUT_RANGE_QUANTILES = (0.01, 0.99)
INPUT_RANGE_VARIABLES = ["loan_amount", INCOME_CLEAN_COL]

# Dashboard theme and chart colours (D-030): a fixed dark theme built only from documented
# values of the dataviz reference palette (dark column). .streamlit/config.toml mirrors
# DASHBOARD_THEME; tests/test_dashboard.py checks that the two agree.
DASHBOARD_THEME = {
    "base": "dark",
    "backgroundColor": "#1a1a19",           # dark chart surface (the surface the palette was validated on)
    "secondaryBackgroundColor": "#2c2c2a",  # dark gridline step
    "textColor": "#ffffff",                 # dark primary ink
    "primaryColor": "#3987e5",              # dark categorical slot 1
    "chartCategoricalColors": ["#3987e5", "#d95926", "#199e70", "#c98500",
                               "#d55181", "#008300", "#9085e9", "#e66767"],
}
DASHBOARD_SIDEBAR_BACKGROUND = "#0d0d0d"    # dark page plane
# Colour roles in the Plotly charts. At most series_1 and series_2 appear together.
DASHBOARD_COLORS = {
    "series_1": "#3987e5",  # main model, grade PD, mean PD, development, raises PD
    "series_2": "#d95926",  # if missing, observed rate, hold-out, lowers PD
    "muted": "#898781",     # reference lines and de-emphasised marks
}
# Chart display settings (display only).
CHART_DIMMED_OPACITY = 0.35   # grades not selected on the grade chart
CHART_MARKER_SIZE = 10        # dots on the calibration and segment charts (dataviz: >= 8 px)
CHART_RATE_AXIS_MAX = 1.05    # x-axis end of 0-1 rate and AUC charts, room for value labels
# Chart heights for a 1920x1080 screen with the browser maximised: title, chart and side
# panel fit above the fold (D-030). Narrower windows stay usable.
DASHBOARD_CHART_HEIGHT = 420
DASHBOARD_HERO_HEIGHT = 340   # Overview hero chart, under the pipeline row
CHART_BAND_OPACITY = 0.18     # light band behind the selected grade
CHART_SELECTED_MARKER_SIZE = 16
AUC_COIN_FLIP = 0.5           # AUC of random ranking (a definition, drawn as a reference line)
GRADE_SLIDER_STEP = 0.1       # grade slider step, in PD percentage points
# Figure 05 fields redrawn on the Data leakage page (from sql_dq_missingness.csv), in the
# figure's order. `income = 0` is not shown: no committed table holds it (D-030).
LEAKAGE_CHART_FIELDS = ["Interest_rate_spread", "rate_of_interest", "property_value",
                        "Upfront_charges", "dtir1", "age"]
# The category shown as one bar next to them (dq_categorical_levels.csv): column, level.
LEAKAGE_CHART_CATEGORY = ("credit_type", EQUI_LEVEL)
LEAKAGE_README_FIGURE = "05_leakage_indicators.png"  # the README figure this chart redraws
# "Where the model misses" views on the Validation page: view -> (label, source).
# Deciles come from validation_feature_deciles.csv, levels from validation_segments.csv;
# both are D-025 pre-set checks on the hold-out.
MISS_VIEWS = {
    "loan_amount": ("Loan amount deciles", "deciles"),
    INCOME_CLEAN_COL: ("Income deciles", "deciles"),
    "loan_purpose": ("Loan purpose", "segments"),
    "loan_type": ("Loan type", "segments"),
}
# Overview pipeline strip: step id -> (label, Material icon).
PIPELINE_STEPS = {
    "raw": ("Raw data", ":material/database:"),
    "split": ("Split", ":material/call_split:"),
    "scope": ("In scope", ":material/filter_alt:"),
    "model": ("Model", ":material/functions:"),
    "validation": ("Validation", ":material/fact_check:"),
    "grades": ("Grades", ":material/stacked_bar_chart:"),
    "monitoring": ("Monitoring", ":material/monitoring:"),
}

# Chart style for static matplotlib figures (light mode).
FIGURE_DPI = 150
COLOR_PRIMARY = "#2a78d6"    # single series / first series
COLOR_SECONDARY = "#eb6834"  # second series
COLOR_INK = "#0b0b0b"
COLOR_INK_MUTED = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"
COLOR_DIVERGING_MID = "#f0efec"  # midpoint of the blue <-> red heatmap scale
COLOR_DIVERGING_NEG = "#2a78d6"
COLOR_DIVERGING_POS = "#e34948"
