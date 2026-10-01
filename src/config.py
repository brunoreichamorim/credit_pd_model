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
# once for final evaluation. Stratified random split on Status: this is
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
SAMPLE_HOLDOUT = "holdout"          # 30%: final evaluation, used once

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
GRADE_SCALE_TABLE = "grade_scale"  # grade, grade_rank, pd_lower, pd_upper, grade_pd
PD_GRADES_VIEW = "pd_grades"       # pd_scores + grade (sql/grade_assignment.sql)
GRADES_ARTIFACT_PREFIX = "grades_"  # artifacts/grades_*.csv

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
