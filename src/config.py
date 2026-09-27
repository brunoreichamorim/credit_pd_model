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
