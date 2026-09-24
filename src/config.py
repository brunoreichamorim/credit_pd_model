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

# D-011: pricing variables whose missingness pattern must be investigated
# against Status before any modelling decision is made about them.
PRICING_COLUMNS_UNDER_INVESTIGATION = [
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges",
]
