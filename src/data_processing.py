"""Stage 2 data pipeline: raw CSV -> validated, profiled, cleaned Parquet.

Steps (each is a separate function so it can be tested on its own):

    load_raw_data        read data/raw/Loan_Default.csv (the file is never modified)
    validate_schema      fail loudly if the file is not the documented dataset
    profile_data         aggregate data-quality tables (missingness, categories, numerics)
    find_* functions     row-level evidence for the known data-quality issues
    apply_cleaning_rules the documented rules D-004, D-005, D-006, D-009 -- nothing else
    save_processed_data  write data/processed/loans_clean.parquet

Cleaning principles (see docs/decision_log.md):
    * rows are never deleted;
    * raw columns are never overwritten -- cleaned versions are added as new columns;
    * no capping, imputation, category merging or feature selection happens here.
      Those are modelling choices and belong to later stages.

Run from the project root:

    python -m src.data_processing
"""

from pathlib import Path

import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Load and validate
# ---------------------------------------------------------------------------


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as provided."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Download Loan_Default.csv from Kaggle "
            "and place it there (see README, decision D-002)."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the data is not the documented dataset (D-001, D-003, D-015).

    Checks the column names and order, the ID key and the target values.
    Anything softer (missing values, odd values) is profiled, not rejected.
    """
    columns = list(df.columns)
    if columns != config.EXPECTED_RAW_COLUMNS:
        missing = [c for c in config.EXPECTED_RAW_COLUMNS if c not in columns]
        extra = [c for c in columns if c not in config.EXPECTED_RAW_COLUMNS]
        raise ValueError(
            f"Raw columns do not match the expected schema. Missing: {missing}. "
            f"Unexpected: {extra}. (If both are empty, the column order differs.)"
        )

    ids = df[config.ID_COL]
    if ids.isna().any():
        raise ValueError(f"{config.ID_COL} has {ids.isna().sum()} missing values.")
    if not ids.is_unique:
        raise ValueError(f"{config.ID_COL} has {ids.duplicated().sum()} duplicated values.")

    target = df[config.TARGET_COL]
    if target.isna().any():
        raise ValueError(f"{config.TARGET_COL} has {target.isna().sum()} missing values.")
    unexpected = set(target.unique()) - set(config.TARGET_VALUES)
    if unexpected:
        raise ValueError(f"{config.TARGET_COL} contains unexpected values: {sorted(unexpected)}.")


# ---------------------------------------------------------------------------
# Profiling (aggregate tables only -- no raw records are written to disk)
# ---------------------------------------------------------------------------


def summarise_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Headline facts: shape, key uniqueness, duplicates, target and year distribution."""
    target = df[config.TARGET_COL]
    facts = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "schema_matches_expected": list(df.columns) == config.EXPECTED_RAW_COLUMNS,
        "id_missing": int(df[config.ID_COL].isna().sum()),
        "id_unique": bool(df[config.ID_COL].is_unique),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_ignoring_id": int(df.drop(columns=config.ID_COL).duplicated().sum()),
        "status_0": int((target == 0).sum()),
        "status_1": int((target == 1).sum()),
        "status_missing": int(target.isna().sum()),
        "default_rate": round(float(target.mean()), 6),
        "year_values": ", ".join(
            f"{year}: {n}" for year, n in df[config.YEAR_COL].value_counts(dropna=False).items()
        ),
    }
    return pd.DataFrame({"check": list(facts), "value": [str(v) for v in facts.values()]})


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and share per column, plus the default rate with / without a value.

    The two default-rate columns are what D-011 asks for: they show whether
    *being missing* is associated with Status. They say nothing about why.
    """
    rows = []
    for col in df.columns:
        is_missing = df[col].isna()
        if not is_missing.any():
            continue
        rows.append({
            "column": col,
            "n_missing": int(is_missing.sum()),
            "pct_missing": round(100 * is_missing.mean(), 2),
            "default_rate_if_missing": round(df.loc[is_missing, config.TARGET_COL].mean(), 4),
            "default_rate_if_present": round(df.loc[~is_missing, config.TARGET_COL].mean(), 4),
        })
    return pd.DataFrame(rows).sort_values("n_missing", ascending=False, ignore_index=True)


def categorical_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Every level of every text column (missing shown as '<missing>'), with its default rate."""
    rows = []
    for col in df.select_dtypes(exclude="number").columns:
        values = df[col].fillna("<missing>")
        grouped = df.groupby(values)[config.TARGET_COL].agg(["size", "mean"])
        for level, (n, default_rate) in grouped.iterrows():
            rows.append({
                "column": col,
                "level": level,
                "n": int(n),
                "pct": round(100 * n / len(df), 2),
                "default_rate": round(default_rate, 4),
            })
    return pd.DataFrame(rows)


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of every numeric column, with tail quantiles to expose extremes."""
    numeric = df.select_dtypes(include="number").drop(columns=[config.ID_COL, config.TARGET_COL])
    profile = numeric.describe(percentiles=[0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]).T
    profile.insert(0, "n_missing", numeric.isna().sum())
    profile.insert(1, "n_zero", (numeric == 0).sum())
    profile.insert(2, "n_negative", (numeric < 0).sum())
    return profile.round(4).rename_axis("column").reset_index()


def profile_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All aggregate data-quality tables, keyed by the name they are saved under."""
    return {
        "dq_summary": summarise_dataset(df),
        "dq_missingness": missingness_table(df),
        "dq_categorical_levels": categorical_levels(df),
        "dq_numeric_profile": numeric_profile(df),
    }


# ---------------------------------------------------------------------------
# Known-issue investigation (row-level evidence, printed but not saved)
# ---------------------------------------------------------------------------

_EVIDENCE_COLS = [
    config.ID_COL, "loan_amount", "property_value", "LTV",
    "rate_of_interest", "Interest_rate_spread", "income", config.TARGET_COL,
]


def find_nonpositive_interest_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Rows affected by D-009 (rate_of_interest <= 0)."""
    mask = df["rate_of_interest"] <= config.RATE_OF_INTEREST_MIN_EXCLUSIVE
    return df.loc[mask, _EVIDENCE_COLS]


def find_implausible_property_values(df: pd.DataFrame) -> pd.DataFrame:
    """Rows affected by D-005 (property_value below the investigation heuristic)."""
    mask = df["property_value"] < config.PROPERTY_VALUE_MIN_VALID
    return df.loc[mask, _EVIDENCE_COLS].sort_values("LTV", ascending=False)


def ltv_matches_components(df: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    """True if the raw LTV equals 100 * loan_amount / property_value wherever both exist."""
    recomputed = 100 * df["loan_amount"] / df["property_value"]
    both = recomputed.notna() & df["LTV"].notna()
    same_missingness = (df["LTV"].isna() == df["property_value"].isna()).all()
    return bool(same_missingness and ((recomputed[both] - df.loc[both, "LTV"]).abs() <= tolerance).all())


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def apply_cleaning_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply only the documented cleaning rules and return a new DataFrame.

    D-004  year is constant, so it carries no information and is dropped.
           If it is ever NOT constant, stop: that would reopen out-of-time validation.
    D-005  property_value < PROPERTY_VALUE_MIN_VALID -> property_value_clean = NaN (HEURISTIC)
    D-009  rate_of_interest <= 0                     -> rate_of_interest_clean = NaN (ASSUMPTION)
    D-006  LTV_clean = 100 * loan_amount / property_value_clean
    D-008  income <= 0                               -> income_clean = NaN (MODELLING CHOICE)

    The raw columns stay untouched, so every rule can be audited or reversed later.
    The row count never changes.
    """
    if df[config.YEAR_COL].nunique(dropna=False) != 1:
        raise ValueError(
            f"'{config.YEAR_COL}' is not constant. D-004 assumed it was; revisit that "
            "decision (and the possibility of out-of-time validation) before cleaning."
        )
    clean = df.drop(columns=config.YEAR_COL)

    property_value = clean["property_value"]
    clean[config.PROPERTY_VALUE_CLEAN_COL] = property_value.where(
        property_value >= config.PROPERTY_VALUE_MIN_VALID
    )

    rate = clean["rate_of_interest"]
    clean[config.RATE_OF_INTEREST_CLEAN_COL] = rate.where(
        rate > config.RATE_OF_INTEREST_MIN_EXCLUSIVE
    )

    clean[config.LTV_CLEAN_COL] = 100 * clean["loan_amount"] / clean[config.PROPERTY_VALUE_CLEAN_COL]

    income = clean["income"]
    clean[config.INCOME_CLEAN_COL] = income.where(income > config.INCOME_MIN_EXCLUSIVE)

    return clean


def processed_columns() -> list[str]:
    """The schema of the processed dataset: raw columns minus year, plus the cleaned columns."""
    raw = [c for c in config.EXPECTED_RAW_COLUMNS if c != config.YEAR_COL]
    return raw + [
        config.PROPERTY_VALUE_CLEAN_COL, config.RATE_OF_INTEREST_CLEAN_COL,
        config.LTV_CLEAN_COL, config.INCOME_CLEAN_COL,
    ]


# ---------------------------------------------------------------------------
# Save and run
# ---------------------------------------------------------------------------


def save_processed_data(df: pd.DataFrame, path: Path = config.CLEAN_DATA_PATH) -> Path:
    """Write the processed dataset to Parquet (keeps dtypes, unlike CSV)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def save_profile_tables(tables: dict[str, pd.DataFrame], directory: Path = config.DQ_REPORT_DIR) -> None:
    """Write the aggregate data-quality tables as small CSVs (committed as evidence)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)


def run_pipeline(
    raw_path: Path = config.RAW_DATA_PATH,
    output_path: Path = config.CLEAN_DATA_PATH,
    report_dir: Path = config.DQ_REPORT_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load -> validate -> profile -> clean -> save. Returns the processed DataFrame."""
    raw = load_raw_data(raw_path)
    validate_schema(raw)

    tables = profile_data(raw)
    save_profile_tables(tables, report_dir)

    clean = apply_cleaning_rules(raw)
    save_processed_data(clean, output_path)

    if verbose:
        _print_report(raw, clean, tables, output_path, report_dir)
    return clean


def _print_report(raw, clean, tables, output_path, report_dir) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("=== Dataset summary ===")
        print(tables["dq_summary"].to_string(index=False))
        print("\n=== Missingness vs Status ===")
        print(tables["dq_missingness"].to_string(index=False))
        print(f"\n=== D-009: rate_of_interest <= {config.RATE_OF_INTEREST_MIN_EXCLUSIVE} ===")
        print(find_nonpositive_interest_rates(raw).to_string(index=False))
        print(f"\n=== D-005: property_value < {config.PROPERTY_VALUE_MIN_VALID:,} ===")
        print(find_implausible_property_values(raw).to_string(index=False))
        print(f"\nRaw LTV == 100 * loan_amount / property_value: {ltv_matches_components(raw)}")
        print(f"\nProcessed: {clean.shape[0]:,} rows x {clean.shape[1]} columns -> {output_path}")
        print(f"Data-quality tables -> {report_dir}")


if __name__ == "__main__":
    run_pipeline()
