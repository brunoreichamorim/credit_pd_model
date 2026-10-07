"""Tests for the Stage 2 data pipeline (src/data_processing.py).

Two groups:

* Unit tests on a tiny hand-made DataFrame. They check the logic of each rule
  and run anywhere, including on GitHub where the raw CSV is not available.
* Tests on the real raw file. They reproduce the dataset facts recorded in
  docs/decision_log.md (D-001, D-003, D-004, D-009, D-015) and are skipped
  automatically if data/raw/Loan_Default.csv has not been downloaded.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src import data_processing as dp

# ---------------------------------------------------------------------------
# Unit tests on a small synthetic dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_raw() -> pd.DataFrame:
    """Four rows with the exact raw schema, covering each cleaning rule."""
    n = 4
    df = pd.DataFrame({col: ["x"] * n for col in config.EXPECTED_RAW_COLUMNS})
    df[config.ID_COL] = [1, 2, 3, 4]
    df[config.YEAR_COL] = 2019
    df[config.TARGET_COL] = [0, 1, 0, 1]
    df["loan_amount"] = [200_000, 400_000, 100_000, 300_000]
    df["property_value"] = [250_000, 8_000, np.nan, 10_000]    # row 2 below threshold, row 4 at it
    df["LTV"] = 100 * df["loan_amount"] / df["property_value"]
    df["rate_of_interest"] = [4.0, 0.0, np.nan, -1.0]           # rows 2 and 4 non-positive
    df["income"] = [5_000.0, 0.0, np.nan, 600.0]                # row 2 zero
    return df


def test_validate_schema_accepts_valid_data(toy_raw):
    dp.validate_schema(toy_raw)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda df: df.drop(columns="LTV"),                                   # missing column
        lambda df: df[list(reversed(df.columns))],                           # wrong order
        lambda df: df.assign(**{config.ID_COL: [1, 1, 3, 4]}),               # duplicate ID
        lambda df: df.assign(**{config.ID_COL: [1, np.nan, 3, 4]}),          # missing ID
        lambda df: df.assign(**{config.TARGET_COL: [0, 1, 2, 1]}),           # unexpected target
        lambda df: df.assign(**{config.TARGET_COL: [0, 1, np.nan, 1]}),      # missing target
    ],
)
def test_validate_schema_rejects_invalid_data(toy_raw, corrupt):
    with pytest.raises(ValueError):
        dp.validate_schema(corrupt(toy_raw))


def test_cleaning_sets_nonpositive_interest_rate_to_missing(toy_raw):
    # D-009: <= 0 becomes NaN; valid and already-missing values are unchanged
    clean = dp.apply_cleaning_rules(toy_raw)
    rate = clean[config.RATE_OF_INTEREST_CLEAN_COL]
    assert rate.iloc[0] == 4.0
    assert rate.iloc[1:].isna().all()


def test_cleaning_sets_implausible_property_value_to_missing(toy_raw):
    # D-005: strictly below the threshold becomes NaN; the threshold itself is kept
    clean = dp.apply_cleaning_rules(toy_raw)
    pv = clean[config.PROPERTY_VALUE_CLEAN_COL]
    assert pv.iloc[0] == 250_000
    assert np.isnan(pv.iloc[1]) and np.isnan(pv.iloc[2])
    assert pv.iloc[3] == config.PROPERTY_VALUE_MIN_VALID


def test_ltv_clean_is_recomputed_from_clean_property_value(toy_raw):
    # D-006
    clean = dp.apply_cleaning_rules(toy_raw)
    ltv = clean[config.LTV_CLEAN_COL]
    assert ltv.iloc[0] == pytest.approx(80.0)
    assert np.isnan(ltv.iloc[1])  # would have been 5,000% with the raw value
    assert ltv.iloc[3] == pytest.approx(3_000.0)


def test_cleaning_sets_zero_income_to_missing(toy_raw):
    # D-008: income <= 0 becomes NaN; positive and already-missing values are unchanged
    clean = dp.apply_cleaning_rules(toy_raw)
    income = clean[config.INCOME_CLEAN_COL]
    assert income.iloc[0] == 5_000.0 and income.iloc[3] == 600.0
    assert income.iloc[1:3].isna().all()


def test_cleaning_keeps_rows_and_raw_columns(toy_raw):
    original = toy_raw.copy()
    clean = dp.apply_cleaning_rules(toy_raw)
    assert len(clean) == len(toy_raw)
    pd.testing.assert_frame_equal(toy_raw, original)  # input not mutated
    for col in ["property_value", "rate_of_interest", "LTV", "income"]:
        pd.testing.assert_series_equal(clean[col], original[col])


def test_cleaning_output_schema(toy_raw):
    clean = dp.apply_cleaning_rules(toy_raw)
    assert list(clean.columns) == dp.processed_columns()
    assert config.YEAR_COL not in clean.columns  # D-004


def test_cleaning_stops_if_year_is_not_constant(toy_raw):
    toy_raw.loc[0, config.YEAR_COL] = 2018
    with pytest.raises(ValueError, match="not constant"):
        dp.apply_cleaning_rules(toy_raw)


def test_missingness_table_reports_default_rates(toy_raw):
    table = dp.missingness_table(toy_raw).set_index("column")
    row = table.loc["rate_of_interest"]  # only row 3 (Status 0) is missing
    assert row["n_missing"] == 1
    assert row["pct_missing"] == 25.0
    assert row["default_rate_if_missing"] == 0.0
    assert row["default_rate_if_present"] == pytest.approx(2 / 3, abs=1e-4)


def test_ltv_matches_components(toy_raw):
    assert dp.ltv_matches_components(toy_raw)
    toy_raw.loc[0, "LTV"] = 1.0
    assert not dp.ltv_matches_components(toy_raw)


# ---------------------------------------------------------------------------
# Tests on the real raw file (skipped if it has not been downloaded)
# ---------------------------------------------------------------------------

needs_raw_data = pytest.mark.skipif(
    not config.RAW_DATA_PATH.exists(),
    reason="data/raw/Loan_Default.csv not present (it is not committed, see D-002)",
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return dp.load_raw_data()


@needs_raw_data
def test_raw_schema_and_size(raw):
    # D-001
    dp.validate_schema(raw)
    assert list(raw.columns) == config.EXPECTED_RAW_COLUMNS
    assert raw.shape == (config.EXPECTED_N_ROWS, 34)


@needs_raw_data
def test_raw_id_is_unique_and_complete(raw):
    # D-003
    assert raw[config.ID_COL].notna().all()
    assert raw[config.ID_COL].is_unique


@needs_raw_data
def test_raw_has_no_duplicate_rows(raw):
    assert not raw.duplicated().any()
    assert not raw.drop(columns=config.ID_COL).duplicated().any()


@needs_raw_data
def test_raw_target_values(raw):
    # D-015: binary and complete; its business definition is NOT established here
    target = raw[config.TARGET_COL]
    assert target.notna().all()
    assert set(target.unique()) == set(config.TARGET_VALUES)


@needs_raw_data
def test_raw_year_is_constant(raw):
    # D-004: no time dimension, so no out-of-time validation is possible
    assert set(raw[config.YEAR_COL].unique()) == {2019}


@needs_raw_data
def test_raw_ltv_is_derived_from_loan_and_property_value(raw):
    # D-006: justifies recomputing LTV from its components
    assert dp.ltv_matches_components(raw)


@needs_raw_data
def test_pipeline_creates_loadable_processed_dataset(tmp_path):
    output = tmp_path / "loans_clean.parquet"
    clean = dp.run_pipeline(output_path=output, report_dir=tmp_path, verbose=False)

    reloaded = pd.read_parquet(output)
    assert list(reloaded.columns) == dp.processed_columns()
    assert len(reloaded) == config.EXPECTED_N_ROWS
    pd.testing.assert_frame_equal(reloaded, clean, check_dtype=False)

    # the documented rules affect exactly the rows they target
    raw = dp.load_raw_data()
    newly_missing_rate = reloaded[config.RATE_OF_INTEREST_CLEAN_COL].isna() & raw["rate_of_interest"].notna()
    assert newly_missing_rate.sum() == (raw["rate_of_interest"] <= 0).sum()
    newly_missing_pv = reloaded[config.PROPERTY_VALUE_CLEAN_COL].isna() & raw["property_value"].notna()
    assert newly_missing_pv.sum() == (raw["property_value"] < config.PROPERTY_VALUE_MIN_VALID).sum()
    newly_missing_income = reloaded[config.INCOME_CLEAN_COL].isna() & raw["income"].notna()
    assert newly_missing_income.sum() == (raw["income"] <= config.INCOME_MIN_EXCLUSIVE).sum() == 1_260

    for name in ["dq_summary", "dq_missingness", "dq_categorical_levels", "dq_numeric_profile"]:
        assert (tmp_path / f"{name}.csv").exists()
