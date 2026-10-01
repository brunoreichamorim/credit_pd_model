"""Tests for the Stage 4 SQL / DuckDB layer (src/db.py and sql/*.sql).

Two groups:

* Unit tests on a small synthetic dataset, loaded into a temporary DuckDB file.
  They check the split, the model_dataset view and each SQL analysis against
  hand-computed values or the pandas pipeline, and run anywhere.
* Tests on the real data. They rebuild the database from the raw CSV in a
  temporary folder and are skipped if data/raw/Loan_Default.csv is absent (D-002).
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src import data_processing as dp
from src import db

# ---------------------------------------------------------------------------
# Unit tests on a small synthetic dataset
# ---------------------------------------------------------------------------

N_TOY = 40
TOY_DEFAULT_IDS = [1, 2, 3, 4, 5, 21, 22, 23, 24, 25]  # 10 defaults, 25%


@pytest.fixture
def toy_processed() -> pd.DataFrame:
    """40 processed rows: IDs 1-20 are type1, 21-40 type2; IDs 1-2 are EQUI."""
    ids = np.arange(1, N_TOY + 1)
    raw = pd.DataFrame({col: ["x"] * N_TOY for col in config.EXPECTED_RAW_COLUMNS})
    raw[config.ID_COL] = ids
    raw[config.YEAR_COL] = 2019
    raw[config.TARGET_COL] = np.isin(ids, TOY_DEFAULT_IDS).astype(int)
    for col in ["rate_of_interest", "Interest_rate_spread", "Upfront_charges",
                "Credit_Score", "dtir1", "total_units"]:
        raw[col] = 1.0
    raw["property_value"] = 500_000.0  # valid under D-005, so no cleaned value is missing
    raw["loan_amount"] = ids * 1_000.0
    raw["LTV"] = 100 * raw["loan_amount"] / raw["property_value"]
    raw["income"] = ids * 100.0
    raw["term"] = 360.0
    raw.loc[raw[config.ID_COL] == 40, "term"] = np.nan
    raw["loan_type"] = np.where(ids <= 20, "type1", "type2")
    raw["credit_type"] = np.where(ids <= 2, config.EQUI_LEVEL, "CIB")
    raw.loc[raw[config.ID_COL].isin([7, 30]), "loan_limit"] = np.nan
    return dp.apply_cleaning_rules(raw)


@pytest.fixture
def toy_con(toy_processed, tmp_path):
    """A DuckDB database with the toy data loaded, split and the model view created."""
    parquet = tmp_path / "loans_clean.parquet"
    dp.save_processed_data(toy_processed, parquet)
    con = db.connect(tmp_path / "toy.duckdb")
    db.load_loans(con, parquet)
    db.assign_sample_split(con)
    db.create_model_dataset_view(con)
    db.create_leakage_demo_view(con)
    yield con
    con.close()


def test_load_loans_requires_processed_data(tmp_path):
    con = db.connect(":memory:")
    with pytest.raises(FileNotFoundError, match="data_processing"):
        db.load_loans(con, tmp_path / "missing.parquet")


def test_database_objects_have_configured_names(toy_con):
    tables = {row[0] for row in toy_con.execute("SHOW TABLES").fetchall()}
    assert {config.LOANS_TABLE, config.SPLIT_TABLE, config.MODEL_DATASET_VIEW, config.LEAKAGE_DEMO_VIEW} <= tables


def test_split_is_stratified_complete_and_deterministic(toy_processed):
    ids, target = toy_processed[config.ID_COL], toy_processed[config.TARGET_COL]
    split = db.split_ids(ids, target)

    # every ID exactly once, 30% in the hold-out sample (D-013)
    assert sorted(split[config.ID_COL]) == sorted(ids)
    holdout = split.loc[split["sample"] == config.SAMPLE_HOLDOUT, config.ID_COL]
    assert len(holdout) == round(config.TEST_SIZE * N_TOY)
    # stratified: 25% defaults in the hold-out sample as in the whole dataset
    assert target[ids.isin(holdout)].mean() == pytest.approx(0.25)

    # the same split again, even if the rows arrive in a different order
    shuffled = toy_processed.sample(frac=1, random_state=0)
    again = db.split_ids(shuffled[config.ID_COL], shuffled[config.TARGET_COL])
    pd.testing.assert_frame_equal(split, again)


def test_model_dataset_contains_only_admissible_features(toy_con):
    view = toy_con.execute(f"SELECT * FROM {config.MODEL_DATASET_VIEW}").df()
    expected = [config.ID_COL, "sample", config.TARGET_COL] + config.MAIN_MODEL_CANDIDATE_FEATURES
    assert list(view.columns) == expected
    assert not set(view.columns) & set(config.EXCLUDED_FROM_MAIN_MODEL.keys() - {config.ID_COL})
    assert len(view) == N_TOY
    assert set(view["sample"]) == {config.SAMPLE_DEVELOPMENT, config.SAMPLE_HOLDOUT}


def test_model_scope_view_leaves_out_exactly_the_equi_loans(toy_con):
    # D-026: IDs 1-2 are EQUI in the toy data; the scope view drops them and nothing else
    scope = toy_con.execute(f"SELECT * FROM {config.MODEL_SCOPE_VIEW}").df()
    full = toy_con.execute(f"SELECT * FROM {config.MODEL_DATASET_VIEW}").df()
    assert list(scope.columns) == list(full.columns)  # credit_type is not added as a column
    assert set(full[config.ID_COL]) - set(scope[config.ID_COL]) == {1, 2}
    assert len(scope) == N_TOY - 2
    # the SQL file hard-codes the level; it must match config
    assert f"'{config.EQUI_LEVEL}'" in db.read_sql("model_scope_dataset")


def test_leakage_demo_view_matches_config(toy_con):
    # D-024: the SQL view lists its columns literally; they must match config exactly
    view = toy_con.execute(f"SELECT * FROM {config.LEAKAGE_DEMO_VIEW}").df()
    expected = ([config.ID_COL, "sample", config.TARGET_COL]
                + config.MAIN_MODEL_FEATURES + config.LEAKAGE_DEMO_EXTRA_FEATURES)
    assert list(view.columns) == expected


def test_sql_missingness_matches_pandas(toy_con, toy_processed):
    # the same table computed two ways must agree
    sql = db.run_query(toy_con, "dq_missingness").set_index("column_name")
    pandas = dp.missingness_table(toy_processed).set_index("column")
    assert set(sql.index) == set(pandas.index) == {"loan_limit", "term"}
    for col in pandas.index:
        assert sql.loc[col, "n_missing"] == pandas.loc[col, "n_missing"]
        assert 100 * sql.loc[col, "share_missing"] == pytest.approx(pandas.loc[col, "pct_missing"], abs=0.01)
        for rate in ["default_rate_if_missing", "default_rate_if_present"]:
            assert sql.loc[col, rate] == pytest.approx(pandas.loc[col, rate], abs=1e-4)


def test_reconciliation_counts(toy_con):
    row = db.run_query(toy_con, "dq_reconciliation").iloc[0]
    assert row["n_rows"] == row["n_distinct_ids"] == N_TOY
    assert row["n_defaults"] == len(TOY_DEFAULT_IDS)
    assert row["default_rate"] == pytest.approx(0.25)


def test_portfolio_count_and_amount_weighted_default_rates(toy_con):
    table = db.run_query(toy_con, "portfolio_by_segment", min_segment_size=1)
    assert set(table["segment_column"]) == set(config.PORTFOLIO_SEGMENT_COLUMNS)

    type1 = table.set_index(["segment_column", "segment_level"]).loc[("loan_type", "type1")]
    assert type1["n_loans"] == 20
    assert type1["exposure"] == 210_000        # 1,000 + 2,000 + ... + 20,000
    assert type1["default_rate"] == pytest.approx(5 / 20)
    # defaulted IDs 1-5 hold 15,000 of the 210,000 lent: small loans default here
    assert type1["default_rate_amount_weighted"] == pytest.approx(15 / 210)

    term = table[table["segment_column"] == "term"].set_index("segment_level")
    assert term.loc["<missing>", "n_loans"] == 1  # missing values form their own level
    assert term["share_loans"].sum() == pytest.approx(1.0)


def test_small_segments_are_hidden_without_changing_shown_rows(toy_con):
    # D-021: the single loan with a missing term is hidden; the 360 row keeps its share
    shown = db.run_query(toy_con, "portfolio_by_segment", min_segment_size=10)
    term = shown[shown["segment_column"] == "term"].set_index("segment_level")
    assert list(term.index) == ["360.0"]
    assert term.loc["360.0", "share_loans"] == pytest.approx(39 / 40)
    assert (shown["n_loans"] >= 10).all()

    crosses = db.run_query(toy_con, "risk_segment_crosses", equi=config.EQUI_LEVEL,
                           min_segment_size=19)
    assert (crosses["n_loans"] >= 19).all()


def test_risk_deciles_bins_and_outside_equi_population(toy_con):
    table = db.run_query(toy_con, "risk_deciles", n_bins=4, equi=config.EQUI_LEVEL)
    table = table.set_index(["population", "variable", "bin"])

    first = table.loc[("all", "loan_amount", 1)]  # IDs 1-10: defaults 1-5
    assert first["n_loans"] == 10
    assert first["default_rate"] == pytest.approx(0.5)
    assert (first["bin_min"], first["bin_max"]) == (1_000, 10_000)

    # IDs 1-2 (EQUI) are left out: 38 loans, first bin is IDs 3-12 with 3 defaults
    outside = table.loc[("outside_EQUI", "loan_amount")]
    assert outside["n_loans"].sum() == N_TOY - 2
    assert outside.loc[1, "default_rate"] == pytest.approx(3 / 10)


def test_segment_crosses_match_config(toy_con):
    table = db.run_query(toy_con, "risk_segment_crosses", equi=config.EQUI_LEVEL,
                         min_segment_size=1)
    crosses = set(zip(table["feature_1"], table["feature_2"]))
    assert crosses == set(config.SEGMENT_CROSSES)
    for population in ["all", "outside_EQUI"]:
        per_cross = table[table["population"] == population].groupby(["feature_1", "feature_2"])["n_loans"].sum()
        assert (per_cross == (N_TOY if population == "all" else N_TOY - 2)).all()


def test_split_summary_reconciles_with_split(toy_con):
    summary = db.run_query(toy_con, "split_summary").set_index("sample")
    assert summary["n_loans"].sum() == N_TOY
    assert summary.loc[config.SAMPLE_HOLDOUT, "n_loans"] == 12
    assert summary["n_defaults"].sum() == len(TOY_DEFAULT_IDS)


def test_build_database_writes_aggregate_tables(toy_processed, tmp_path):
    parquet = tmp_path / "loans_clean.parquet"
    dp.save_processed_data(toy_processed, parquet)
    tables = db.build_database(tmp_path / "toy.duckdb", parquet, tmp_path, verbose=False)
    for name in db.ANALYSIS_QUERIES:
        path = tmp_path / f"{config.SQL_ARTIFACT_PREFIX}{name}.csv"
        assert path.exists()
        # aggregates only: no borrower ID column in any saved table
        assert config.ID_COL not in pd.read_csv(path).columns
    assert set(tables) == set(db.ANALYSIS_QUERIES)


# ---------------------------------------------------------------------------
# Tests on the real data (skipped if the raw file has not been downloaded)
# ---------------------------------------------------------------------------

needs_raw_data = pytest.mark.skipif(
    not config.RAW_DATA_PATH.exists(),
    reason="data/raw/Loan_Default.csv not present (it is not committed, see D-002)",
)


@pytest.fixture(scope="module")
def real_build(tmp_path_factory):
    """Run the Stage 2 pipeline and the Stage 4 build into a temporary folder."""
    folder = tmp_path_factory.mktemp("real")
    parquet = folder / "loans_clean.parquet"
    dp.run_pipeline(output_path=parquet, report_dir=folder, verbose=False)
    tables = db.build_database(folder / "credit_risk.duckdb", parquet, folder, verbose=False)
    return folder, parquet, tables


@needs_raw_data
def test_real_reconciliation(real_build):
    _, _, tables = real_build
    row = tables["dq_reconciliation"].iloc[0]
    assert row["n_rows"] == row["n_distinct_ids"] == config.EXPECTED_N_ROWS
    assert row["n_defaults"] == 36_639


@needs_raw_data
def test_real_split_sizes_and_default_rates(real_build):
    # D-013: 70% development / 30% hold-out, stratified on Status
    _, _, tables = real_build
    summary = tables["split_summary"].set_index("sample")
    assert summary.loc[config.SAMPLE_DEVELOPMENT, "n_loans"] == 104_069
    assert summary.loc[config.SAMPLE_HOLDOUT, "n_loans"] == 44_601
    rates = summary["default_rate"]
    assert rates.max() - rates.min() < 1e-4


@needs_raw_data
def test_real_split_is_identical_on_rebuild(real_build, tmp_path):
    folder, parquet, first_tables = real_build
    first = db.connect(folder / "credit_risk.duckdb")
    second_path = tmp_path / "again.duckdb"
    second_tables = db.build_database(second_path, parquet, tmp_path, verbose=False)
    second = db.connect(second_path)
    query = f"SELECT * FROM {config.SPLIT_TABLE} ORDER BY {config.ID_COL}"
    pd.testing.assert_frame_equal(first.execute(query).df(), second.execute(query).df())
    first.close()
    second.close()

    # risk_deciles uses NTILE on a column with many tied values (D-020): its ORDER BY
    # must break ties on ID, or which rows land in which bin depends on execution order
    pd.testing.assert_frame_equal(first_tables["risk_deciles"], second_tables["risk_deciles"])


@needs_raw_data
def test_real_segment_tables_have_no_small_segments(real_build):
    # D-021: no committed segment row describes fewer than SQL_MIN_SEGMENT_SIZE loans
    _, _, tables = real_build
    for name in ["portfolio_by_segment", "risk_segment_crosses", "risk_deciles"]:
        assert (tables[name]["n_loans"] >= config.SQL_MIN_SEGMENT_SIZE).all(), name


@needs_raw_data
def test_real_sql_missingness_matches_pipeline_table(real_build):
    # the D-011 / D-017 evidence, reproduced in SQL
    folder, _, tables = real_build
    sql = tables["dq_missingness"].set_index("column_name")
    pandas = pd.read_csv(folder / "dq_missingness.csv").set_index("column")  # raw columns
    for col in pandas.index.drop(config.YEAR_COL, errors="ignore"):
        assert sql.loc[col, "n_missing"] == pandas.loc[col, "n_missing"]
        for rate in ["default_rate_if_missing", "default_rate_if_present"]:
            assert sql.loc[col, rate] == pytest.approx(pandas.loc[col, rate], abs=1e-4)
    assert sql.loc["Interest_rate_spread", "n_missing"] == 36_639
    assert sql.loc["Interest_rate_spread", "default_rate_if_missing"] == 1.0
    assert sql.loc["Interest_rate_spread", "default_rate_if_present"] == 0.0
