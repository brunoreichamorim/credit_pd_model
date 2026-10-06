"""Tests for the Stage 8 monitoring (src/monitoring.py, sql/monitoring_*.sql).

Unit tests use small synthetic data, and the toy database from tests/test_model.py
(through the Stage 6 and Stage 7 fixtures). Tests on the real data skip automatically if
data/raw/Loan_Default.csv or the Stage 4-7 outputs are absent (D-002).
"""

import re

import numpy as np
import pandas as pd
import pytest

from src import config
from src import db
from src import grades
from src import monitoring
from tests.test_grades import toy_stage7  # noqa: F401  (pytest fixture)
from tests.test_holdout import toy_stage6  # noqa: F401  (pytest fixture)
from tests.test_model import toy_con, toy_processed  # noqa: F401  (pytest fixtures)

# ---------------------------------------------------------------------------
# PSI on synthetic data
# ---------------------------------------------------------------------------


def test_psi_is_zero_for_identical_distributions():
    shares = np.array([0.2, 0.3, 0.5])
    assert monitoring.psi_contributions(shares, shares).sum() == pytest.approx(0.0)


def test_psi_matches_a_hand_computed_value():
    expected, actual = np.array([0.5, 0.5]), np.array([0.6, 0.4])
    hand = (0.6 - 0.5) * np.log(0.6 / 0.5) + (0.4 - 0.5) * np.log(0.4 / 0.5)
    assert monitoring.psi_contributions(expected, actual).sum() == pytest.approx(hand)


def test_psi_contributions_are_never_negative_and_psi_is_symmetric():
    e, a = np.array([0.1, 0.2, 0.7]), np.array([0.3, 0.3, 0.4])
    assert (monitoring.psi_contributions(e, a) >= 0).all()
    assert monitoring.psi_contributions(e, a).sum() == pytest.approx(monitoring.psi_contributions(a, e).sum())


def test_an_empty_bin_gives_a_large_but_finite_psi():
    value = monitoring.psi_contributions([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]).sum()
    assert np.isfinite(value) and value > 0.25


@pytest.mark.parametrize("value, light", [(0.0, "green"), (0.10, "green"), (0.1001, "amber"),
                                          (0.25, "amber"), (0.2501, "red")])
def test_psi_light_limits_are_inclusive(value, light):
    assert monitoring.psi_light(value) == light


def test_a_shifted_population_turns_amber_then_red():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=20_000))
    bins = monitoring.numeric_bins("x", base)
    same = monitoring.psi_from_values(base, pd.Series(rng.normal(size=20_000)), bins)
    medium = monitoring.psi_from_values(base, pd.Series(rng.normal(0.45, 1, size=20_000)), bins)
    large = monitoring.psi_from_values(base, pd.Series(rng.normal(1.0, 1, size=20_000)), bins)
    assert monitoring.psi_light(same) == "green" and same < 0.01
    assert monitoring.psi_light(medium) == "amber"
    assert monitoring.psi_light(large) == "red"


def test_chi2_reference_matches_its_formula_and_flags_a_real_shift():
    ref = monitoring.psi_chi2_reference(0.0003, 90_000, 40_000, 10)
    scale = 1 / 90_000 + 1 / 40_000
    assert ref["psi_expected_no_shift"] == pytest.approx(9 * scale)
    assert ref["chi2_statistic"] == pytest.approx(0.0003 / scale)
    assert ref["chi2_df"] == 9
    assert monitoring.psi_chi2_reference(0.01, 90_000, 40_000, 10)["chi2_p_value"] < 1e-6


def test_chi2_reference_is_roughly_calibrated_with_no_shift():
    # two samples from the same distribution: p-values should not pile up near 0
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(size=5_000))
    bins = monitoring.numeric_bins("x", base)
    p_values = []
    for _ in range(200):
        mon = pd.Series(rng.normal(size=2_000))
        value = monitoring.psi_from_values(base, mon, bins)
        p_values.append(monitoring.psi_chi2_reference(value, 5_000, 2_000, len(bins) - 1)["chi2_p_value"])
    assert 0.02 <= np.mean(np.array(p_values) < 0.05) <= 0.10


# ---------------------------------------------------------------------------
# Bins on synthetic data
# ---------------------------------------------------------------------------


def test_numeric_bins_cover_the_whole_line_with_a_missing_bin():
    bins = monitoring.numeric_bins("x", pd.Series(np.arange(1_000.0)), n_bins=10)
    ranges = bins[bins["kind"] == "range"]
    assert len(ranges) == 10
    assert ranges["lower"].iloc[0] == -np.inf and ranges["upper"].iloc[-1] == np.inf
    assert (ranges["lower"].iloc[1:].to_numpy() == ranges["upper"].iloc[:-1].to_numpy()).all()
    assert bins["bin_label"].is_unique
    assert bins["kind"].iloc[-1] == "missing"


def test_tied_values_give_fewer_bins_but_none_empty():
    values = pd.Series([1.0] * 500 + [2.0] * 300 + [3.0] * 200)
    bins = monitoring.numeric_bins("x", values, n_bins=10)
    counts = monitoring.bin_counts(values, bins)
    ranges = bins["kind"] == "range"
    assert ranges.sum() < 10
    assert (counts[ranges.to_numpy()] > 0).all()


def test_a_value_on_an_edge_goes_to_the_upper_bin():
    bins = monitoring.numeric_bins("x", pd.Series(np.arange(100.0)), n_bins=2)
    edge = bins["upper"].iloc[0]
    assert monitoring.assign_bins([edge], bins)[0] == bins["bin_label"].iloc[1]


def test_missing_values_and_unseen_levels_get_their_own_bins():
    cat_bins = monitoring.categorical_bins("c", pd.Series(["a", "b", None]))
    labels = monitoring.assign_bins(pd.Series(["a", None, "new"]), cat_bins)
    assert labels.tolist() == ["a", config.MISSING_BIN_LABEL, config.UNSEEN_BIN_LABEL]
    num_bins = monitoring.numeric_bins("x", pd.Series([1.0, 2.0, 3.0, np.nan]), n_bins=2)
    assert monitoring.assign_bins(pd.Series([np.nan]), num_bins)[0] == config.MISSING_BIN_LABEL


def test_baseline_shares_add_up_to_one_per_variable():
    rng = np.random.default_rng(2)
    dev = pd.DataFrame({
        config.MONITORING_SCORE_VARIABLE: rng.uniform(0, 1, 500),
        "loan_amount": rng.lognormal(12, 0.4, 500),
        config.INCOME_CLEAN_COL: np.where(rng.random(500) < 0.1, np.nan, rng.lognormal(8.5, 0.6, 500)),
        config.MONITORING_GRADE_VARIABLE: rng.choice(list("ABC"), 500),
        "lump_sum_payment": rng.choice(["lpsm", "not_lpsm"], 500),
        "Neg_ammortization": rng.choice(["neg_amm", "not_neg"], 500),
        "loan_type": rng.choice(["type1", "type2"], 500),
        "loan_purpose": rng.choice(["p1", "p3"], 500),
    })
    baseline = monitoring.build_baseline(dev)
    assert set(baseline["variable"]) == set(config.MONITORING_NUMERIC_VARIABLES + config.MONITORING_CATEGORICAL_VARIABLES)
    assert baseline.groupby("variable")["share"].sum().round(9).eq(1).all()
    assert (baseline.groupby("variable")["n_loans"].sum() == 500).all()


def test_grade_backtest_lights_follow_the_d025_thresholds():
    scale = pd.DataFrame({"grade": ["A", "B", "C"], "grade_pd": [0.10, 0.20, 0.30]})
    y = np.r_[np.repeat([1, 0], [100, 900]), np.repeat([1, 0], [300, 700]), np.zeros(0)]
    grade = np.r_[["A"] * 1_000, ["B"] * 1_000]
    table = monitoring.grade_backtest(y, grade, scale, "test")
    a, b, c = (table.iloc[i] for i in range(3))
    assert a["gap"] == pytest.approx(0.0) and a["gap_light"] == "green" and a["binomial_light"] == "green"
    assert b["gap"] == pytest.approx(0.10) and b["gap_light"] == "red" and b["binomial_light"] == "red"
    assert c["n_loans"] == 0 and c["gap_light"] == "n/a"


def test_hide_small_counts_blanks_only_counts_between_one_and_nine():
    table = pd.DataFrame({"n_loans": [0, 1, 9, 10], "share": [0.0, 0.1, 0.2, 0.7]})
    out = monitoring.hide_small_counts(table, {"n_loans": "share"})
    assert out["n_loans"].isna().tolist() == [False, True, True, False]
    assert out["share"].isna().tolist() == [False, True, True, False]


def test_sql_variable_names_and_labels_match_config():
    sql = db.read_sql("monitoring_bin_counts")
    numeric_in = re.search(r"num_value FOR variable IN \((.*?)\)", sql).group(1).split(", ")
    assert numeric_in == config.MONITORING_NUMERIC_VARIABLES
    categorical_in = re.search(r"cat_value FOR variable IN \((.*?)\)", sql).group(1).split(", ")
    assert categorical_in == config.MONITORING_CATEGORICAL_VARIABLES
    assert f"'{config.MISSING_BIN_LABEL}'" in sql and f"'{config.UNSEEN_BIN_LABEL}'" in sql
    assert "Status" not in sql.split("CREATE OR REPLACE VIEW")[1]  # no outcome is read


def test_missing_pd_grades_fails_with_a_clear_message(toy_con):  # noqa: F811
    with pytest.raises(RuntimeError, match="python -m src.grades"):
        monitoring.load_monitoring_inputs(toy_con)


# ---------------------------------------------------------------------------
# Toy database: Stage 5 -> Stage 6 -> Stage 7 -> Stage 8
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_stage8(toy_stage6, toy_stage7, tmp_path):  # noqa: F811
    _, out6, model_path, _, db_path = toy_stage6
    out = tmp_path / "monitoring_out"
    tables = monitoring.run_stage8(db_path, output_dir=out, model_path=model_path, verbose=False)
    return tables, out, db_path, model_path


def test_run_stage8_writes_every_artifact_to_the_given_folder(toy_stage8):
    tables, out, *_ = toy_stage8
    for name in tables:
        assert (out / f"{config.MONITORING_ARTIFACT_PREFIX}{name}.csv").exists()


def test_sql_bin_counts_match_python_and_cover_every_scored_loan(toy_stage8):
    tables, _, db_path, _ = toy_stage8
    con = db.connect(db_path)
    try:
        inputs = monitoring.load_monitoring_inputs(con)
    finally:
        con.close()
    baseline, psi_bins = tables["baseline"], tables["psi_bins"]
    dev = inputs[inputs["sample"] == config.SAMPLE_DEVELOPMENT]
    hold = inputs[inputs["sample"] == config.SAMPLE_HOLDOUT]
    for variable, bins in baseline.groupby("variable", sort=False):
        sql = psi_bins[psi_bins["variable"] == variable]
        assert sql["n_baseline"].tolist() == bins["n_loans"].tolist()
        assert sql["n_monitored"].tolist() == monitoring.bin_counts(hold[variable], bins).tolist()
        assert sql["n_baseline"].sum() == len(dev) and sql["n_monitored"].sum() == len(hold)


def test_sql_psi_matches_python_psi(toy_stage8):
    tables, _, db_path, _ = toy_stage8
    con = db.connect(db_path)
    try:
        inputs = monitoring.load_monitoring_inputs(con)
    finally:
        con.close()
    dev = inputs[inputs["sample"] == config.SAMPLE_DEVELOPMENT]
    hold = inputs[inputs["sample"] == config.SAMPLE_HOLDOUT]
    summary = tables["psi"].set_index("variable")
    for variable, bins in tables["baseline"].groupby("variable", sort=False):
        python_psi = monitoring.psi_from_values(dev[variable], hold[variable], bins)
        assert summary.loc[variable, "psi"] == pytest.approx(python_psi, rel=1e-9, abs=1e-12)


def test_characteristic_changes_add_up_to_the_change_in_mean_log_odds(toy_stage8):
    tables, _, db_path, model_path = toy_stage8
    char = tables["characteristic"].set_index("feature")
    pipeline = monitoring.holdout.load_frozen_model(model_path)
    con = db.connect(db_path)
    try:
        inputs = monitoring.load_monitoring_inputs(con)
    finally:
        con.close()

    def mean_log_odds(sample: str) -> float:
        part = inputs[inputs["sample"] == sample]
        return float(pipeline.decision_function(part[config.MAIN_MODEL_FEATURES]).mean())

    change = mean_log_odds(config.SAMPLE_HOLDOUT) - mean_log_odds(config.SAMPLE_DEVELOPMENT)
    assert char.loc["total", "change_in_log_odds"] == pytest.approx(change)
    assert char.drop("total")["change_in_log_odds"].sum() == pytest.approx(change)


def test_watch_list_has_one_light_per_kpi(toy_stage8):
    watch = toy_stage8[0]["watch_list"]
    assert len(watch) == 6
    assert set(watch["light"]) <= {"green", "amber", "red"}
    assert watch["kpi"].is_unique


def test_backtest_uses_development_samples_only(toy_stage8):
    backtest = toy_stage8[0]["grade_backtest"]
    assert set(backtest["sample"]) == {"development_in_sample", "development_out_of_fold"}


def test_holdout_outcomes_do_not_change_any_stage8_output(toy_stage8, tmp_path):
    """D-025 / D-028: flipping every hold-out Status must leave every Stage 8 table unchanged."""
    before, _, db_path, model_path = toy_stage8
    con = db.connect(db_path)
    try:
        con.execute(
            f"UPDATE {config.LOANS_TABLE} SET {config.TARGET_COL} = 1 - {config.TARGET_COL} "
            f"WHERE {config.ID_COL} IN (SELECT {config.ID_COL} FROM {config.SPLIT_TABLE} WHERE sample = $s)",
            {"s": config.SAMPLE_HOLDOUT},
        )
    finally:
        con.close()
    after = monitoring.run_stage8(db_path, output_dir=tmp_path / "after", model_path=model_path, verbose=False)
    for name in before:
        pd.testing.assert_frame_equal(before[name], after[name])


def test_an_unseen_level_in_the_monitored_sample_is_counted_and_red(toy_stage8, tmp_path):
    _, _, db_path, model_path = toy_stage8
    con = db.connect(db_path)
    try:
        hold_id = con.execute(
            f"SELECT min({config.ID_COL}) FROM {config.SPLIT_TABLE} WHERE sample = $s", {"s": config.SAMPLE_HOLDOUT}
        ).fetchone()[0]
        con.execute(f"UPDATE {config.LOANS_TABLE} SET loan_type = 'type9' WHERE {config.ID_COL} = $id", {"id": hold_id})
        inputs = monitoring.load_monitoring_inputs(con)
        baseline = monitoring.build_baseline(inputs[inputs["sample"] == config.SAMPLE_DEVELOPMENT])
        monitoring.write_baseline(con, baseline)
        psi_bins = db.run_query(con, "monitoring_psi", baseline=config.SAMPLE_DEVELOPMENT,
                                monitored=config.SAMPLE_HOLDOUT, eps=config.PSI_EPSILON)
        scope = db.run_query(con, "grades_scope", equi=config.EQUI_LEVEL)
    finally:
        con.close()
    unseen = psi_bins[(psi_bins["variable"] == "loan_type") & (psi_bins["bin_label"] == config.UNSEEN_BIN_LABEL)]
    assert unseen["n_monitored"].iloc[0] == 1
    watch = monitoring.watch_list(psi_bins, baseline, scope, config.SAMPLE_DEVELOPMENT, config.SAMPLE_HOLDOUT)
    assert watch.set_index("kpi").loc["n_unseen_levels", "light"] == "red"


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

needs_real_outputs = pytest.mark.skipif(
    not (config.RAW_DATA_PATH.exists() and config.DUCKDB_PATH.exists() and config.MODEL_PATH.exists()),
    reason="needs the raw CSV and the Stage 4-7 outputs (not available on GitHub)",
)


@pytest.fixture(scope="module")
def real_stage8(tmp_path_factory):
    return monitoring.run_stage8(output_dir=tmp_path_factory.mktemp("monitoring"), verbose=False)


@needs_real_outputs
def test_real_monitoring_covers_both_in_scope_samples(real_stage8):
    psi = real_stage8["psi"]
    assert (psi["n_baseline"] == 93_391).all()
    assert (psi["n_monitored"] == 39_981).all()


@needs_real_outputs
def test_real_random_split_gives_near_zero_psi(real_stage8):
    # D-028: two random samples of one population; near zero by construction, not evidence of stability
    psi = real_stage8["psi"]
    assert (psi["psi"] < 0.01).all()
    assert (psi["light"] == "green").all()


@needs_real_outputs
def test_real_committed_tables_have_no_small_cells(real_stage8):
    bins = real_stage8["psi_bins"]
    for col in ["n_baseline", "n_monitored"]:
        assert not bins[col].between(1, config.SQL_MIN_SEGMENT_SIZE - 1).any()


@needs_real_outputs
def test_real_grade_a_baseline_share_matches_stage7(real_stage8):
    scale = pd.read_csv(config.ARTIFACTS_DIR / f"{config.GRADES_ARTIFACT_PREFIX}scale.csv")
    watch = real_stage8["watch_list"].set_index("kpi")
    assert watch.loc["grade_A_share_change", "baseline_value"] == pytest.approx(scale["share"].iloc[0], abs=1e-6)
