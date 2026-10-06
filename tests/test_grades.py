"""Tests for the Stage 7 illustrative risk grades (src/grades.py).

Unit tests use small synthetic data, and the toy database from tests/test_model.py
(through the Stage 6 fixture in tests/test_holdout.py). Tests on the real data skip
automatically if data/raw/Loan_Default.csv or the Stage 4-6 outputs are absent (D-002).
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src import db
from src import grades
from src import model
from tests.test_holdout import toy_stage6  # noqa: F401  (pytest fixture)
from tests.test_model import toy_con, toy_dev, toy_processed  # noqa: F401  (pytest fixtures)

# ---------------------------------------------------------------------------
# Synthetic PDs
# ---------------------------------------------------------------------------


def synthetic_scores(true_rate, n: int = 20_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """PDs spread evenly over (0, 1) and outcomes drawn with `true_rate(pd)`."""
    rng = np.random.default_rng(seed)
    pd_values = np.sort(rng.uniform(0.01, 0.99, n))
    y = (rng.random(n) < true_rate(pd_values)).astype(int)
    return pd_values, y


def flat_then_rising(p: np.ndarray) -> np.ndarray:
    """Default rate flat at 10% for the lower half of PDs, then rising: like evidence 1 in D-027."""
    return np.where(p < 0.5, 0.10, 0.10 + 0.8 * (p - 0.5))


def test_initial_edges_are_sorted_unique_and_rounded():
    pd_values, _ = synthetic_scores(flat_then_rising)
    edges = grades.initial_edges(pd_values, n_bins=20)
    assert len(edges) == 19
    assert edges == sorted(set(edges))
    assert all(round(e, config.GRADE_EDGE_DECIMALS) == e for e in edges)


def test_initial_edges_never_leave_a_bin_empty():
    pd_values = np.array([0.1] * 50 + [0.2] * 5 + [0.3] * 45)  # heavy ties
    edges = grades.initial_edges(pd_values, n_bins=10)
    sizes = np.bincount(grades.assign_band(pd_values, edges), minlength=len(edges) + 1)
    assert (sizes > 0).all()


def test_assign_band_puts_a_pd_on_a_boundary_in_the_upper_band():
    # same rule as sql/grade_assignment.sql: pd_lower <= pd < pd_upper
    assert grades.assign_band([0.05, 0.1, 0.15, 0.2], [0.1, 0.2]).tolist() == [0, 1, 1, 2]


@pytest.mark.parametrize("bad_pd", [1.5, -0.1, np.nan])
def test_assign_grade_rejects_a_pd_outside_0_1_or_missing(bad_pd):
    # D-027: an impossible PD is an upstream error; it is never graded or clipped
    with pytest.raises(ValueError, match="not graded"):
        grades.assign_grade([0.2, bad_pd], [0.1, 0.2])


def test_assign_grade_uses_the_assign_band_rule_inside_0_1():
    pd_values = [0.0, 0.05, 0.1, 0.15, 0.2, 1.0]
    assert grades.assign_grade(pd_values, [0.1, 0.2]).tolist() == grades.assign_band(pd_values, [0.1, 0.2]).tolist()


def test_step_p_values_small_for_a_real_step_and_large_for_a_step_down():
    counts = pd.DataFrame({"n_loans": [1_000, 1_000, 1_000], "n_defaults": [100, 200, 150]})
    p = grades.step_p_values(counts)
    assert p[0] < 0.001
    assert p[1] > 0.5


def test_merge_bands_merges_a_flat_region_and_leaves_significant_monotone_steps():
    pd_values, y = synthetic_scores(flat_then_rising)
    edges, log = grades.merge_bands(pd_values, y, grades.initial_edges(pd_values, n_bins=20))
    counts = grades.band_counts(pd_values, y, edges)
    assert grades.is_increasing(counts["observed_rate"])
    assert (grades.step_p_values(counts) < config.GRADE_MERGE_ALPHA).all()
    assert (counts["n_loans"] >= grades.min_grade_loans(len(y))).all()
    # the flat lower half ends up in (almost) one grade
    assert counts["n_loans"].iloc[0] / len(y) > 0.35
    assert not log.empty and set(log["reason"]) <= {
        "step not significant", "band below minimum share", "more bands than the maximum"
    }


def test_merge_bands_respects_the_maximum_number_of_grades():
    pd_values, y = synthetic_scores(lambda p: p, n=100_000)  # every step clearly significant
    edges, _ = grades.merge_bands(pd_values, y, grades.initial_edges(pd_values, n_bins=20), max_grades=6)
    assert len(edges) + 1 <= 6


def test_merge_bands_respects_the_minimum_share():
    pd_values, y = synthetic_scores(lambda p: p, n=100_000)
    edges, _ = grades.merge_bands(pd_values, y, grades.initial_edges(pd_values, n_bins=20), min_share=0.15)
    counts = grades.band_counts(pd_values, y, edges)
    assert (counts["n_loans"] >= grades.min_grade_loans(len(y), 0.15)).all()


def test_min_grade_loans_rounds_down():
    # 20 equal-count bins of 93,391 loans hold 4,669 or 4,670 each: 4,669 must still pass (D-027)
    assert grades.min_grade_loans(93_391, 0.05) == 4_669


def test_equal_count_bins_a_fraction_short_of_the_share_are_not_merged():
    pd_values, y = synthetic_scores(lambda p: p, n=100_019)  # 100,019 / 20 is not a whole number
    start = grades.initial_edges(pd_values, n_bins=20)
    edges, log = grades.merge_bands(pd_values, y, start, max_grades=20)
    assert "band below minimum share" not in set(log.get("reason", []))
    assert edges == start


def test_grade_scale_covers_zero_to_one_without_gaps():
    pd_values, _ = synthetic_scores(flat_then_rising)
    scale = grades.build_grade_scale([0.2, 0.5, 0.8], pd_values)
    assert scale["grade"].tolist() == ["A", "B", "C", "D"]
    assert scale["pd_lower"].iloc[0] == 0.0 and scale["pd_upper"].iloc[-1] == 1.0
    assert (scale["pd_lower"].iloc[1:].to_numpy() == scale["pd_upper"].iloc[:-1].to_numpy()).all()
    assert scale["grade_pd"].is_monotonic_increasing


def test_out_of_fold_predictions_use_the_stage5_folds(toy_dev):  # noqa: F811
    X, y = toy_dev[config.MAIN_MODEL_FEATURES], toy_dev[config.TARGET_COL]
    pd_oof, fold = model.out_of_fold_predictions(model.build_pipeline(), X, y)
    cv_metrics, _ = model.cross_validate_model(model.build_pipeline(), X, y)
    for f in range(1, config.CV_FOLDS + 1):
        rows = fold == f
        auc = grades.validation.discrimination_metrics(y[rows], pd_oof[rows])["auc"]
        assert auc == pytest.approx(cv_metrics.loc[cv_metrics["fold"] == f, "auc"].iloc[0])


def test_missing_pd_scores_fails_with_a_clear_message(toy_con):  # noqa: F811
    with pytest.raises(RuntimeError, match="python -m src.holdout"):
        grades.load_development_scores(toy_con)


# ---------------------------------------------------------------------------
# Toy database: Stage 5 -> Stage 6 -> Stage 7
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_stage7(toy_stage6, tmp_path):  # noqa: F811
    *_, db_path = toy_stage6
    out = tmp_path / "grades_out"
    return grades.run_stage7(db_path, output_dir=out, verbose=False), out, db_path


def test_run_stage7_writes_every_artifact_to_the_given_folder(toy_stage7):
    tables, out, _ = toy_stage7
    for name in tables:
        assert (out / f"{config.GRADES_ARTIFACT_PREFIX}{name}.csv").exists()


def test_every_scored_loan_gets_exactly_one_grade(toy_stage7):
    _, _, db_path = toy_stage7
    con = db.connect(db_path)
    try:
        n_scores = con.execute(f"SELECT count(*) FROM {config.PD_SCORES_TABLE}").fetchone()[0]
        graded = con.execute(f"SELECT * FROM {config.PD_GRADES_VIEW}").df()
    finally:
        con.close()
    assert len(graded) == n_scores
    assert graded[config.ID_COL].is_unique


def test_scope_table_reports_no_ungraded_in_scope_loan(toy_stage7):
    tables, *_ = toy_stage7
    scope = tables["scope"]
    assert (scope["n_in_scope_not_graded"] == 0).all()
    assert (scope["n_graded"] + scope["n_out_of_scope_equi"] == scope["n_loans"]).all()


def test_sql_grade_summary_matches_python_band_counts(toy_stage7):
    tables, _, db_path = toy_stage7
    con = db.connect(db_path)
    try:
        dev = grades.load_development_scores(con)
    finally:
        con.close()
    scale = tables["scale"]
    counts = grades.band_counts(dev["pd"], dev[config.TARGET_COL], scale["pd_lower"].iloc[1:].tolist())
    assert scale["n_loans"].tolist() == counts["n_loans"].tolist()
    assert scale["n_defaults"].tolist() == counts["n_defaults"].tolist()
    assert scale["n_loans"].sum() == len(dev)


def test_holdout_outcomes_do_not_change_any_stage7_output(toy_stage7, tmp_path):
    """D-025: flipping every hold-out Status must leave every Stage 7 table unchanged."""
    before, _, db_path = toy_stage7
    con = db.connect(db_path)
    try:
        con.execute(
            f"UPDATE {config.LOANS_TABLE} SET {config.TARGET_COL} = 1 - {config.TARGET_COL} "
            f"WHERE {config.ID_COL} IN (SELECT {config.ID_COL} FROM {config.SPLIT_TABLE} WHERE sample = $s)",
            {"s": config.SAMPLE_HOLDOUT},
        )
    finally:
        con.close()
    after = grades.run_stage7(db_path, output_dir=tmp_path / "after", verbose=False)
    for name in before:
        pd.testing.assert_frame_equal(before[name], after[name])


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

needs_real_outputs = pytest.mark.skipif(
    not (config.RAW_DATA_PATH.exists() and config.DUCKDB_PATH.exists() and config.MODEL_PATH.exists()),
    reason="needs the raw CSV and the Stage 4-6 outputs (not available on GitHub)",
)


@pytest.fixture(scope="module")
def real_stage7(tmp_path_factory):
    return grades.run_stage7(output_dir=tmp_path_factory.mktemp("grades"), verbose=False)


@needs_real_outputs
def test_real_grades_cover_the_in_scope_development_sample(real_stage7):
    scale = real_stage7["scale"]
    assert scale["n_loans"].sum() == 104_069 - 10_678  # D-026: EQUI out of scope
    assert len(scale) <= config.GRADE_MAX_GRADES


@needs_real_outputs
def test_real_grades_pass_every_preset_check(real_stage7):
    checks = real_stage7["checks"]
    assert (checks.loc[checks["status"] != "report", "status"] == "pass").all()


@needs_real_outputs
def test_real_grade_scale_is_reproducible(real_stage7, tmp_path):
    again = grades.run_stage7(output_dir=tmp_path, verbose=False)
    pd.testing.assert_frame_equal(real_stage7["scale"], again["scale"])
