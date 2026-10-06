"""Tests for the Stage 9 dashboard (src/dashboard.py and app.py; decision log D-029).

Unit tests use small synthetic tables and a model fitted on synthetic loans. Tests on
the committed artifacts skip if they are absent; tests that need the fitted model skip
if artifacts/pd_model.joblib is absent (it is gitignored, so it is missing on GitHub).
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src import dashboard
from src import grades
from src import holdout
from src import model

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

SCALE = pd.DataFrame({
    "grade": ["A", "B", "C"],
    "pd_lower": [0.0, 0.1, 0.3],
    "pd_upper": [0.1, 0.3, 1.0],
    "grade_pd": [0.05, 0.2, 0.5],
})
KNOWN_LEVELS = {
    "lump_sum_payment": ["lpsm", "not_lpsm"],
    "Neg_ammortization": ["neg_amm", "not_neg"],
    "loan_type": ["type1", "type2", "type3"],
    "loan_purpose": ["p1", "p2", "p3", "p4"],
}
TYPICAL_LEVELS = dict(config.REFERENCE_LEVELS)


@pytest.fixture(scope="module")
def toy_pipeline():
    """The main-model pipeline fitted on 300 synthetic loans that use every known level."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        config.INCOME_CLEAN_COL: rng.lognormal(8.5, 0.5, n),
        "loan_amount": rng.lognormal(12.0, 0.4, n),
        **{col: rng.choice(levels, n) for col, levels in KNOWN_LEVELS.items()},
    })[config.MAIN_MODEL_FEATURES]
    X.loc[:20, config.INCOME_CLEAN_COL] = np.nan
    y = pd.Series(rng.random(n) < 0.3).astype(int)
    return model.build_pipeline().fit(X, y)


def coefficient_table_of(pipeline) -> pd.DataFrame:
    """A model_coefficients.csv-style table built from a fitted pipeline."""
    return pd.DataFrame({"feature": model.feature_names(pipeline),
                         "coefficient": pipeline["model"].coef_.ravel()})


def table_scale() -> pd.DataFrame:
    return dashboard.load_artifact(dashboard.GRADE_SCALE_FILE)


def write_csv(directory, name: str, table: pd.DataFrame) -> None:
    table.to_csv(directory / name, index=False)


# ---------------------------------------------------------------------------
# Loading artifacts
# ---------------------------------------------------------------------------


def test_load_artifact_reads_a_file_with_the_required_columns(tmp_path):
    columns = config.DASHBOARD_ARTIFACT_COLUMNS["validation_criteria.csv"]
    write_csv(tmp_path, "validation_criteria.csv", pd.DataFrame([["binomial_p", 0.5, "green"]], columns=columns))
    loaded = dashboard.load_artifact("validation_criteria.csv", tmp_path)
    assert loaded.columns.tolist() == columns


def test_load_artifact_fails_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dashboard.load_artifact("validation_criteria.csv", tmp_path)


def test_load_artifact_fails_on_a_missing_column(tmp_path):
    write_csv(tmp_path, "validation_criteria.csv", pd.DataFrame({"criterion": ["binomial_p"], "value": [0.5]}))
    with pytest.raises(ValueError, match="status"):
        dashboard.load_artifact("validation_criteria.csv", tmp_path)


def test_missing_artifacts_lists_every_absent_file(tmp_path):
    write_csv(tmp_path, "dq_summary.csv", pd.DataFrame({"check": ["n_rows"], "value": [1]}))
    missing = dashboard.missing_artifacts(tmp_path)
    assert "dq_summary.csv" not in missing
    assert len(missing) == len(config.DASHBOARD_ARTIFACT_COLUMNS) - 1


def test_criterion_rules_cover_every_d025_criterion():
    assert set(dashboard.criterion_rules()) == {
        "auc_drop_cv_minus_holdout", "binomial_p", "calibration_slope", "max_decile_gap"
    }


def test_decile_gap_rule_is_shown_in_percentage_points():
    gap_g, gap_a = config.CRITERION_DECILE_GAP
    rule = dashboard.criterion_rules()["max_decile_gap"]
    assert rule == f"green <= {gap_g * 100:g} pp, amber <= {gap_a * 100:g} pp"


def test_check_model_coefficients_accepts_the_matching_table(toy_pipeline):
    dashboard.check_model_coefficients(toy_pipeline, coefficient_table_of(toy_pipeline))


def test_check_model_coefficients_rejects_a_changed_coefficient(toy_pipeline):
    table = coefficient_table_of(toy_pipeline)
    table.loc[0, "coefficient"] += 1e-6
    with pytest.raises(ValueError, match="coefficient of"):
        dashboard.check_model_coefficients(toy_pipeline, table)


def test_check_model_coefficients_rejects_a_missing_feature(toy_pipeline):
    with pytest.raises(ValueError, match="features differ"):
        dashboard.check_model_coefficients(toy_pipeline, coefficient_table_of(toy_pipeline).iloc[1:])


def test_backtest_light_rules_show_the_config_thresholds():
    caption = dashboard.backtest_light_rules()
    gap_g, gap_a = config.CRITERION_DECILE_GAP
    p_g, p_a = config.CRITERION_BINOMIAL_P
    for text in (f"green ≤ {gap_g * 100:g} pp", f"amber ≤ {gap_a * 100:g} pp",
                 f"green ≥ {p_g:g}", f"amber ≥ {p_a:g}",
                 "heuristic", "D-025", "D-028", "reference, not a judgement"):
        assert text in caption, text


# ---------------------------------------------------------------------------
# Grade lookup: same rule as sql/grade_assignment.sql
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pd_value, expected", [
    (0.0, "A"), (0.0999, "A"), (0.1, "B"), (0.2999, "B"), (0.3, "C"), (0.99, "C"), (1.0, "C"),
])
def test_grade_for_pd_puts_a_boundary_in_the_upper_grade_and_1_in_the_top(pd_value, expected):
    assert dashboard.grade_for_pd(pd_value, SCALE)["grade"] == expected


@pytest.mark.parametrize("bad_pd", [1.5, -0.1, np.nan])
def test_grade_for_pd_rejects_a_pd_outside_0_1_or_missing(bad_pd):
    # the original 1.5 case is back: a PD above 1 must not be placed in the top grade
    with pytest.raises(ValueError, match="not graded"):
        dashboard.grade_for_pd(bad_pd, SCALE)


def test_check_grade_scale_accepts_a_contiguous_scale():
    dashboard.check_grade_scale(SCALE)


@pytest.mark.parametrize("pd_lower, pd_upper", [
    ([0.0, 0.1, 0.3], [0.1, 0.25, 1.0]),   # gap between 0.25 and 0.3
    ([0.0, 0.1, 0.3], [0.1, 0.35, 1.0]),   # overlap between 0.3 and 0.35
    ([0.05, 0.1, 0.3], [0.1, 0.3, 1.0]),   # does not start at 0
    ([0.0, 0.1, 0.3], [0.1, 0.3, 0.9]),    # does not end at 1
])
def test_check_grade_scale_rejects_gaps_overlaps_and_wrong_ends(pd_lower, pd_upper):
    with pytest.raises(ValueError, match="Invalid grade scale"):
        dashboard.check_grade_scale(SCALE.assign(pd_lower=pd_lower, pd_upper=pd_upper))


def test_check_grade_scale_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="unique"):
        dashboard.check_grade_scale(SCALE.assign(grade=["A", "B", "B"]))


def test_load_artifact_checks_the_grade_scale(tmp_path):
    columns = config.DASHBOARD_ARTIFACT_COLUMNS[dashboard.GRADE_SCALE_FILE]
    bad = pd.DataFrame({c: [0.0, 0.0] for c in columns}).assign(
        grade=["A", "B"], pd_lower=[0.0, 0.6], pd_upper=[0.5, 1.0])  # gap from 0.5 to 0.6
    write_csv(tmp_path, dashboard.GRADE_SCALE_FILE, bad)
    with pytest.raises(ValueError, match="gap or overlap"):
        dashboard.load_artifact(dashboard.GRADE_SCALE_FILE, tmp_path)


# ---------------------------------------------------------------------------
# Input ranges (D-029)
# ---------------------------------------------------------------------------

RANGES = pd.DataFrame({
    "variable": ["loan_amount", config.INCOME_CLEAN_COL],
    "lower": [50_000.0, 1_000.0],
    "upper": [800_000.0, 25_000.0],
})


@pytest.mark.parametrize("loan_amount, income, flagged", [
    (50_000, 1_000, []),                                   # both exactly at the lower bound: inside
    (800_000, 25_000, []),                                 # both exactly at the upper bound: inside
    (49_999, 6_000, ["loan_amount"]),
    (200_000, 25_001, [config.INCOME_CLEAN_COL]),
    (900_000, 999, ["loan_amount", config.INCOME_CLEAN_COL]),
    (900_000, None, ["loan_amount"]),                      # blank income is never flagged
])
def test_out_of_range_flags_only_values_outside_the_bounds(loan_amount, income, flagged):
    assert list(dashboard.out_of_range(loan_amount, income, RANGES)) == flagged


def test_psi_axis_keeps_the_bands_visible_for_small_psi():
    amber = config.PSI_THRESHOLDS[1]
    assert dashboard.psi_axis_max([0.0004, 0.0006]) == pytest.approx(amber * config.PSI_CHART_MIN_AXIS)


@pytest.mark.parametrize("largest", [0.5, 1.7])
def test_psi_axis_always_reaches_a_red_psi(largest):
    # a PSI above the default axis end (a red result) must stay on the chart
    axis_end = dashboard.psi_axis_max([0.01, largest])
    assert axis_end == pytest.approx(largest * config.PSI_CHART_HEADROOM)
    assert axis_end > largest


# ---------------------------------------------------------------------------
# Building and scoring one loan
# ---------------------------------------------------------------------------


def test_build_loan_frame_has_the_main_model_features_in_order():
    loan = dashboard.build_loan_frame(200_000, 6_000, TYPICAL_LEVELS, "CIB", KNOWN_LEVELS)
    assert loan.columns.tolist() == config.MAIN_MODEL_FEATURES
    assert len(loan) == 1


def test_build_loan_frame_scores_blank_income_as_missing():
    loan = dashboard.build_loan_frame(200_000, None, TYPICAL_LEVELS, "CIB", KNOWN_LEVELS)
    assert np.isnan(loan[config.INCOME_CLEAN_COL].iloc[0])


def test_build_loan_frame_refuses_an_out_of_scope_equi_loan():
    with pytest.raises(ValueError, match="D-026"):
        dashboard.build_loan_frame(200_000, 6_000, TYPICAL_LEVELS, config.EQUI_LEVEL, KNOWN_LEVELS)


@pytest.mark.parametrize("loan_amount, income", [(0, 6_000), (-1, 6_000), (200_000, 0), (200_000, -5)])
def test_build_loan_frame_refuses_non_positive_amounts(loan_amount, income):
    with pytest.raises(ValueError):
        dashboard.build_loan_frame(loan_amount, income, TYPICAL_LEVELS, "CIB", KNOWN_LEVELS)


def test_build_loan_frame_refuses_a_level_the_model_has_not_seen():
    levels = {**TYPICAL_LEVELS, "loan_type": "type9"}
    with pytest.raises(ValueError, match="D-023"):
        dashboard.build_loan_frame(200_000, 6_000, levels, "CIB", KNOWN_LEVELS)


def test_load_model_if_available_returns_none_without_a_model(tmp_path):
    assert dashboard.load_model_if_available(tmp_path / "no_model.joblib") is None


def test_model_levels_are_the_encoder_levels(toy_pipeline):
    assert dashboard.model_levels(toy_pipeline) == KNOWN_LEVELS


def test_a_built_loan_scores_with_the_pipeline(toy_pipeline):
    for income in (6_000, None):
        loan = dashboard.build_loan_frame(200_000, income, TYPICAL_LEVELS, "CIB", KNOWN_LEVELS)
        pd_value = holdout.score(toy_pipeline, loan).iloc[0]
        assert 0 < pd_value < 1


# ---------------------------------------------------------------------------
# Committed artifacts and the app itself
# ---------------------------------------------------------------------------

needs_artifacts = pytest.mark.skipif(
    bool(dashboard.missing_artifacts()), reason="the committed artifacts/*.csv tables are absent"
)
needs_model = pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="needs artifacts/pd_model.joblib (gitignored, not on GitHub)"
)
DASHBOARD_PAGES = ["Overview", "Leakage finding", "Model", "Validation", "Risk grades",
                   "Monitoring", "Score a loan"]


@needs_artifacts
@pytest.mark.parametrize("name", list(config.DASHBOARD_ARTIFACT_COLUMNS))
def test_committed_artifact_has_the_columns_the_dashboard_uses(name):
    assert len(dashboard.load_artifact(name)) > 0


@needs_artifacts
def test_every_reused_figure_exists():
    for file_name in config.DASHBOARD_FIGURES.values():
        assert (config.FIGURES_DIR / file_name).exists(), file_name


@needs_artifacts
def test_real_grade_scale_dashboard_agrees_with_stage7_at_every_boundary():
    # reconciliation: at each published boundary and just below it, the dashboard gives the
    # grade Stage 7 gives (grades.assign_band on the same edges); 0 and 1 are included
    scale = dashboard.load_artifact(dashboard.GRADE_SCALE_FILE)
    edges = scale["pd_lower"].iloc[1:].tolist()
    points = [config.PD_MIN, config.PD_MAX] + [p for edge in edges for p in (edge, edge - 1e-9)]
    stage7 = scale["grade"].to_numpy()[grades.assign_band(points, edges)]
    assert [dashboard.grade_for_pd(p, scale)["grade"] for p in points] == stage7.tolist()
    # a PD exactly on a boundary belongs to the upper grade, as in sql/grade_assignment.sql
    for rank, edge in enumerate(edges, start=1):
        assert dashboard.grade_for_pd(edge, scale)["grade"] == scale["grade"].iloc[rank]


@needs_artifacts
def test_every_dashboard_page_renders_without_an_error():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=60).run()
    assert app.sidebar.radio[0].options == DASHBOARD_PAGES
    for page in DASHBOARD_PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {app.exception}"
        assert app.title[0].value


@needs_artifacts
def test_pages_show_the_consistency_notes():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=60).run()

    app.sidebar.radio[0].set_value("Validation").run()
    assert any("Hosmer-Lemeshow" in c.value and "D-025" in c.value for c in app.caption)

    app.sidebar.radio[0].set_value("Leakage finding").run()
    assert any("EQUI included" in c.value and "D-024" in c.value for c in app.caption)
    assert f"{config.CV_FOLDS}-fold" in " ".join(m.value for m in app.markdown)

    app.sidebar.radio[0].set_value("Model").run()
    assert any("never acted on" in c.value and "D-022" in c.value for c in app.caption)
    shown = [df.value for df in app.dataframe if "model" in df.value.columns]
    expected_models = {"main", *[f"without_{f}" for f in config.SENSITIVITY_DROPPED_FEATURES]}
    assert any(set(t["model"]) == expected_models for t in shown)  # the mean rows are on the page

    app.sidebar.radio[0].set_value("Risk grades").run()
    scale = dashboard.load_artifact(dashboard.GRADE_SCALE_FILE)
    n_loans = int(scale["n_loans"].sum())
    text = " ".join(m.value for m in app.markdown) + " ".join(i.value for i in app.info)
    assert f"at least {grades.min_grade_loans(n_loans):,} loans" in text
    assert f"{len(scale)} with the adopted settings" in text
    assert "notebooks/04_risk_grades.ipynb" in text
    assert not app.exception


@needs_artifacts
@needs_model
def test_real_model_matches_the_committed_coefficients():
    dashboard.check_model_coefficients(dashboard.load_model_if_available(),
                                       dashboard.load_artifact("model_coefficients.csv"))


@needs_artifacts
def test_monitoring_page_explains_the_backtest_lights():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Monitoring").run()
    assert not app.exception
    assert dashboard.backtest_light_rules() in [c.value for c in app.caption]


@needs_artifacts
@needs_model
def test_score_page_scores_an_in_scope_loan_and_refuses_equi():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Score a loan").run()
    credit_type = app.selectbox[0]
    in_scope = next(level for level in credit_type.options if level != config.EQUI_LEVEL)

    credit_type.set_value(in_scope)
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert [m.label for m in app.metric] == ["Model PD", "Illustrative grade", "Grade PD"]
    assert not app.warning[1:]  # only the page's standing notice; the medians are in range

    # the PD shown is the frozen model's PD for the same inputs (the starting medians)
    pipeline = dashboard.load_model_if_available()
    ranges = dashboard.load_artifact("sql_model_input_ranges.csv").set_index("variable")
    loan = dashboard.build_loan_frame(
        ranges.loc["loan_amount", "median"], ranges.loc[config.INCOME_CLEAN_COL, "median"],
        TYPICAL_LEVELS, in_scope, dashboard.model_levels(pipeline))
    expected_pd = holdout.score(pipeline, loan).iloc[0]
    assert app.metric[0].value == f"{expected_pd:.2%}"
    assert app.metric[1].value == dashboard.grade_for_pd(expected_pd, table_scale())["grade"]

    app.selectbox[0].set_value(config.EQUI_LEVEL)
    app.button[0].click().run()
    assert not app.metric
    assert "D-026" in app.error[0].value


@needs_artifacts
@needs_model
def test_score_page_warns_on_an_out_of_range_input_but_still_shows_the_pd():
    # D-029: an input above the 99th percentile of in-scope development loans is scored,
    # with an amber extrapolation warning next to the PD
    from streamlit.testing.v1 import AppTest

    ranges = dashboard.load_artifact("sql_model_input_ranges.csv").set_index("variable")
    app = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Score a loan").run()
    app.selectbox[0].set_value(next(v for v in app.selectbox[0].options if v != config.EQUI_LEVEL))
    app.number_input[0].set_value(float(ranges.loc["loan_amount", "upper"]) * 2)
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert [m.label for m in app.metric] == ["Model PD", "Illustrative grade", "Grade PD"]
    assert any("Extrapolation" in w.value and "loan_amount" in w.value for w in app.warning)


@needs_artifacts
@needs_model
def test_real_model_scores_a_typical_in_scope_loan_into_a_published_grade():
    pipeline = dashboard.load_model_if_available()
    known = dashboard.model_levels(pipeline)
    assert all(config.REFERENCE_LEVELS[col] in known[col] for col in config.CATEGORICAL_FEATURES)

    loan = dashboard.build_loan_frame(200_000, 6_000, TYPICAL_LEVELS, "CIB", known)
    pd_value = holdout.score(pipeline, loan).iloc[0]
    scale = dashboard.load_artifact("grades_scale.csv")
    assert 0 < pd_value < 1
    assert dashboard.grade_for_pd(pd_value, scale)["grade"] in set(scale["grade"])
