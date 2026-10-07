"""Tests for the Stage 9 dashboard (app.py, src/dashboard.py, src/dashboard_charts.py,
src/dashboard_text.py; decision log D-029, D-030).

Unit tests use small synthetic tables. Tests on the committed artifacts skip if they
are absent. The dashboard loads no model: the loan-scoring page was removed (D-030).
"""

import base64
import json
import re
import tomllib

import numpy as np
import pandas as pd
import pytest

from src import config
from src import dashboard
from src import dashboard_charts as charts
from src import dashboard_text
from src import grades

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

SCALE = pd.DataFrame({
    "grade": ["A", "B", "C"],
    "pd_lower": [0.0, 0.1, 0.3],
    "pd_upper": [0.1, 0.3, 1.0],
    "grade_pd": [0.05, 0.2, 0.5],
    "observed_rate": [0.06, 0.18, 0.55],
})


def write_csv(directory, name: str, table: pd.DataFrame) -> None:
    table.to_csv(directory / name, index=False)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(config.ARTIFACTS_DIR / name)


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


def test_backtest_light_rules_show_the_config_thresholds():
    caption = dashboard.backtest_light_rules()
    gap_g, gap_a = config.CRITERION_DECILE_GAP
    p_g, p_a = config.CRITERION_BINOMIAL_P
    for text in (f"green ≤ {gap_g * 100:g} pp", f"amber ≤ {gap_a * 100:g} pp",
                 f"green ≥ {p_g:g}", f"amber ≥ {p_a:g}",
                 "heuristic", "D-025", "D-028", "reference, not a judgement"):
        assert text in caption, text


# ---------------------------------------------------------------------------
# Grade lookup: same rule as sql/grade_assignment.sql (used by the grade slider)
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


PSI_TABLE = pd.DataFrame({"variable": ["loan_amount", "score"], "psi": [0.0004, 1.7],
                          "psi_expected_no_shift": [0.0003, 0.0004], "light": ["green", "red"]})


def test_psi_chart_never_cuts_off_a_red_psi():
    # the axis has no fixed end: it fits the data from zero, so a red PSI stays on the chart
    fig = charts.psi(PSI_TABLE)
    assert fig.layout.xaxis.range is None and fig.layout.xaxis.rangemode == "tozero"
    assert list(fig.data[-1].x) == [0.0004, 1.7]      # observed PSI
    assert list(fig.data[-2].x) == [0.0003, 0.0004]   # expected with no shift


# ---------------------------------------------------------------------------
# Presentation helpers (D-030): names, formats, tables, links, text, theme
# ---------------------------------------------------------------------------


def test_there_is_no_scoring_page():
    # D-030: the loan-scoring page was removed; the dashboard has six read-only pages
    assert list(config.DASHBOARD_PAGES) == ["overview", "leakage", "model", "validation", "grades", "monitoring"]
    for name in ("scoring_switched_on", "load_model_if_available", "build_loan_frame"):
        assert not hasattr(dashboard, name), name


def test_page_titles_drop_the_number_and_next_page_follows_reading_order():
    pages = list(config.DASHBOARD_PAGES)
    assert dashboard.page_title("model") == "Model"
    assert [dashboard.next_page(p) for p in pages] == pages[1:] + [None]


@pytest.mark.parametrize("p_value, shown", [
    (0.0, "< 0.001"), (0.0009, "< 0.001"), (0.001, "0.001"), (0.4691, "0.469"), (np.nan, ""),
])
def test_p_values_are_shown_as_below_the_floor_or_rounded(p_value, shown):
    assert dashboard.format_p_value(p_value) == shown


def test_readable_table_labels_columns_and_shows_p_values_and_lights_as_text():
    raw = pd.DataFrame({"grade": ["A", "B"], "observed_rate": [0.1, 0.2], "binomial_p": [0.5, 1e-9],
                        "gap_light": ["green", "red"], "n_loans": [10.0, 20.0], "gap": [0.01, -0.02]})
    shown = dashboard.readable_table(raw)
    assert shown.columns.tolist() == [config.COLUMN_LABELS[c] for c in raw.columns]
    assert shown[config.COLUMN_LABELS["binomial_p"]].tolist() == ["0.500", "< 0.001"]
    assert shown[config.COLUMN_LABELS["gap_light"]].tolist() == [config.LIGHT_LABELS["green"],
                                                                 config.LIGHT_LABELS["red"]]
    assert raw["binomial_p"].tolist() == [0.5, 1e-9]  # the input table is not changed
    assert dashboard.column_formats(raw) == {
        config.COLUMN_LABELS["observed_rate"]: "percent",
        config.COLUMN_LABELS["n_loans"]: "count",
        config.COLUMN_LABELS["gap"]: "percent",
    }


@pytest.mark.parametrize("heading, anchor", [
    ("D-030 Dashboard presentation and public-copy rules",
     "d-030-dashboard-presentation-and-public-copy-rules"),
    ("D-026 The main model's scope excludes `credit_type = EQUI`",
     "d-026-the-main-models-scope-excludes-credit_type--equi"),
    ("D-005 `property_value < 10,000` is treated as invalid (set to missing, rows kept)",
     "d-005-property_value--10000-is-treated-as-invalid-set-to-missing-rows-kept"),
])
def test_github_anchor_follows_githubs_heading_rule(heading, anchor):
    assert dashboard.github_anchor(heading) == anchor


def test_every_linked_decision_exists_in_the_decision_log():
    for page, ids in config.DASHBOARD_PAGE_DECISIONS.items():
        links = dashboard.decision_links(ids)
        assert [text.split(" ", 1)[0] for text, _ in links] == ids, page
        for _, url in links:
            assert url.startswith(f"{config.REPO_URL}/blob/{config.DOCS_GIT_REF}/docs/decision_log.md#d-")


def test_decision_links_refuse_an_unknown_id():
    with pytest.raises(ValueError, match="D-999"):
        dashboard.decision_links(["D-001", "D-999"])


def test_dashboard_text_types_no_result_numbers():
    # D-030: numbers on the dashboard come from the CSVs or config. The text may only hold
    # decision ids, stage names and the numbers that define a measure (0.5 coin flip, 0/1,
    # 2 x AUC - 1).
    source = (config.PROJECT_ROOT / "src" / "dashboard_text.py").read_text(encoding="utf-8")
    without_ids = re.sub(r"D-\d{3}|Stage \d+", "", source)
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", without_ids))
    assert numbers <= {"0", "0.5", "1", "2"}, numbers


def test_streamlit_theme_is_the_fixed_dark_theme_in_config():
    # D-030: .streamlit/config.toml mirrors config.DASHBOARD_THEME (documented dark palette values)
    with open(config.PROJECT_ROOT / ".streamlit" / "config.toml", "rb") as f:
        theme = tomllib.load(f)["theme"]
    sidebar = theme.pop("sidebar")
    assert theme == config.DASHBOARD_THEME
    assert theme["base"] == "dark"
    assert sidebar == {"backgroundColor": config.DASHBOARD_SIDEBAR_BACKGROUND}
    assert config.DASHBOARD_COLORS["series_1"] == theme["chartCategoricalColors"][0]
    assert config.DASHBOARD_COLORS["series_2"] == theme["chartCategoricalColors"][1]


# ---------------------------------------------------------------------------
# Charts (src/dashboard_charts.py): data from the tables, colours from config
# ---------------------------------------------------------------------------


def figure_colours(fig) -> set[str]:
    """Every colour a figure sets on its marks, lines, reference lines and annotations."""
    found = set()
    for trace in fig.data:
        for value in (getattr(getattr(trace, "marker", None), "color", None),
                      getattr(getattr(trace, "line", None), "color", None)):
            if isinstance(value, str):
                found.add(value)
            elif value is not None:
                found.update(v for v in value if isinstance(v, str))
    for shape in fig.layout.shapes:
        if shape.line.color:
            found.add(shape.line.color)
    for note in fig.layout.annotations:
        if note.font.color:
            found.add(note.font.color)
    return found


CV = pd.DataFrame({
    "model": [config.CV_MODEL_MAIN, config.CV_MODEL_LEAKAGE_FULL, config.CV_MODEL_LEAKAGE_ABLATION],
    "fold": [config.CV_MEAN_ROW] * 3, "auc": [0.67, 0.99, 1.0],
})


def test_auc_comparison_shows_the_mean_auc_of_all_three_models():
    fig = charts.auc_comparison(CV)
    assert list(fig.data[0].x) == [0.67, 0.99, 1.0]
    assert list(fig.data[0].y) == [config.CV_MODEL_LABELS[m] for m in CV["model"]]


def test_coefficient_chart_splits_raises_and_lowers_the_pd():
    coef = pd.DataFrame({"feature": ["a", "b", "c"], "coefficient": [0.5, -0.2, 1.1],
                         "odds_ratio": [1.6, 0.8, 3.0], "sm_p_value": [0.01, 1e-9, 0.2],
                         "cv_sign_share": [1.0, 1.0, 0.8]})
    fig = charts.coefficients(coef)
    raises, lowers = fig.data
    assert set(raises.y) == {"a", "c"} and set(lowers.y) == {"b"}
    assert list(fig.layout.yaxis.categoryarray) == ["b", "a", "c"]  # sorted by size


def test_segment_chart_has_one_pair_of_dots_per_level():
    seg = pd.DataFrame({"variable": ["loan_type"] * 2 + ["loan_purpose"], "level": ["t1", "t2", "p1"],
                        "n_loans": [10, 20, 30], "mean_pd": [0.1, 0.2, 0.3],
                        "observed_rate": [0.13, 0.18, 0.31], "gap": [0.03, -0.02, 0.01], "auc": [0.6, 0.7, 0.65]})
    fig = charts.segments(seg, "loan_type")
    _, mean_pd, observed = fig.data
    assert list(mean_pd.y) == ["t2", "t1"] and list(observed.x) == [0.18, 0.13]  # sorted by gap
    assert [t for t in observed.text if t] == ["+3.0 pp"]  # the largest gap, from the gap column


OOF = pd.DataFrame({"sample": [config.GRADE_OOF_POOLED] * 3 + ["fold_1"] * 3, "grade": ["A", "B", "C"] * 2,
                    "observed_rate": [0.061, 0.181, 0.551, 0.9, 0.9, 0.9]})
SHARES = SCALE.assign(share=[0.5, 0.3, 0.2])


def test_grade_rates_show_the_pooled_out_of_fold_rates_and_mark_the_selected_grade():
    fig = charts.grade_rates(SHARES, OOF, selected="B")
    grade_pd, _, out_of_fold = fig.data
    assert list(out_of_fold.y) == [0.061, 0.181, 0.551]  # pooled rows only
    assert list(grade_pd.marker.size) == [config.CHART_MARKER_SIZE, config.CHART_SELECTED_MARKER_SIZE,
                                         config.CHART_MARKER_SIZE]
    (band,) = fig.layout.shapes
    assert (band.x0, band.x1) == (0.5, 1.5)  # behind grade B, the second category


def test_grade_shares_show_each_grade_share_and_label_only_the_first():
    fig = charts.grade_shares(SHARES, selected="C")
    assert list(fig.data[0].y) == [0.5, 0.3, 0.2]
    assert [t for t in fig.data[0].text if t] == [dashboard.format_rate(0.5)]
    assert (fig.layout.shapes[0].x0, fig.layout.shapes[0].x1) == (1.5, 2.5)


def test_feature_decile_chart_labels_only_the_largest_gap():
    deciles = pd.DataFrame({"variable": ["loan_amount"] * 3, "bin": [1, 2, 3], "bin_min": [1, 2, 3],
                            "bin_max": [2, 3, 4], "n_loans": [10, 10, 10], "mean_pd": [0.2, 0.15, 0.1],
                            "observed_rate": [0.21, 0.15, 0.135], "gap": [0.01, 0.0, 0.035]})
    mean_pd, observed = charts.feature_deciles(deciles, "loan_amount").data
    assert list(mean_pd.y) == [0.2, 0.15, 0.1] and list(observed.y) == [0.21, 0.15, 0.135]
    assert list(observed.text) == ["", "", "+3.5 pp"]


def test_bin_share_chart_shows_the_chosen_variable_in_bin_order():
    bins = pd.DataFrame({"variable": ["x", "x", "y"], "bin_order": [2, 1, 1], "bin_label": ["b2", "b1", "c1"],
                         "share_baseline": [0.6, 0.4, 1.0], "share_monitored": [0.5, 0.5, 1.0]})
    fig = charts.bin_shares(bins, "x")
    assert list(fig.data[0].x) == ["b1", "b2"] and list(fig.data[1].y) == [0.5, 0.5]


@pytest.mark.parametrize("build", [
    lambda: charts.auc_comparison(CV),
    lambda: charts.grade_rates(SHARES, OOF, selected="A"),
    lambda: charts.grade_shares(SHARES, selected="A"),
    lambda: charts.calibration(pd.DataFrame({"bin": [1, 2], "n_loans": [5, 5], "mean_pd": [0.1, 0.2],
                                             "observed_rate": [0.12, 0.19]})),
    lambda: charts.psi(PSI_TABLE),
])
def test_charts_use_only_the_configured_dark_colours(build):
    assert figure_colours(build()) <= set(config.DASHBOARD_COLORS.values())


# ---------------------------------------------------------------------------
# Committed artifacts: tiles, Overview and charts read the CSVs
# ---------------------------------------------------------------------------

needs_artifacts = pytest.mark.skipif(
    bool(dashboard.missing_artifacts()), reason="the committed artifacts/*.csv tables are absent"
)
DASHBOARD_PAGES = list(config.DASHBOARD_PAGES.values())


@needs_artifacts
@pytest.mark.parametrize("name", list(config.DASHBOARD_ARTIFACT_COLUMNS))
def test_committed_artifact_has_the_columns_the_dashboard_uses(name):
    assert len(dashboard.load_artifact(name)) > 0


@needs_artifacts
def test_charts_have_a_readable_name_for_every_field_they_show():
    variables = set(config.MAIN_MODEL_FEATURES) | set(config.LEAKAGE_CHART_FIELDS) | set(config.MISS_VIEWS)
    variables |= set(read_csv("monitoring_psi.csv")["variable"]) | {config.LEAKAGE_CHART_CATEGORY[0]}
    assert sorted(variables - set(config.VARIABLE_LABELS)) == []
    assert sorted(set(read_csv("model_coefficients.csv")["feature"]) - set(config.COEFFICIENT_LABELS)) == []


@needs_artifacts
def test_every_column_the_dashboard_shows_has_a_plain_label():
    shown_columns = {"field", "reason", "rule"}  # tables built in app.py
    for name in config.DASHBOARD_ARTIFACT_COLUMNS:
        shown_columns |= set(dashboard.load_artifact(name).columns)
    assert sorted(shown_columns - set(config.COLUMN_LABELS)) == []


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
def test_page_tiles_show_the_values_in_the_committed_csvs():
    # every tile value is the CSV value, only formatted: nothing is computed anew (D-030)
    def values(page):
        return [t.value for t in dashboard.page_tiles(page)]

    cv = read_csv("model_cv_metrics.csv")
    cv_mean = cv[cv["fold"] == "mean"].set_index("model")["auc"]
    assert values("leakage") == [f"{cv_mean[m]:.3f}" for m in
                                 ("main", "leakage_full", "leakage_ablation_indicators_only")]

    sens = read_csv("model_sensitivity.csv")
    sens_mean = sens[sens["fold"] == "mean"].set_index("model")["auc"]
    assert values("model") == [str(len(config.MAIN_MODEL_FEATURES)), f"{cv_mean['main']:.3f}",
                               f"{sens_mean['without_lump_sum_payment']:.3f}"]

    metrics = read_csv("validation_metrics.csv").set_index("population").loc["holdout_in_scope"]
    calibration = read_csv("validation_calibration.csv").set_index("population").loc["holdout_in_scope"]
    criteria = read_csv("validation_criteria.csv")["status"]
    assert values("validation") == [
        *[f"{metrics[m]:.3f}" for m in ("auc", "gini", "ks", "brier")],
        f"{calibration['mean_pd']:.1%}", f"{calibration['observed_rate']:.1%}",
        f"{(criteria == 'green').sum()} of {len(criteria)}",
    ]

    scale = read_csv("grades_scale.csv")
    assert values("grades") == [str(len(scale)), f"{scale['share'].iloc[0]:.1%}",
                                f"{scale['observed_rate'].iloc[-1]:.1%}"]

    psi = read_csv("monitoring_psi.csv")
    top = psi.loc[psi["psi"].idxmax()]
    watch = read_csv("monitoring_watch_list.csv")["light"]
    mscope = read_csv("monitoring_scope.csv").set_index("sample")["out_of_scope_share"]
    assert values("monitoring") == [
        f"{top['psi']:.4f}",
        f"{(watch == 'green').sum()} of {len(watch)}", f"{mscope['holdout']:.1%}",
    ]
    assert values("overview") == []  # the Overview shows the pipeline and findings instead


@needs_artifacts
def test_every_tile_has_a_label_and_help_text():
    for page in config.DASHBOARD_PAGES:
        for tile in dashboard.page_tiles(page):
            assert tile.label and tile.help, tile


@needs_artifacts
def test_overview_pipeline_and_findings_show_the_csv_values():
    dq = read_csv("dq_summary.csv").set_index("check")["value"]
    split = read_csv("sql_split_summary.csv").set_index("sample")["share_loans"]
    scope = read_csv("validation_scope.csv").set_index(["sample", "scope"])["n_loans"]
    cv = read_csv("model_cv_metrics.csv")
    cv_mean = cv[cv["fold"] == "mean"].set_index("model")["auc"]
    holdout_auc = read_csv("validation_metrics.csv").set_index("population").loc["holdout_in_scope", "auc"]
    calibration = read_csv("validation_calibration.csv").set_index("population").loc["holdout_in_scope"]
    scale = read_csv("grades_scale.csv")
    psi_table = read_csv("monitoring_psi.csv")
    top = psi_table.loc[psi_table["psi"].idxmax()]
    psi = top["psi"]

    steps = dashboard.overview_pipeline()
    assert [s.label for s in steps] == [label for label, _ in config.PIPELINE_STEPS.values()]
    assert [s.value for s in steps] == [
        f"{int(dq['n_rows']):,}", f"{split['development']:.0%} / {split['holdout']:.0%}",
        f"{int(scope['development', 'in_scope']):,}", f"{cv_mean['main']:.3f}", f"{holdout_auc:.3f}",
        str(len(scale)), f"{psi:.4f} {config.LIGHT_ICONS[top['light']]}",
    ]

    findings = " ".join(f.text for f in dashboard.overview_findings())
    for value in (f"{cv_mean['leakage_ablation_indicators_only']:.3f}", f"{cv_mean['main']:.3f}",
                  f"{holdout_auc:.3f}", f"{calibration['mean_pd']:.1%}", f"{calibration['observed_rate']:.1%}",
                  f"{scale['share'].iloc[0]:.1%}", f"{psi:.4f}"):
        assert value in findings, value


@needs_artifacts
def test_leakage_chart_rows_are_the_figure_05_fields_and_the_equi_rate():
    rows = dashboard.leakage_chart_rows()
    missing = read_csv("sql_dq_missingness.csv").set_index("column_name")
    for field, row in zip(config.LEAKAGE_CHART_FIELDS, rows.itertuples()):
        assert row.rate_if_true == missing.loc[field, "default_rate_if_missing"]
        assert row.rate_otherwise == missing.loc[field, "default_rate_if_present"]
    levels = read_csv("dq_categorical_levels.csv")
    column, level = config.LEAKAGE_CHART_CATEGORY
    equi = levels[(levels["column"] == column) & (levels["level"] == level)].iloc[0]
    last = rows.iloc[-1]
    assert last["rate_if_true"] == equi["default_rate"] and np.isnan(last["rate_otherwise"])
    assert len(rows) == len(config.LEAKAGE_CHART_FIELDS) + 1


@needs_artifacts
def test_coefficient_chart_shows_every_committed_coefficient():
    coef = read_csv("model_coefficients.csv")
    fig = charts.coefficients(coef)
    shown = {y: x for trace in fig.data for x, y in zip(trace.x, trace.y)}
    assert shown == {config.COEFFICIENT_LABELS[f]: c for f, c in zip(coef["feature"], coef["coefficient"])}


# ---------------------------------------------------------------------------
# The app itself (Streamlit AppTest)
# ---------------------------------------------------------------------------


def open_page(page: str):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=60).run()
    return app.sidebar.radio[0].set_value(config.DASHBOARD_PAGES[page]).run()


def chart_specs(app) -> list[dict]:
    """The page's Plotly figure specs, as sent to the browser."""
    return [json.loads(c.proto.spec) for c in app.get("plotly_chart")]


def values(array) -> list:
    """A trace's x or y as a list. Plotly sends numeric arrays as base64 bytes plus their dtype."""
    if isinstance(array, dict) and "bdata" in array:
        return np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"]).tolist()
    return list(array)


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
@pytest.mark.parametrize("page", list(config.DASHBOARD_PAGES))
def test_each_page_follows_the_layout(page):
    # D-030 layout: title without the number and a subtitle, a chart whose title carries the
    # "how to read it" help, the tiles from page_tiles, the page's caveat, plain column
    # headers, no PNG image, and the decision-log links at the end
    app = open_page(page)
    assert not app.exception
    assert app.title[0].value == dashboard.page_title(page)
    assert dashboard_text.PAGES[page]["subtitle"] in [c.value for c in app.caption]
    assert app.get("plotly_chart") and not app.image
    chart_titles = {title for title, _ in dashboard_text.PAGES[page]["charts"].values()}
    assert chart_titles <= {s.value for s in app.subheader}

    if page in dashboard_text.CAVEATS:
        box, message = dashboard_text.CAVEATS[page]
        assert message in [b.value for b in (app.warning if box == "warning" else app.info)]

    if page != "overview":
        assert [(m.label, m.value) for m in app.metric] == [(t.label, t.value) for t in dashboard.page_tiles(page)]
        assert dashboard_text.PANEL_CAPTIONS[page] in [c.value for c in app.caption]

    for df in app.dataframe:
        assert not [c for c in df.value.columns if "_" in str(c)], "snake_case header left"

    text = " ".join(m.value for m in app.markdown)
    assert "Decision log" in text
    for _, url in dashboard.decision_links(config.DASHBOARD_PAGE_DECISIONS[page]):
        assert url in text


@needs_artifacts
def test_overview_shows_the_pipeline_findings_and_model_card():
    app = open_page("overview")
    text = " ".join(m.value for m in app.markdown)
    for step in dashboard.overview_pipeline():
        assert f"#### {step.value}" in text and step.label in text
    assert ":material/arrow_forward:" not in text  # the pipeline has no arrow elements
    assert not app.toggle  # the hero chart always shows all three models
    for finding in dashboard.overview_findings():
        assert finding.title in text and finding.text in text
    assert "**Not for:** credit decisions." in text
    assert dashboard_text.MODEL_CARD["Limitations"] in text


@needs_artifacts
def test_grade_slider_selects_the_grade_of_the_chosen_pd():
    scale = dashboard.load_artifact(dashboard.GRADE_SCALE_FILE)
    grade_list = scale["grade"].tolist()
    app = open_page("grades")
    for pd_percent in (0.0, float(scale["pd_lower"].iloc[3] * 100) + 0.1, 100.0):
        app.slider(key="grade_pd").set_value(pd_percent).run()
        expected = dashboard.grade_for_pd(pd_percent / 100, scale)["grade"]
        assert f"{dashboard_text.SELECTED_GRADE}: {expected}" in " ".join(m.value for m in app.markdown)
        i = grade_list.index(expected)
        for spec in chart_specs(app):  # the band sits behind the selected grade in both charts
            assert [(shape["x0"], shape["x1"]) for shape in spec["layout"]["shapes"]] == [(i - 0.5, i + 0.5)]


@needs_artifacts
def test_miss_and_bin_selectors_redraw_with_the_chosen_view():
    app = open_page("validation")
    deciles = dashboard.load_artifact("validation_feature_deciles.csv")
    first = next(iter(config.MISS_VIEWS))  # the default view: feature deciles
    expected = deciles[deciles["variable"] == first].sort_values("bin")
    assert values(chart_specs(app)[1]["data"][1]["y"]) == expected["observed_rate"].tolist()

    segments = dashboard.load_artifact("validation_segments.csv")
    app.segmented_control(key="miss_view").set_value("loan_type").run()
    shown = values(chart_specs(app)[1]["data"][1]["y"])
    assert shown == segments[segments["variable"] == "loan_type"].sort_values("gap")["level"].tolist()

    bins = dashboard.load_artifact("monitoring_psi_bins.csv")
    app = open_page("monitoring")
    app.selectbox(key="bin_variable").set_value("loan_type").run()
    expected = bins[bins["variable"] == "loan_type"].sort_values("bin_order")["bin_label"].astype(str).tolist()
    assert [str(x) for x in values(chart_specs(app)[1]["data"][0]["x"])] == expected


@needs_artifacts
def test_next_button_opens_the_following_page():
    app = open_page("overview")
    app.button(key="next_overview").click().run()
    assert app.sidebar.radio[0].value == config.DASHBOARD_PAGES["leakage"]
    assert app.title[0].value == dashboard.page_title("leakage")


@needs_artifacts
def test_last_page_offers_a_way_back_to_the_first():
    app = open_page("monitoring")
    app.button(key="back_monitoring").click().run()
    assert app.sidebar.radio[0].value == config.DASHBOARD_PAGES["overview"]


@needs_artifacts
def test_pages_keep_the_reading_notes():
    app = open_page("validation")
    captions = [c.value for c in app.caption]
    assert dashboard_text.HOSMER_LEMESHOW_CAPTION in captions
    assert dashboard_text.EQUI_CAPTION["validation"] in captions

    app = open_page("leakage")
    captions = [c.value for c in app.caption]
    assert dashboard_text.EQUI_CAPTION["leakage"] in captions
    assert "income = 0" in dashboard_text.PAGES["leakage"]["charts"]["indicators"][1]  # in the chart help

    app = open_page("model")
    assert dashboard_text.LOAN_AMOUNT_CAPTION in [c.value for c in app.caption]
    shown = [df.value for df in app.dataframe if "Model" in df.value.columns]
    expected_models = {config.CV_MODEL_MAIN, *[f"without_{f}" for f in config.SENSITIVITY_DROPPED_FEATURES]}
    assert any(set(t["Model"]) == expected_models for t in shown)  # the sensitivity table is on the page


@needs_artifacts
def test_monitoring_page_explains_the_backtest_lights():
    app = open_page("monitoring")
    assert not app.exception
    assert dashboard.backtest_light_rules() in [c.value for c in app.caption]
