"""Stage 9: helpers for the Streamlit dashboard (app.py; decision log D-029, D-030).

The dashboard is read-only. It shows the tables that Stages 2-8 already wrote to
artifacts/, and never refits a model or computes a new metric. It loads no model:
the loan-scoring page was removed (D-030). The grade slider is a lookup in the
committed grade scale.

Nothing here imports Streamlit, so every function can be unit-tested.

    load_artifact            one committed CSV, with its required columns checked
    check_grade_scale        the grade scale covers [0, 1] once, with no gaps or overlaps
    missing_artifacts        required CSVs that are absent
    criterion_rules          the D-025 thresholds as readable text, from config
    backtest_light_rules     caption for the Stage 8 grade-backtest lights, from config
    grade_for_pd             the D-027 grade of a PD, through grades.assign_grade

Presentation (D-030):

    page_title / next_page   page name without its number; the page after a page
    format_*                 display formats for counts, rates, metrics and p-values
    readable_table           a table with plain column labels, p-values and lights as text
    column_formats           number format per (labelled) column: percent, count or decimal
    Tile / page_tiles        the headline numbers of a page, read from the committed CSVs
    overview_pipeline        the Overview's pipeline steps, one number each
    overview_findings        the Overview's key findings, filled from the CSVs
    largest_psi              the monitoring_psi.csv row with the largest PSI
    leakage_chart_rows       the rows of the Data leakage chart (figure 05 fields + EQUI)
    github_anchor            the anchor GitHub gives a Markdown heading
    decision_links           (heading, URL) of decision-log entries on GitHub
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src import dashboard_text
from src import grades
from src import holdout

# ---------------------------------------------------------------------------
# Committed artifacts
# ---------------------------------------------------------------------------

GRADE_SCALE_FILE = "grades_scale.csv"  # checked by check_grade_scale whenever it is loaded


def load_artifact(name: str, directory: Path = config.ARTIFACTS_DIR) -> pd.DataFrame:
    """Read `directory/name` and check it has the columns the dashboard uses."""
    path = Path(directory) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact {path}. Rebuild it with the stage that writes it.")
    table = pd.read_csv(path)
    missing = [c for c in config.DASHBOARD_ARTIFACT_COLUMNS.get(name, []) if c not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing the columns {missing}.")
    if name == GRADE_SCALE_FILE:
        check_grade_scale(table)
    return table


def check_grade_scale(scale: pd.DataFrame) -> None:
    """Raise unless the scale runs from PD_MIN to PD_MAX with each grade starting
    exactly where the one below ends (no gap, no overlap), rising boundaries and
    unique labels. Boundaries are rounded once in Stage 7, so equality is exact (D-027)."""
    lower, upper = scale["pd_lower"].to_numpy(float), scale["pd_upper"].to_numpy(float)
    problems = []
    if lower[0] != config.PD_MIN or upper[-1] != config.PD_MAX:
        problems.append(f"it must run from {config.PD_MIN} to {config.PD_MAX}")
    if (upper[:-1] != lower[1:]).any():
        problems.append("each grade must start where the one below ends (gap or overlap found)")
    if not (lower[1:] > lower[:-1]).all() or not (upper > lower).all():
        problems.append("boundaries must rise from grade to grade")
    if scale["grade"].duplicated().any():
        problems.append("grade labels must be unique")
    if problems:
        raise ValueError("Invalid grade scale: " + "; ".join(problems) + ".")


def missing_artifacts(directory: Path = config.ARTIFACTS_DIR) -> list[str]:
    """Names of the dashboard's CSVs that are not in `directory`."""
    return [name for name in config.DASHBOARD_ARTIFACT_COLUMNS if not (Path(directory) / name).exists()]


def criterion_rules() -> dict[str, str]:
    """The pre-set D-025 criteria as text, keyed like validation_criteria.csv."""
    auc_g, auc_a = config.CRITERION_AUC_DROP
    p_g, p_a = config.CRITERION_BINOMIAL_P
    gap_g, gap_a = config.CRITERION_DECILE_GAP
    (s_lo, s_hi), (a_lo, a_hi) = config.CRITERION_SLOPE_GREEN, config.CRITERION_SLOPE_AMBER
    return {
        "auc_drop_cv_minus_holdout": f"green <= {auc_g}, amber <= {auc_a}",
        "binomial_p": f"green >= {p_g}, amber >= {p_a}",
        "calibration_slope": f"green {s_lo}-{s_hi}, amber {a_lo}-{a_hi}",
        "max_decile_gap": f"green <= {gap_g * 100:g} pp, amber <= {gap_a * 100:g} pp",
    }


def backtest_light_rules() -> str:
    """Caption for the grade-backtest lights (monitoring.grade_backtest): the D-025
    thresholds reused by D-028, with the gap in percentage points. Limits are inclusive."""
    gap_g, gap_a = config.CRITERION_DECILE_GAP
    p_g, p_a = config.CRITERION_BINOMIAL_P
    return (
        "Lights use the D-025 heuristic thresholds, reused by D-028. "
        f"Gap = |observed − grade PD|: green ≤ {gap_g * 100:g} pp, amber ≤ {gap_a * 100:g} pp, red above. "
        f"Binomial test p-value: green ≥ {p_g:g}, amber ≥ {p_a:g}, red below. "
        "In-sample these lights are a reference, not a judgement: the grade PD was set on the same loans."
    )


def grade_for_pd(pd_value: float, scale: pd.DataFrame) -> pd.Series:
    """The grade_scale row of `pd_value`, assigned by grades.assign_grade: the same rule
    as Stage 7 and sql/grade_assignment.sql (pd_lower <= pd < pd_upper). A PD outside
    [0, 1] or NaN raises ValueError instead of being graded."""
    edges = scale["pd_lower"].iloc[1:].tolist()
    return scale.iloc[int(grades.assign_grade([pd_value], edges)[0])]


# ---------------------------------------------------------------------------
# Presentation (D-030): page names, formats, tables, tiles and links
# ---------------------------------------------------------------------------


def page_title(page: str) -> str:
    """The sidebar name of `page` without its number: "3. Model" -> "Model"."""
    return config.DASHBOARD_PAGES[page].split(". ", 1)[1]


def next_page(page: str) -> str | None:
    """The page after `page` in reading order, or None for the last page."""
    pages = list(config.DASHBOARD_PAGES)
    position = pages.index(page)
    return pages[position + 1] if position + 1 < len(pages) else None


def format_count(value: float) -> str:
    return f"{int(round(float(value))):,}"


def format_rate(value: float) -> str:
    return f"{float(value):.{config.DISPLAY_RATE_DECIMALS}%}"


def format_metric(value: float) -> str:
    return f"{float(value):.{config.DISPLAY_METRIC_DECIMALS}f}"


def format_p_value(value: float) -> str:
    """'< 0.001' below the display floor, otherwise rounded; blank when missing."""
    if pd.isna(value):
        return ""
    if value < config.DISPLAY_P_VALUE_FLOOR:
        return f"< {config.DISPLAY_P_VALUE_FLOOR:g}"
    return f"{value:.{config.DISPLAY_P_VALUE_DECIMALS}f}"


def readable_table(table: pd.DataFrame) -> pd.DataFrame:
    """A copy for display: p-values as text, lights as icon + word, and plain column
    labels (config.COLUMN_LABELS). The values themselves are not changed."""
    shown = table.copy()
    for col in shown.columns:
        if col in config.P_VALUE_COLUMNS:
            shown[col] = shown[col].map(format_p_value)
        elif col in config.LIGHT_COLUMNS:
            shown[col] = shown[col].map(lambda v: config.LIGHT_LABELS.get(v, v))
    return shown.rename(columns=config.COLUMN_LABELS)


def column_formats(table: pd.DataFrame) -> dict[str, str]:
    """Number format of each numeric column, keyed by its plain label:
    'percent' (stored as a fraction), 'count' or 'decimal'."""
    formats = {}
    for col in table.columns:
        label = config.COLUMN_LABELS.get(col, col)
        if col in config.PERCENT_COLUMNS:
            formats[label] = "percent"
        elif col in config.COUNT_COLUMNS:
            formats[label] = "count"
        elif col not in config.P_VALUE_COLUMNS and pd.api.types.is_float_dtype(table[col]):
            formats[label] = "decimal"
    return formats


@dataclass(frozen=True)
class Tile:
    """One headline number: a short label, the formatted value and its help text."""
    label: str
    value: str
    help: str


def _tile(tile_id: str, value: str, **label_parts: str) -> Tile:
    label = config.TILE_LABELS[tile_id].format(**label_parts)
    return Tile(label, value, dashboard_text.TILE_HELP[tile_id])


def _one_row(table: pd.DataFrame, **match) -> pd.Series:
    """The single row whose columns equal `match`; raises if there is not exactly one."""
    mask = np.ones(len(table), dtype=bool)
    for col, value in match.items():
        mask &= (table[col].astype(str) == str(value)).to_numpy()
    rows = table[mask]
    if len(rows) != 1:
        raise ValueError(f"expected one row with {match}, found {len(rows)}")
    return rows.iloc[0]


def _green_count(lights: pd.Series) -> str:
    return f"{int((lights == 'green').sum())} of {len(lights)}"


def page_tiles(page: str, load: Callable[[str], pd.DataFrame] = load_artifact) -> list[Tile]:
    """The headline tiles of `page`. Every value is read from a committed CSV through
    `load` (or from config), never computed anew (D-029, D-030)."""
    if page == "leakage":
        cv = load("model_cv_metrics.csv")
        auc = {name: format_metric(_one_row(cv, model=name, fold=config.CV_MEAN_ROW)["auc"])
               for name in (config.CV_MODEL_MAIN, config.CV_MODEL_LEAKAGE_FULL, config.CV_MODEL_LEAKAGE_ABLATION)}
        return [
            _tile("cv_auc_main", auc[config.CV_MODEL_MAIN]),
            _tile("cv_auc_leakage_full", auc[config.CV_MODEL_LEAKAGE_FULL]),
            _tile("cv_auc_leakage_ablation", auc[config.CV_MODEL_LEAKAGE_ABLATION]),
        ]
    if page == "model":
        main = _one_row(load("model_cv_metrics.csv"), model=config.CV_MODEL_MAIN, fold=config.CV_MEAN_ROW)
        dropped = config.SENSITIVITY_DROPPED_FEATURES
        without = _one_row(load("model_sensitivity.csv"), model="without_" + "_".join(dropped),
                           fold=config.CV_MEAN_ROW)
        dropped = [config.VARIABLE_LABELS.get(f, f) for f in dropped]
        return [
            _tile("n_features", str(len(config.MAIN_MODEL_FEATURES))),
            _tile("cv_auc", format_metric(main["auc"])),
            _tile("cv_auc_without", format_metric(without["auc"]), fields=", ".join(dropped)),
        ]
    if page == "validation":
        metrics = _one_row(load("validation_metrics.csv"), population=holdout.HOLDOUT_POPULATION)
        calibration = _one_row(load("validation_calibration.csv"), population=holdout.HOLDOUT_POPULATION)
        return [
            _tile("holdout_auc", format_metric(metrics["auc"])),
            _tile("holdout_gini", format_metric(metrics["gini"])),
            _tile("holdout_ks", format_metric(metrics["ks"])),
            _tile("holdout_brier", format_metric(metrics["brier"])),
            _tile("holdout_mean_pd", format_rate(calibration["mean_pd"])),
            _tile("holdout_observed", format_rate(calibration["observed_rate"])),
            _tile("criteria_green", _green_count(load("validation_criteria.csv")["status"])),
        ]
    if page == "grades":
        scale = load(GRADE_SCALE_FILE)
        first, top = scale.iloc[0], scale.iloc[-1]
        return [
            _tile("n_grades", str(len(scale))),
            _tile("first_grade_share", format_rate(first["share"]), grade=str(first["grade"])),
            _tile("top_grade_rate", format_rate(top["observed_rate"]), grade=str(top["grade"])),
        ]
    if page == "monitoring":
        psi = load("monitoring_psi.csv")
        largest = largest_psi(psi)
        hold = _one_row(load("monitoring_scope.csv"), sample="holdout")
        return [
            _tile("largest_psi", f"{largest['psi']:.{config.DISPLAY_PSI_DECIMALS}f}",
                  variable=config.VARIABLE_LABELS.get(largest["variable"], largest["variable"])),
            _tile("watch_green", _green_count(load("monitoring_watch_list.csv")["light"])),
            _tile("equi_share_holdout", format_rate(hold["out_of_scope_share"])),
        ]
    return []


def _cv_mean_auc(cv: pd.DataFrame, model_name: str) -> float:
    return float(_one_row(cv, model=model_name, fold=config.CV_MEAN_ROW)["auc"])


def largest_psi(psi: pd.DataFrame) -> pd.Series:
    """The monitoring_psi.csv row with the largest PSI (its variable, value and light)."""
    return psi.loc[psi["psi"].idxmax()]


@dataclass(frozen=True)
class Step:
    """One step of the Overview pipeline strip: label, icon, one number and its note."""
    label: str
    icon: str
    value: str
    note: str


def overview_pipeline(load: Callable[[str], pd.DataFrame] = load_artifact) -> list[Step]:
    """The pipeline steps of the Overview, each with one number read from a CSV."""
    dq = load("dq_summary.csv").set_index("check")["value"]
    split = load("sql_split_summary.csv").set_index("sample")["share_loans"]
    scope = _one_row(load("validation_scope.csv"), sample="development", scope="in_scope")
    cv = load("model_cv_metrics.csv")
    metrics = _one_row(load("validation_metrics.csv"), population=holdout.HOLDOUT_POPULATION)
    psi = largest_psi(load("monitoring_psi.csv"))
    values = {
        "raw": format_count(dq["n_rows"]),
        "split": " / ".join(f"{split[sample]:.{config.DISPLAY_SPLIT_DECIMALS}%}"
                            for sample in ("development", "holdout")),
        "scope": format_count(scope["n_loans"]),
        "model": format_metric(_cv_mean_auc(cv, config.CV_MODEL_MAIN)),
        "validation": format_metric(metrics["auc"]),
        "grades": str(len(load(GRADE_SCALE_FILE))),
        "monitoring": f"{psi['psi']:.{config.DISPLAY_PSI_DECIMALS}f} {config.LIGHT_ICONS[psi['light']]}",
    }
    return [Step(label, icon, values[step], dashboard_text.PIPELINE_NOTES[step])
            for step, (label, icon) in config.PIPELINE_STEPS.items()]


@dataclass(frozen=True)
class Finding:
    """One key finding on the Overview: a short title and one sentence with CSV values."""
    title: str
    text: str


def overview_findings(load: Callable[[str], pd.DataFrame] = load_artifact) -> list[Finding]:
    """The Overview's key findings: dashboard_text templates filled with CSV values."""
    cv = load("model_cv_metrics.csv")
    metrics = _one_row(load("validation_metrics.csv"), population=holdout.HOLDOUT_POPULATION)
    calibration = _one_row(load("validation_calibration.csv"), population=holdout.HOLDOUT_POPULATION)
    scale = load(GRADE_SCALE_FILE)
    psi = largest_psi(load("monitoring_psi.csv"))
    values = {
        "leakage": {"ablation": format_metric(_cv_mean_auc(cv, config.CV_MODEL_LEAKAGE_ABLATION)),
                    "main": format_metric(_cv_mean_auc(cv, config.CV_MODEL_MAIN))},
        "holdout": {"auc": format_metric(metrics["auc"]), "mean_pd": format_rate(calibration["mean_pd"]),
                    "observed": format_rate(calibration["observed_rate"])},
        "grades": {"n_grades": str(len(scale)), "first": str(scale["grade"].iloc[0]),
                   "share": format_rate(scale["share"].iloc[0])},
        "monitoring": {"psi": f"{psi['psi']:.{config.DISPLAY_PSI_DECIMALS}f}"},
    }
    return [Finding(title, template.format(**values[key]))
            for key, (title, template) in dashboard_text.FINDINGS.items()]


def leakage_chart_rows(load: Callable[[str], pd.DataFrame] = load_artifact) -> pd.DataFrame:
    """Rows of the Data leakage chart: for each figure-05 field (config.LEAKAGE_CHART_FIELDS)
    the default rate when it is missing and when it is present (sql_dq_missingness.csv),
    then the EQUI category's default rate (dq_categorical_levels.csv), which has no
    'otherwise' rate in a committed table."""
    missing = load("sql_dq_missingness.csv").set_index("column_name")
    rows = [{"label": f"{config.VARIABLE_LABELS[field]} missing", "rate_if_true": missing.loc[field, "default_rate_if_missing"],
             "rate_otherwise": missing.loc[field, "default_rate_if_present"],
             "n_loans": missing.loc[field, "n_missing"]}
            for field in config.LEAKAGE_CHART_FIELDS]
    column, level = config.LEAKAGE_CHART_CATEGORY
    category = _one_row(load("dq_categorical_levels.csv"), column=column, level=level)
    rows.append({"label": f"{config.VARIABLE_LABELS[column]} = {level}", "rate_if_true": category["default_rate"],
                 "rate_otherwise": np.nan, "n_loans": category["n"]})
    return pd.DataFrame(rows)


def github_anchor(heading: str) -> str:
    """The anchor GitHub gives a Markdown heading: lower case, punctuation other than
    '-' and '_' removed, spaces turned into hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def decision_headings(path: Path = config.DECISION_LOG_PATH) -> dict[str, str]:
    """Decision id -> its heading text ("D-017 Variables whose ..."), from the decision log."""
    headings = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^### (D-\d{3}) ", line)
        if match:
            headings[match.group(1)] = line[len("### "):].strip()
    return headings


def decision_links(ids: list[str], path: Path = config.DECISION_LOG_PATH) -> list[tuple[str, str]]:
    """(heading, GitHub URL) for each decision id, in the given order. Raises if an id
    has no heading in the decision log."""
    headings = decision_headings(path)
    missing = [i for i in ids if i not in headings]
    if missing:
        raise ValueError(f"no decision-log heading for {missing}")
    base = f"{config.REPO_URL}/blob/{config.DOCS_GIT_REF}/docs/decision_log.md"
    return [(headings[i], f"{base}#{github_anchor(headings[i])}") for i in ids]
