"""Stage 9: helpers for the Streamlit dashboard (app.py; decision log D-029).

The dashboard is read-only. It shows the tables that Stages 2-8 already wrote to
artifacts/, and never refits a model or computes a new metric. The one live
calculation is the optional "score a loan" page, which uses the frozen in-scope
model when it is present (it is gitignored, so it is missing on GitHub).

Nothing here imports Streamlit, so every function can be unit-tested.

    load_artifact            one committed CSV, with its required columns checked
    check_grade_scale        the grade scale covers [0, 1] once, with no gaps or overlaps
    missing_artifacts        required CSVs that are absent
    criterion_rules          the D-025 thresholds as readable text, from config
    backtest_light_rules     caption for the Stage 8 grade-backtest lights, from config
    scoring_switched_on      the D-030 switch: scoring is off unless it is explicitly turned on
    load_model_if_available  the frozen model, or None if it has not been built
    check_model_coefficients the loaded model is the committed in-scope model
    model_levels             the categorical levels the model was trained on
    build_loan_frame         one validated loan, ready for holdout.score
    out_of_range             inputs outside the in-scope development range (D-029)
    psi_axis_max             PSI chart axis end: the bands and the largest PSI always fit
    grade_for_pd             the D-027 grade of a PD, through grades.assign_grade
"""

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src import grades
from src import holdout
from src import model

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


# ---------------------------------------------------------------------------
# Scoring one loan with the frozen model
# ---------------------------------------------------------------------------


def scoring_switched_on(environ: Mapping[str, str] = os.environ) -> bool:
    """True only if the scoring switch is explicitly on (config.SCORING_ENV_VAR equals
    config.SCORING_ENV_ON). Off by default, so a public copy never scores (D-030)."""
    return environ.get(config.SCORING_ENV_VAR) == config.SCORING_ENV_ON


def load_model_if_available(path: Path = config.MODEL_PATH) -> Pipeline | None:
    """The frozen in-scope model, or None if `python -m src.model` has not been run."""
    if not Path(path).exists():
        return None
    return holdout.load_frozen_model(path)


def check_model_coefficients(pipeline: Pipeline, coefficients: pd.DataFrame) -> None:
    """Raise unless the loaded model's coefficients equal the committed in-scope ones
    (model_coefficients.csv), feature by feature. Catches a stale or pre-D-026 binary
    that has the same feature names (D-029)."""
    fitted = pd.Series(pipeline["model"].coef_.ravel(), index=model.feature_names(pipeline))
    committed = coefficients.set_index("feature")["coefficient"]
    if set(fitted.index) != set(committed.index):
        raise ValueError(f"features differ: {sorted(set(fitted.index) ^ set(committed.index))}")
    committed = committed.reindex(fitted.index)
    if not np.allclose(fitted, committed, rtol=config.MODEL_COEFFICIENT_RTOL, atol=0):
        worst = (fitted - committed).abs().idxmax()
        raise ValueError(f"coefficient of {worst} is {fitted[worst]:.6g}, committed {committed[worst]:.6g}")


def model_levels(pipeline: Pipeline) -> dict[str, list[str]]:
    """Levels per categorical feature that the fitted one-hot encoder knows.
    Only these can be scored: the encoder raises on any other level (D-023)."""
    encoder = pipeline["preprocess"].named_transformers_["cat"]["onehot"]
    return {
        col: [str(level) for level in levels]
        for col, levels in zip(config.CATEGORICAL_FEATURES, encoder.categories_)
    }


def build_loan_frame(
    loan_amount: float,
    income: float | None,
    levels: dict[str, str],
    credit_type: str,
    known_levels: dict[str, list[str]],
) -> pd.DataFrame:
    """One loan as a row of MAIN_MODEL_FEATURES, after the D-029 checks.

    Raises ValueError for an EQUI loan (outside the model's scope, D-026), a
    non-positive loan amount, a non-positive income (D-008) or a level the model
    has not seen. `income=None` means "not provided" and is scored as missing,
    as the model does for missing income (D-023).
    """
    if credit_type == config.EQUI_LEVEL:
        raise ValueError(f"credit_type = {config.EQUI_LEVEL} is outside the model's scope (D-026).")
    if not loan_amount > 0:
        raise ValueError("loan_amount must be greater than 0.")
    if income is not None and not income > 0:
        raise ValueError("income must be greater than 0, or left blank (D-008).")
    for col in config.CATEGORICAL_FEATURES:
        if levels.get(col) not in known_levels[col]:
            raise ValueError(f"{col} = {levels.get(col)!r} was not seen in model development (D-023).")

    row = {
        config.INCOME_CLEAN_COL: np.nan if income is None else float(income),
        "loan_amount": float(loan_amount),
        **{col: levels[col] for col in config.CATEGORICAL_FEATURES},
    }
    return pd.DataFrame([row], columns=config.MAIN_MODEL_FEATURES)


def out_of_range(loan_amount: float, income: float | None, ranges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """The inputs that lie outside [lower, upper] of the in-scope development loans
    (sql_model_input_ranges.csv, D-029), with that range. Blank income is never flagged."""
    values = {"loan_amount": loan_amount, config.INCOME_CLEAN_COL: income}
    bounds = ranges.set_index("variable")
    flagged = {}
    for variable in config.INPUT_RANGE_VARIABLES:
        value, lower, upper = values[variable], bounds.loc[variable, "lower"], bounds.loc[variable, "upper"]
        if value is not None and not lower <= value <= upper:
            flagged[variable] = (float(lower), float(upper))
    return flagged


def psi_axis_max(psi_values) -> float:
    """End of the PSI chart's x-axis: far enough to show the amber limit, and never
    shorter than the largest PSI, so a red result is never cut off the chart."""
    amber = config.PSI_THRESHOLDS[1]
    largest = float(np.nanmax(np.asarray(psi_values, dtype=float)))
    return max(amber * config.PSI_CHART_MIN_AXIS, largest * config.PSI_CHART_HEADROOM)


def grade_for_pd(pd_value: float, scale: pd.DataFrame) -> pd.Series:
    """The grade_scale row of `pd_value`, assigned by grades.assign_grade: the same rule
    as Stage 7 and sql/grade_assignment.sql (pd_lower <= pd < pd_upper). A PD outside
    [0, 1] or NaN raises ValueError instead of being graded."""
    edges = scale["pd_lower"].iloc[1:].tolist()
    return scale.iloc[int(grades.assign_grade([pd_value], edges)[0])]
