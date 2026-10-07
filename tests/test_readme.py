"""Check that the README's results table (section 12) matches the committed artifacts.

Every value in that table is typed by hand, so it can drift from the CSVs after a
rebuild. Each test recomputes one table cell from its source CSV, rounds it as the
README does, and compares the strings. Only committed artifacts/*.csv are read, so
these tests also run without the raw data (D-002).
"""

import re

import pandas as pd
import pytest

from src import config

RESULTS_HEADING = "## 12. Results"
NEXT_HEADING = "## 13."


def results_table(readme_text: str) -> dict[str, str]:
    """Return {measure: value} for the rows of the section 12 table."""
    section = readme_text.split(RESULTS_HEADING, 1)[1].split(NEXT_HEADING, 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] not in ("Measure", "---"):
            rows[cells[0]] = cells[1]
    return rows


def artifact(name: str) -> pd.DataFrame:
    """Read one committed artifact table."""
    return pd.read_csv(config.ARTIFACTS_DIR / name)


def value(table: pd.DataFrame, key_col: str, key: str, col: str) -> float:
    """Return the single value of `col` in the row where `key_col` equals `key`."""
    match = table.loc[table[key_col] == key, col]
    assert len(match) == 1, f"expected one row with {key_col} = {key}"
    return float(match.iloc[0])


def pct(x: float, decimals: int = 2) -> str:
    """0.246445 -> '24.64%'."""
    return f"{100 * x:.{decimals}f}%"


def num(x: float, decimals: int = 3) -> str:
    """0.674605 -> '0.675'."""
    return f"{x:.{decimals}f}"


def count(x: float) -> str:
    """148670 -> '148,670'."""
    return f"{int(x):,}"


def cv_mean(name: str, model: str, metric: str) -> float:
    """Mean-over-folds CV metric of one model."""
    table = artifact(name)
    table = table[table["fold"].astype(str) == config.CV_MEAN_ROW]
    return value(table, "model", model, metric)


def scope_rate(sample: str) -> float:
    """In-scope observed default rate of one sample."""
    table = artifact("validation_scope.csv")
    table = table[table["scope"] == "in_scope"]
    return value(table, "sample", sample, "observed_rate")


def scope_loans(sample: str) -> float:
    """In-scope loan count of one sample."""
    table = artifact("validation_scope.csv")
    table = table[table["scope"] == "in_scope"]
    return value(table, "sample", sample, "n_loans")


def holdout(name: str, col: str) -> float:
    """Value of `col` for the in-scope hold-out row of a validation table."""
    return value(artifact(name), "population", "holdout_in_scope", col)


def criteria_text() -> str:
    """'3 green, 1 amber (largest PD-decile gap 2.5 pp)' from validation_criteria.csv."""
    table = artifact("validation_criteria.csv")
    counts = table["status"].value_counts()
    gap = value(table, "criterion", "max_decile_gap", "value")
    return (f"{counts.get('green', 0)} green, {counts.get('amber', 0)} amber "
            f"(largest PD-decile gap {100 * gap:.1f} pp)")


def expected_values() -> dict[str, str]:
    """The section 12 table as it should read, computed from the CSVs."""
    dq = artifact("dq_summary.csv")
    split = artifact("sql_split_summary.csv")
    screening = artifact("model_screening.csv")
    ci = artifact("validation_confidence_intervals.csv")
    grades = artifact("grades_scale.csv")
    psi = artifact("monitoring_psi.csv")
    auc_ci = ci[ci["population"] == "holdout_in_scope"]
    main = "model_cv_metrics.csv"
    return {
        "Loans": count(value(dq, "check", "n_rows", "value")),
        "Default rate, all loans": pct(value(dq, "check", "default_rate", "value")),
        "Development / hold-out loans": (
            f"{count(value(split, 'sample', 'development', 'n_loans'))} / "
            f"{count(value(split, 'sample', 'holdout', 'n_loans'))}"),
        "In-scope loans, development / hold-out": (
            f"{count(scope_loans('development'))} / {count(scope_loans('holdout'))}"),
        "In-scope default rate, development / hold-out": (
            f"{pct(scope_rate('development'))} / {pct(scope_rate('holdout'))}"),
        "Features in the main model": (
            f"{int(screening['kept'].sum())} of {len(screening)} candidates"),
        "CV AUC / Gini / KS / Brier (in-scope development)": " / ".join(
            num(cv_mean(main, config.CV_MODEL_MAIN, m)) for m in ("auc", "gini", "ks", "brier")),
        "CV AUC, leakage demonstration: full / indicators only": (
            f"{num(cv_mean(main, config.CV_MODEL_LEAKAGE_FULL, 'auc'))} / "
            f"{num(cv_mean(main, config.CV_MODEL_LEAKAGE_ABLATION, 'auc'))}"),
        "CV AUC without `lump_sum_payment` (reported only)": num(
            cv_mean("model_sensitivity.csv", "without_lump_sum_payment", "auc")),
        "Hold-out AUC (95% bootstrap interval)": (
            f"{num(value(auc_ci, 'metric', 'auc', 'estimate'))} "
            f"({num(value(auc_ci, 'metric', 'auc', 'ci_lower'))} to "
            f"{num(value(auc_ci, 'metric', 'auc', 'ci_upper'))})"),
        "Hold-out Gini / KS / Brier": " / ".join(
            num(holdout("validation_metrics.csv", m)) for m in ("gini", "ks", "brier")),
        "Hold-out mean PD vs observed default rate": (
            f"{pct(holdout('validation_calibration.csv', 'mean_pd'))} vs "
            f"{pct(holdout('validation_calibration.csv', 'observed_rate'))}"),
        "Hold-out calibration slope": num(holdout("validation_calibration.csv", "calibration_slope")),
        "Pre-set validation criteria": criteria_text(),
        "Illustrative risk grades / share of loans in grade A": (
            f"{len(grades)} / {pct(value(grades, 'grade', 'A', 'share'), 1)}"),
        "Largest PSI, development vs hold-out (all green)": num(psi["psi"].max(), 4),
    }


@pytest.fixture(scope="module")
def readme_rows() -> dict[str, str]:
    return results_table(config.README_PATH.read_text(encoding="utf-8"))


def test_results_table_has_exactly_the_checked_rows(readme_rows):
    assert set(readme_rows) == set(expected_values())


@pytest.mark.parametrize("measure", list(expected_values()))
def test_results_value_matches_its_csv(readme_rows, measure):
    assert readme_rows[measure] == expected_values()[measure]


def test_every_psi_is_green():
    # the table row says "all green"
    assert (artifact("monitoring_psi.csv")["light"] == "green").all()


def test_grades_count_matches_the_grade_checks():
    checks = artifact("grades_checks.csv")
    assert value(checks, "check", "n_grades", "value") == len(artifact("grades_scale.csv"))


def test_every_source_named_in_the_table_exists():
    section = config.README_PATH.read_text(encoding="utf-8").split(RESULTS_HEADING, 1)[1]
    section = section.split(NEXT_HEADING, 1)[0]
    names = set(re.findall(r"`([a-z0-9_]+\.csv)`", section))
    assert names, "no source tables found in section 12"
    for name in names:
        assert (config.ARTIFACTS_DIR / name).exists(), name
