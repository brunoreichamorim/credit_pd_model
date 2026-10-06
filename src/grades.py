"""Stage 7: illustrative risk grades from the frozen model's PD (decision log D-014, D-027).

The grades are *illustrative internal risk grades for this project*: not an official
banking methodology, not a regulatory master scale and not an underwriting decision.

The scale is built from development loans only (in-sample PDs from `pd_scores`), and
checked on out-of-fold PDs from the Stage 5 CV folds. No hold-out outcome is read
(D-025). EQUI loans are out of the model's scope (D-026) and get no grade.

    load_development_scores   ID, pd, Status of the in-scope development loans
    initial_edges             equal-count PD cut points to start from
    assign_band               value -> band number for a list of cut points (also used for monitoring bins)
    assign_grade              PD -> grade number; rejects a PD outside [0, 1] or NaN
    band_counts               loans, defaults, mean PD and default rate per band
    step_p_values             one-sided test: does each band default more than the one below?
    merge_bands               merge adjacent bands until every D-027 rule holds; logs each merge
    build_grade_scale         labels, PD bounds and grade PD (mean PD of the grade)
    write_grade_scale         the grade_scale table in DuckDB
    create_grade_view         the pd_grades view (sql/grade_assignment.sql)
    out_of_fold_check         default rate per grade on out-of-fold PDs, pooled and per fold
    assess_grades             the D-027 pre-set checks
    run_stage7                everything above -> artifacts/grades_*.csv

Run from the project root (after `python -m src.holdout`, which writes pd_scores):

    python -m src.grades
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config
from src import db
from src import model
from src import validation

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_development_scores(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """ID, pd and Status of the development loans in pd_scores (in scope only, D-026),
    ordered by ID. Never reads the hold-out sample (D-025)."""
    tables = set(con.execute("SELECT table_name FROM information_schema.tables").df()["table_name"])
    if config.PD_SCORES_TABLE not in tables:
        raise RuntimeError(f"No {config.PD_SCORES_TABLE} table. Run `python -m src.holdout` first.")
    return con.execute(
        f"SELECT s.{config.ID_COL}, s.pd, l.{config.TARGET_COL} "
        f"FROM {config.PD_SCORES_TABLE} AS s "
        f"JOIN {config.LOANS_TABLE} AS l USING ({config.ID_COL}) "
        f"WHERE s.sample = $sample ORDER BY {config.ID_COL}",
        {"sample": config.SAMPLE_DEVELOPMENT},
    ).df()


# ---------------------------------------------------------------------------
# Building the scale (D-027)
# ---------------------------------------------------------------------------


def initial_edges(
    pd_values: pd.Series,
    n_bins: int = config.GRADE_START_BINS,
    decimals: int = config.GRADE_EDGE_DECIMALS,
) -> list[float]:
    """Cut points between `n_bins` equal-count PD bins, rounded to `decimals`.
    Repeated cut points (from tied PDs) are kept once, and a cut point that would leave
    a bin empty is dropped, so bins can be slightly unequal but never empty."""
    values = np.asarray(pd_values, dtype=float)
    cuts = np.quantile(values, np.arange(1, n_bins) / n_bins)
    edges = sorted(set(np.round(cuts, decimals).tolist()))
    while edges:
        sizes = np.bincount(assign_band(values, edges), minlength=len(edges) + 1)
        empty = np.flatnonzero(sizes == 0)
        if empty.size == 0:
            break
        del edges[min(empty[0], len(edges) - 1)]
    return edges


def assign_band(pd_values, edges: list[float]) -> np.ndarray:
    """Band number (0 = lowest PD) of each PD. A band covers lower <= pd < upper,
    the same rule as sql/grade_assignment.sql."""
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(pd_values, dtype=float), side="right")


def assign_grade(pd_values, edges: list[float]) -> np.ndarray:
    """Grade number (0 = best grade) of each PD, with assign_band's rule. A PD below
    config.PD_MIN, above config.PD_MAX or missing can only come from an upstream error,
    so it raises instead of being graded or clipped (D-027)."""
    values = np.asarray(pd_values, dtype=float)
    invalid = np.isnan(values) | (values < config.PD_MIN) | (values > config.PD_MAX)
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} PD value(s) outside [{config.PD_MIN}, {config.PD_MAX}] or missing, "
            f"for example {values[invalid][0]}. A PD like this is an error and is not graded."
        )
    return assign_band(values, edges)


def band_counts(pd_values, y, edges: list[float]) -> pd.DataFrame:
    """Loans, defaults, mean PD and observed default rate of every band."""
    frame = pd.DataFrame({"band": assign_band(pd_values, edges), "pd": np.asarray(pd_values, dtype=float),
                          "y": np.asarray(y, dtype=float)})
    counts = frame.groupby("band").agg(
        n_loans=("y", "size"), n_defaults=("y", "sum"), mean_pd=("pd", "mean")
    )
    counts["observed_rate"] = counts["n_defaults"] / counts["n_loans"]
    return counts.reset_index()


def step_p_values(counts: pd.DataFrame) -> np.ndarray:
    """One-sided two-proportion z-test for each pair of adjacent bands: the p-value of
    'the upper band defaults more often than the lower one'. Small p = a real step up;
    a step down gives p > 0.5."""
    n, k = counts["n_loans"].to_numpy(float), counts["n_defaults"].to_numpy(float)
    rate = k / n
    pooled = (k[:-1] + k[1:]) / (n[:-1] + n[1:])
    se = np.sqrt(pooled * (1 - pooled) * (1 / n[:-1] + 1 / n[1:]))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (rate[1:] - rate[:-1]) / se, 0.0)
    return norm.sf(z)


def min_grade_loans(n_total: int, min_share: float = config.GRADE_MIN_SHARE) -> int:
    """Smallest number of loans a grade may hold: floor(min_share * n_total).
    Counted in loans, because equal-count bins of N loans can fall a fraction of a
    loan short of an exact share (20 bins of 93,391 loans are 4.9994% each; D-027)."""
    return int(np.floor(min_share * n_total))


def merge_bands(
    pd_values,
    y,
    edges: list[float],
    alpha: float = config.GRADE_MERGE_ALPHA,
    min_share: float = config.GRADE_MIN_SHARE,
    max_grades: int = config.GRADE_MAX_GRADES,
) -> tuple[list[float], pd.DataFrame]:
    """Remove cut points one at a time until every step is significant (p < alpha),
    every band holds at least `min_grade_loans` loans and there are at most `max_grades`
    bands. Returns (final cut points, log of every merge).

    Order of the rules: first the least significant step; then a too-small band,
    merged with the neighbour it differs from least; then, if there are still too
    many bands, the least significant step again.
    """
    edges = list(edges)
    log = []
    min_loans = min_grade_loans(len(np.asarray(y)), min_share)
    while edges:
        counts = band_counts(pd_values, y, edges)
        p = step_p_values(counts)
        sizes = counts["n_loans"].to_numpy()
        worst = int(np.argmax(p))
        if p[worst] >= alpha:
            cut, reason = worst, "step not significant"
        elif (sizes < min_loans).any():
            small = int(np.argmin(sizes))
            neighbours = [i for i in (small - 1, small) if 0 <= i < len(p)]
            cut, reason = max(neighbours, key=lambda i: p[i]), "band below minimum share"
        elif len(counts) > max_grades:
            cut, reason = worst, "more bands than the maximum"
        else:
            break
        log.append({
            "merge_step": len(log) + 1,
            "n_bands_before": len(counts),
            "edge_removed": edges[cut],
            "step_p_value": float(p[cut]),
            "lower_band_rate": float(counts["observed_rate"].iloc[cut]),
            "upper_band_rate": float(counts["observed_rate"].iloc[cut + 1]),
            "reason": reason,
        })
        del edges[cut]
    return edges, pd.DataFrame(log)


def build_grade_scale(edges: list[float], pd_values) -> pd.DataFrame:
    """The grade definitions: label, rank, PD bounds (0 and 1 at the ends) and grade PD,
    the mean model PD of the grade's development loans (D-027, S5)."""
    n_grades = len(edges) + 1
    if n_grades > len(config.GRADE_LABELS):
        raise ValueError(f"{n_grades} grades, but only {len(config.GRADE_LABELS)} labels.")
    bounds = [0.0, *edges, 1.0]
    mean_pd = pd.Series(np.asarray(pd_values, dtype=float)).groupby(assign_band(pd_values, edges)).mean()
    return pd.DataFrame({
        "grade": list(config.GRADE_LABELS[:n_grades]),
        "grade_rank": range(1, n_grades + 1),
        "pd_lower": bounds[:-1],
        "pd_upper": bounds[1:],
        "grade_pd": mean_pd.reindex(range(n_grades)).to_numpy(),
    })


def write_grade_scale(con: duckdb.DuckDBPyConnection, scale: pd.DataFrame) -> None:
    """Store the grade definitions as the grade_scale table."""
    con.register("scale_frame", scale)
    con.execute(f"CREATE OR REPLACE TABLE {config.GRADE_SCALE_TABLE} AS SELECT * FROM scale_frame")
    con.unregister("scale_frame")


def create_grade_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create the pd_grades view (sql/grade_assignment.sql)."""
    con.execute(db.read_sql("grade_assignment"))


# ---------------------------------------------------------------------------
# Checks (D-027)
# ---------------------------------------------------------------------------


def is_increasing(rates) -> bool:
    """True if every value is strictly higher than the one before."""
    return bool(np.all(np.diff(np.asarray(rates, dtype=float)) > 0))


def out_of_fold_check(con: duckdb.DuckDBPyConnection, scale: pd.DataFrame) -> pd.DataFrame:
    """Grade the out-of-fold PDs of the in-scope development loans (Stage 5 folds) with
    the same scale. One row per grade for the pooled folds ('pooled') and for each fold."""
    dev = model.load_development_data(con)
    X, y = dev[config.MAIN_MODEL_FEATURES], dev[config.TARGET_COL]
    pd_oof, fold = model.out_of_fold_predictions(model.build_pipeline(), X, y)
    frame = pd.DataFrame({
        "fold": fold,
        "grade": scale["grade"].to_numpy()[assign_grade(pd_oof, scale["pd_lower"].iloc[1:].tolist())],
        "pd": pd_oof,
        "y": y.to_numpy(float),
    })
    tables = []
    for label, part in [("pooled", frame), *[(f"fold_{f}", frame[frame["fold"] == f]) for f in sorted(frame["fold"].unique())]]:
        table = part.groupby("grade").agg(n_loans=("y", "size"), mean_pd=("pd", "mean"), observed_rate=("y", "mean"))
        tables.append(table.reset_index().assign(sample=label))
    columns = ["sample", "grade", "n_loans", "mean_pd", "observed_rate"]
    return pd.concat(tables, ignore_index=True)[columns]


def assess_grades(summary: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """The D-027 checks table: one row per check with its value, rule and status
    ('pass', 'fail' or 'report' for checks that are reported without a judgement)."""
    in_sample_up = is_increasing(summary["observed_rate"])
    pooled_up = is_increasing(oof.loc[oof["sample"] == "pooled", "observed_rate"])
    min_loans = min_grade_loans(int(summary["n_loans"].sum()))
    smallest = int(summary["n_loans"].min())
    max_p = float(summary["step_p_value"].max())  # NaN for grade A is skipped
    n_grades = len(summary)

    def verdict(passed: bool) -> str:
        return "pass" if passed else "fail"

    rows = [
        ("rates_increase_in_sample", in_sample_up, "must be True", verdict(in_sample_up)),
        ("rates_increase_out_of_fold_pooled", pooled_up, "must be True", verdict(pooled_up)),
        ("min_grade_loans", smallest, f">= {min_loans} (floor of {config.GRADE_MIN_SHARE} x loans)",
         verdict(smallest >= min_loans)),
        ("max_step_p_value", max_p, f"< {config.GRADE_MERGE_ALPHA}", verdict(max_p < config.GRADE_MERGE_ALPHA)),
        ("n_grades", n_grades, f"<= {config.GRADE_MAX_GRADES}", verdict(n_grades <= config.GRADE_MAX_GRADES)),
    ]
    for fold in sorted(s for s in oof["sample"].unique() if s != "pooled"):
        up = is_increasing(oof.loc[oof["sample"] == fold, "observed_rate"])
        rows.append((f"rates_increase_out_of_fold_{fold}", up, "reported only", "report"))
    return pd.DataFrame(rows, columns=["check", "value", "rule", "status"])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_stage7(
    db_path: Path | str = config.DUCKDB_PATH,
    output_dir: Path = config.ARTIFACTS_DIR,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build the grade scale on development PDs -> store grade_scale and pd_grades ->
    grade summary, out-of-fold check and D-027 checks -> save to `output_dir`
    (tests pass a temporary folder)."""
    con = db.connect(db_path)
    try:
        dev = load_development_scores(con)
        edges, merge_log = merge_bands(dev["pd"], dev[config.TARGET_COL], initial_edges(dev["pd"]))
        scale = build_grade_scale(edges, dev["pd"])
        write_grade_scale(con, scale)
        create_grade_view(con)
        summary = db.run_query(
            con, "grades_summary", sample=config.SAMPLE_DEVELOPMENT, min_size=config.SQL_MIN_SEGMENT_SIZE
        )
        scope = db.run_query(con, "grades_scope", equi=config.EQUI_LEVEL)
        oof = out_of_fold_check(con, scale)
    finally:
        con.close()

    summary = summary.merge(scale[["grade", "pd_lower", "pd_upper"]], on="grade")
    summary["step_p_value"] = [np.nan, *step_p_values(summary)]
    dev["grade"] = scale["grade"].to_numpy()[assign_grade(dev["pd"], edges)]
    summary["binomial_p"] = summary["grade"].map(
        lambda g: validation.calibration_in_the_large(
            dev.loc[dev["grade"] == g, config.TARGET_COL], dev.loc[dev["grade"] == g, "pd"]
        )["binomial_p"]
    )
    columns = ["grade", "grade_rank", "pd_lower", "pd_upper", "grade_pd", "n_loans", "n_defaults",
               "share", "mean_pd", "observed_rate", "gap", "step_p_value", "binomial_p"]
    summary = summary[columns]
    checks = assess_grades(summary, oof)

    tables = {"scale": summary, "merge_log": merge_log, "oof_check": oof, "checks": checks, "scope": scope}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.round(6).to_csv(output_dir / f"{config.GRADES_ARTIFACT_PREFIX}{name}.csv", index=False)
    if verbose:
        _print_report(tables)
    return tables


def _print_report(tables: dict[str, pd.DataFrame]) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for name, table in tables.items():
            print(f"\n=== {name} ===")
            print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    run_stage7()
