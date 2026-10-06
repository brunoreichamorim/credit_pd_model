"""Stage 8: monitoring of the frozen PD model with PSI / stability measures (decision log D-028).

A baseline is built from the in-scope development loans: frozen bins for the PD score,
the illustrative grades and the six model features. A monitored sample is then compared
with it. In this project the monitored sample is the hold-out, standing in for a "next
period". Only its inputs and scores are read, never its outcomes (D-025, D-028).

What this cannot show here: `year` is constant (D-004) and the split is random (D-013), so
there is no drift to detect and the PSI is close to zero by construction. A green result
shows that the monitoring works; it is not evidence of stability over time.

    load_monitoring_inputs   ID, sample, score, grade and model features (no Status)
    numeric_bins             frozen equal-count development bins + a missing bin
    categorical_bins         one bin per development level + missing + unseen
    build_baseline           all bins with their development counts
    assign_bins              value -> bin label, the same rule as sql/monitoring_bin_counts.sql
    psi_contributions        (a - e) * ln(a / e) per bin, shares floored at PSI_EPSILON
    psi_from_values          PSI of one variable straight from two arrays of values
    psi_summary              PSI, chi-square reference and traffic light per variable
    characteristic_analysis  change in mean log-odds contribution per feature
    watch_list               D-023 / D-026 / D-027 KPIs with traffic lights
    grade_backtest           observed default rate vs grade PD per grade (development only)
    run_stage8               everything above -> artifacts/monitoring_*.csv

Run from the project root (after `python -m src.grades`, which writes pd_grades):

    python -m src.monitoring
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.pipeline import Pipeline

from src import config
from src import db
from src import grades
from src import holdout
from src import model
from src import validation

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_monitoring_inputs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """ID, sample, PD ('score'), grade and the model features of every scored loan,
    ordered by ID. Status is not selected, so no outcome can enter monitoring (D-025)."""
    views = set(con.execute("SELECT table_name FROM information_schema.tables").df()["table_name"])
    if config.PD_GRADES_VIEW not in views:
        raise RuntimeError(f"No {config.PD_GRADES_VIEW} view. Run `python -m src.grades` first.")
    features = ", ".join(f"m.{c}" for c in config.MAIN_MODEL_FEATURES)
    return con.execute(
        f"SELECT g.{config.ID_COL}, g.sample, g.pd AS {config.MONITORING_SCORE_VARIABLE}, "
        f"g.grade AS {config.MONITORING_GRADE_VARIABLE}, {features} "
        f"FROM {config.PD_GRADES_VIEW} AS g "
        f"JOIN {config.MODEL_SCOPE_VIEW} AS m USING ({config.ID_COL}) "
        f"ORDER BY {config.ID_COL}"
    ).df()


# ---------------------------------------------------------------------------
# Baseline bins (D-028)
# ---------------------------------------------------------------------------


def _range_label(lower: float, upper: float) -> str:
    return f"[{lower:.10g}, {upper:.10g})"


def numeric_bins(
    variable: str,
    values: pd.Series,
    n_bins: int = config.MONITORING_N_BINS,
    decimals: int = config.MONITORING_EDGE_DECIMALS,
) -> pd.DataFrame:
    """Equal-count bins of the non-missing development values (lower <= value < upper,
    open-ended at both ends), plus a missing bin. Repeated edges from tied values are kept
    once, so a rounded variable can get fewer than `n_bins` bins (grades.initial_edges)."""
    present = pd.Series(values, dtype=float).dropna()
    edges = grades.initial_edges(present, n_bins, decimals) if len(present) else []
    bounds = [-np.inf, *edges, np.inf]
    rows = [
        {"variable": variable, "kind": "range", "bin_label": _range_label(lo, hi),
         "lower": lo, "upper": hi, "level": None}
        for lo, hi in zip(bounds[:-1], bounds[1:])
    ]
    rows.append({"variable": variable, "kind": "missing", "bin_label": config.MISSING_BIN_LABEL,
                 "lower": np.nan, "upper": np.nan, "level": None})
    return pd.DataFrame(rows)


def categorical_bins(variable: str, values: pd.Series) -> pd.DataFrame:
    """One bin per development level (sorted), plus a missing and an unseen-level bin."""
    levels = sorted(pd.Series(values).dropna().astype(str).unique())
    rows = [{"variable": variable, "kind": "level", "bin_label": lvl,
             "lower": np.nan, "upper": np.nan, "level": lvl} for lvl in levels]
    for kind, label in [("missing", config.MISSING_BIN_LABEL), ("unseen", config.UNSEEN_BIN_LABEL)]:
        rows.append({"variable": variable, "kind": kind, "bin_label": label,
                     "lower": np.nan, "upper": np.nan, "level": None})
    return pd.DataFrame(rows)


def assign_bins(values, bins: pd.DataFrame) -> np.ndarray:
    """Bin label of each value, for the bins of one variable. Same rule as
    sql/monitoring_bin_counts.sql: numeric lower <= value < upper, missing -> '<missing>';
    categorical level -> its bin, missing -> '<missing>', anything else -> '<unseen>'."""
    ranges = bins[bins["kind"] == "range"]
    if not ranges.empty:
        numbers = pd.Series(values, dtype=float).to_numpy()
        band = grades.assign_band(numbers, ranges["lower"].iloc[1:].tolist())
        labels = ranges["bin_label"].to_numpy()[np.minimum(band, len(ranges) - 1)].astype(object)
        labels[np.isnan(numbers)] = config.MISSING_BIN_LABEL
        return labels
    series = pd.Series(values, dtype=object).reset_index(drop=True)
    levels = set(bins.loc[bins["kind"] == "level", "level"])
    text = series.astype(str)
    labels = np.where(text.isin(levels), text, config.UNSEEN_BIN_LABEL).astype(object)
    labels[series.isna().to_numpy()] = config.MISSING_BIN_LABEL
    return labels


def bin_counts(values, bins: pd.DataFrame) -> np.ndarray:
    """Number of values in each bin, in the row order of `bins` (0 for an empty bin)."""
    labels = pd.Series(assign_bins(values, bins))
    return labels.value_counts().reindex(bins["bin_label"], fill_value=0).to_numpy()


def build_baseline(dev_inputs: pd.DataFrame) -> pd.DataFrame:
    """Every monitoring bin of every variable, with its development count and share."""
    tables = []
    for variable in config.MONITORING_NUMERIC_VARIABLES + config.MONITORING_CATEGORICAL_VARIABLES:
        if variable in config.MONITORING_NUMERIC_VARIABLES:
            bins = numeric_bins(variable, dev_inputs[variable])
        else:
            bins = categorical_bins(variable, dev_inputs[variable])
        bins.insert(1, "bin_order", range(1, len(bins) + 1))
        bins["n_loans"] = bin_counts(dev_inputs[variable], bins)
        bins["share"] = bins["n_loans"] / len(dev_inputs)
        tables.append(bins)
    columns = ["variable", "bin_order", "bin_label", "kind", "lower", "upper", "level", "n_loans", "share"]
    return pd.concat(tables, ignore_index=True)[columns]


def write_baseline(con: duckdb.DuckDBPyConnection, baseline: pd.DataFrame) -> None:
    """Store the baseline bins as the monitoring_baseline table and create the
    monitoring_bin_counts view on top of it (sql/monitoring_bin_counts.sql)."""
    con.register("baseline_frame", baseline)
    con.execute(f"CREATE OR REPLACE TABLE {config.MONITORING_BASELINE_TABLE} AS SELECT * FROM baseline_frame")
    con.unregister("baseline_frame")
    con.execute(db.read_sql("monitoring_bin_counts"))


# ---------------------------------------------------------------------------
# PSI (D-028)
# ---------------------------------------------------------------------------


def psi_contributions(expected_share, actual_share, eps: float = config.PSI_EPSILON) -> np.ndarray:
    """PSI contribution of each bin: (a - e) * ln(a / e), with both shares floored at
    `eps` so an empty bin gives a large but finite value. Never negative."""
    e = np.maximum(np.asarray(expected_share, dtype=float), eps)
    a = np.maximum(np.asarray(actual_share, dtype=float), eps)
    return (a - e) * np.log(a / e)


def psi_from_values(baseline_values, monitored_values, bins: pd.DataFrame,
                    eps: float = config.PSI_EPSILON) -> float:
    """PSI of one variable, computed directly from the two samples' values."""
    base, mon = bin_counts(baseline_values, bins), bin_counts(monitored_values, bins)
    return float(psi_contributions(base / base.sum(), mon / mon.sum(), eps).sum())


def psi_light(value: float) -> str:
    """Traffic light for a PSI / CSI value (PSI_THRESHOLDS, both limits inclusive)."""
    return validation.traffic_light(value, *config.PSI_THRESHOLDS, higher_is_better=False)


def psi_chi2_reference(psi: float, n_baseline: int, n_monitored: int, n_bins: int) -> dict[str, float]:
    """With no shift, PSI x n1*n2/(n1+n2) is approximately chi-square with n_bins - 1
    degrees of freedom. Returns the PSI expected from sampling noise alone and the
    p-value of the observed PSI. Reported without a judgement (D-028)."""
    scale = 1 / n_baseline + 1 / n_monitored
    df = max(n_bins - 1, 1)
    return {
        "psi_expected_no_shift": df * scale,
        "chi2_statistic": psi / scale,
        "chi2_df": df,
        "chi2_p_value": float(chi2.sf(psi / scale, df)),
    }


def _role(variable: str) -> str:
    if variable == config.MONITORING_SCORE_VARIABLE:
        return "score"
    return "grade" if variable == config.MONITORING_GRADE_VARIABLE else "feature"


def psi_summary(psi_bins: pd.DataFrame) -> pd.DataFrame:
    """One row per variable: PSI (sum of the bin contributions), the chi-square
    reference and the traffic light. Bins empty in both samples are not counted."""
    rows = []
    order = config.MONITORING_NUMERIC_VARIABLES + config.MONITORING_CATEGORICAL_VARIABLES
    for variable in order:
        part = psi_bins[psi_bins["variable"] == variable]
        n_base, n_mon = int(part["n_baseline"].sum()), int(part["n_monitored"].sum())
        used = int(((part["n_baseline"] + part["n_monitored"]) > 0).sum())
        value = float(part["psi_contribution"].sum())
        rows.append({
            "variable": variable, "role": _role(variable), "n_bins_used": used,
            "n_baseline": n_base, "n_monitored": n_mon, "psi": value,
            **psi_chi2_reference(value, n_base, n_mon, used),
            "light": psi_light(value),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Characteristic analysis
# ---------------------------------------------------------------------------


def _feature_of_term(term: str) -> str:
    """The model feature a coefficient belongs to ('loan_type_type2' -> 'loan_type')."""
    matches = [f for f in config.MAIN_MODEL_FEATURES if term == f or term.startswith(f"{f}_")]
    return max(matches, key=len)


def characteristic_analysis(pipeline: Pipeline, base: pd.DataFrame, monitored: pd.DataFrame) -> pd.DataFrame:
    """For each feature: mean contribution to the log-odds (coefficient x mean transformed
    value, summed over the feature's terms) in each sample, and the change. The changes add
    up to the change in mean log-odds (the 'total' row), so they show which feature moves
    the score. Uses inputs only (D-025)."""
    preprocess = pipeline["preprocess"]
    coefficients = pipeline["model"].coef_[0]
    terms = pd.DataFrame({
        "feature": [_feature_of_term(t) for t in model.feature_names(pipeline)],
        "baseline_contribution": coefficients * preprocess.transform(base[config.MAIN_MODEL_FEATURES]).mean(axis=0),
        "monitored_contribution": coefficients * preprocess.transform(monitored[config.MAIN_MODEL_FEATURES]).mean(axis=0),
    })
    table = terms.groupby("feature", sort=False).sum().reindex(config.MAIN_MODEL_FEATURES).reset_index()
    total = table[["baseline_contribution", "monitored_contribution"]].sum()
    table = pd.concat([table, pd.DataFrame([{"feature": "total", **total.to_dict()}])], ignore_index=True)
    table["change_in_log_odds"] = table["monitored_contribution"] - table["baseline_contribution"]
    return table


# ---------------------------------------------------------------------------
# Watch list (D-023, D-026, D-027)
# ---------------------------------------------------------------------------


def watch_list(psi_bins: pd.DataFrame, baseline: pd.DataFrame, scope: pd.DataFrame,
               baseline_sample: str, monitored_sample: str) -> pd.DataFrame:
    """The D-028 watch-list KPIs, from aggregate counts only: baseline value, monitored
    value, the rule and a traffic light."""
    def share(variable: str, label: str) -> tuple[float, float]:
        row = psi_bins[(psi_bins["variable"] == variable) & (psi_bins["bin_label"] == label)].iloc[0]
        return float(row["share_baseline"]), float(row["share_monitored"])

    def light(value: float, limits: tuple[float, float]) -> str:
        return validation.traffic_light(value, *limits, higher_is_better=False)

    rows = []
    green, amber = config.WATCH_TOP_DECILE_SHARE
    for feature in config.WATCH_TOP_DECILE_FEATURES:
        top = baseline[(baseline["variable"] == feature) & (baseline["kind"] == "range")]["bin_label"].iloc[-1]
        base, mon = share(feature, top)
        rows.append((f"top_decile_share_{feature}", "D-023", base, mon,
                     f"green <= {green}, amber <= {amber}", light(mon, config.WATCH_TOP_DECILE_SHARE)))

    best = config.GRADE_LABELS[0]
    base, mon = share(config.MONITORING_GRADE_VARIABLE, best)
    green, amber = config.WATCH_GRADE_A_SHARE_CHANGE
    rows.append((f"grade_{best}_share_change", "D-027", base, mon,
                 f"|change| green <= {green}, amber <= {amber}", light(abs(mon - base), config.WATCH_GRADE_A_SHARE_CHANGE)))

    scope_share = (scope["n_out_of_scope_equi"] / scope["n_loans"]).set_axis(scope["sample"])
    green, amber = config.WATCH_OUT_OF_SCOPE_SHARE
    rows.append(("out_of_scope_share", "D-026", float(scope_share[baseline_sample]),
                 float(scope_share[monitored_sample]), f"green <= {green}, amber <= {amber}",
                 light(float(scope_share[monitored_sample]), config.WATCH_OUT_OF_SCOPE_SHARE)))

    column, level = config.WATCH_RARE_LEVEL
    base, mon = share(column, level)
    rows.append((f"share_{column}_{level}", "D-023", base, mon,
                 f"green >= {config.RARE_LEVEL_MIN_SHARE}, else amber",
                 "green" if mon >= config.RARE_LEVEL_MIN_SHARE else "amber"))

    unseen = psi_bins[psi_bins["bin_label"] == config.UNSEEN_BIN_LABEL]
    n_unseen = int(unseen["n_monitored"].sum())
    rows.append(("n_unseen_levels", "D-023", float(unseen["n_baseline"].sum()), float(n_unseen),
                 "green = 0, else red", "green" if n_unseen == 0 else "red"))
    return pd.DataFrame(rows, columns=["kpi", "decision", "baseline_value", "monitored_value", "rule", "light"])


# ---------------------------------------------------------------------------
# Per-grade backtest (development outcomes only, D-028 option a)
# ---------------------------------------------------------------------------


def grade_backtest(y, grade, scale: pd.DataFrame, label: str) -> pd.DataFrame:
    """Observed default rate of each grade against its grade PD (D-027), with a binomial
    test. Lights reuse the D-025 thresholds: |gap| with CRITERION_DECILE_GAP and the
    binomial p with CRITERION_BINOMIAL_P. A grade with no loans gets 'n/a'."""
    frame = pd.DataFrame({"grade": np.asarray(grade), "y": np.asarray(y, dtype=float)})
    rows = []
    for g, grade_pd in zip(scale["grade"], scale["grade_pd"]):
        part = frame.loc[frame["grade"] == g, "y"]
        row = {"sample": label, "grade": g, "grade_pd": float(grade_pd), "n_loans": len(part),
               "n_defaults": int(part.sum())}
        if part.empty:
            row.update(observed_rate=np.nan, gap=np.nan, binomial_p=np.nan, gap_light="n/a", binomial_light="n/a")
        else:
            test = validation.calibration_in_the_large(part, np.full(len(part), grade_pd))
            gap = test["observed_rate"] - grade_pd
            row.update(
                observed_rate=test["observed_rate"], gap=gap, binomial_p=test["binomial_p"],
                gap_light=validation.traffic_light(abs(gap), *config.CRITERION_DECILE_GAP, higher_is_better=False),
                binomial_light=validation.traffic_light(test["binomial_p"], *config.CRITERION_BINOMIAL_P,
                                                        higher_is_better=True),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def development_backtest(con: duckdb.DuckDBPyConnection, scale: pd.DataFrame) -> pd.DataFrame:
    """The per-grade backtest on development loans: in-sample PDs and out-of-fold PDs from
    the Stage 5 folds, graded with the same scale. Never reads a hold-out outcome."""
    edges = scale["pd_lower"].iloc[1:].tolist()
    labels = scale["grade"].to_numpy()
    scores = grades.load_development_scores(con)
    in_sample = grade_backtest(scores[config.TARGET_COL], labels[grades.assign_band(scores["pd"], edges)],
                               scale, "development_in_sample")
    dev = model.load_development_data(con)
    pd_oof, _ = model.out_of_fold_predictions(
        model.build_pipeline(), dev[config.MAIN_MODEL_FEATURES], dev[config.TARGET_COL]
    )
    out_of_fold = grade_backtest(dev[config.TARGET_COL], labels[grades.assign_band(pd_oof, edges)],
                                 scale, "development_out_of_fold")
    return pd.concat([in_sample, out_of_fold], ignore_index=True)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def hide_small_counts(table: pd.DataFrame, count_to_share: dict[str, str],
                      min_size: int = config.SQL_MIN_SEGMENT_SIZE) -> pd.DataFrame:
    """Blank out counts between 1 and min_size - 1, and their shares, before a table is
    written, so no committed row describes (almost) a single borrower (D-021). Zero
    counts are kept: an empty bin describes nobody. `count_to_share` maps each count
    column to its share column."""
    table = table.copy()
    for count_col, share_col in count_to_share.items():
        small = table[count_col].between(1, min_size - 1)
        table[count_col] = table[count_col].astype("Int64").mask(small)
        table[share_col] = table[share_col].mask(small)
    return table


def run_stage8(
    db_path: Path | str = config.DUCKDB_PATH,
    output_dir: Path = config.ARTIFACTS_DIR,
    model_path: Path = config.MODEL_PATH,
    monitored_sample: str = config.SAMPLE_HOLDOUT,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build the development baseline -> bin every sample in DuckDB -> PSI per bin and
    variable, characteristic analysis, watch list, development backtest -> save to
    `output_dir` (tests pass a temporary folder)."""
    pipeline = holdout.load_frozen_model(model_path)
    baseline_sample = config.SAMPLE_DEVELOPMENT
    con = db.connect(db_path)
    try:
        inputs = load_monitoring_inputs(con)
        base_inputs = inputs[inputs["sample"] == baseline_sample]
        monitored_inputs = inputs[inputs["sample"] == monitored_sample]
        baseline = build_baseline(base_inputs)
        write_baseline(con, baseline)
        psi_bins = db.run_query(con, "monitoring_psi", baseline=baseline_sample,
                                monitored=monitored_sample, eps=config.PSI_EPSILON)
        scope = db.run_query(con, "grades_scope", equi=config.EQUI_LEVEL)
        scale = con.execute(f"SELECT * FROM {config.GRADE_SCALE_TABLE} ORDER BY grade_rank").df()
        backtest = development_backtest(con, scale)
    finally:
        con.close()

    variables = config.MONITORING_NUMERIC_VARIABLES + config.MONITORING_CATEGORICAL_VARIABLES
    psi_bins["variable"] = pd.Categorical(psi_bins["variable"], categories=variables, ordered=True)
    psi_bins = psi_bins.sort_values(["variable", "bin_order"], ignore_index=True)
    psi_bins["variable"] = psi_bins["variable"].astype(str)
    scope["out_of_scope_share"] = scope["n_out_of_scope_equi"] / scope["n_loans"]
    tables = {
        "baseline": baseline,
        "psi": psi_summary(psi_bins),
        "psi_bins": psi_bins,
        "characteristic": characteristic_analysis(pipeline, base_inputs, monitored_inputs),
        "watch_list": watch_list(psi_bins, baseline, scope, baseline_sample, monitored_sample),
        "grade_backtest": backtest,
        "scope": scope,
    }
    written = {
        **tables,
        "baseline": hide_small_counts(baseline, {"n_loans": "share"}),
        "psi_bins": hide_small_counts(psi_bins, {"n_baseline": "share_baseline", "n_monitored": "share_monitored"}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in written.items():
        table.round(6).to_csv(output_dir / f"{config.MONITORING_ARTIFACT_PREFIX}{name}.csv", index=False)
    if verbose:
        _print_report(tables)
    return tables


def _print_report(tables: dict[str, pd.DataFrame]) -> None:
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        for name in ["psi", "watch_list", "characteristic", "grade_backtest", "scope"]:
            print(f"\n=== {name} ===")
            print(tables[name].round(4).to_string(index=False))


if __name__ == "__main__":
    run_stage8()
