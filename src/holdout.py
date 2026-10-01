"""Stage 6: score the hold-out once with the frozen Stage 5 model (decision log D-025).

The model is loaded from disk and never refitted. The hold-out sample is scored, and a
diagnostic-only refit of the same features on the hold-out compares coefficients with
development (the D-023 trigger); that refit is never used for scoring.

    load_scoring_data        every in-scope row, both samples (D-026: no EQUI)
    load_frozen_model        the saved Stage 5 pipeline; fails loudly if it is missing
    write_pd_scores          the pd_scores table (ID, sample, pd) for Stages 7 and 8
    coefficient_stability    diagnostic refit on the hold-out vs development coefficients
    run_stage6               everything above -> artifacts/validation_*.csv

Run from the project root (after `python -m src.db` and `python -m src.model`):

    python -m src.holdout
"""

from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src import config
from src import db
from src import model
from src import validation

HOLDOUT_POPULATION = "holdout_in_scope"


def load_scoring_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """All in-scope rows (both samples), ordered by ID. Loans with credit_type = EQUI
    are outside the model's scope (D-026) and are not scored.
    """
    return con.execute(f"SELECT * FROM {config.MODEL_SCOPE_VIEW} ORDER BY {config.ID_COL}").df()


def load_frozen_model(path: Path = config.MODEL_PATH) -> Pipeline:
    """Load the Stage 5 pipeline and check it is the expected model."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No fitted model at {path}. Run `python -m src.model` first.")
    pipeline = joblib.load(path)
    model.check_expected_sign_coverage(model.feature_names(pipeline))
    return pipeline


def score(pipeline: Pipeline, data: pd.DataFrame) -> pd.Series:
    """Predicted PD for each row of `data`, using only the main-model features."""
    proba = pipeline.predict_proba(data[config.MAIN_MODEL_FEATURES])[:, 1]
    return pd.Series(proba, index=data.index, name="pd")


def write_pd_scores(con: duckdb.DuckDBPyConnection, data: pd.DataFrame, pd_scores: pd.Series) -> None:
    """Store ID, sample and PD in DuckDB, for in-scope loans only (D-026).
    Development scores are in-sample."""
    frame = pd.DataFrame({
        config.ID_COL: data[config.ID_COL], "sample": data["sample"], "pd": pd_scores
    })
    con.register("pd_frame", frame)
    con.execute(f"CREATE OR REPLACE TABLE {config.PD_SCORES_TABLE} AS SELECT * FROM pd_frame")
    con.unregister("pd_frame")


def coefficient_stability(
    pipeline: Pipeline, holdout: pd.DataFrame, dev_coefficients: pd.DataFrame
) -> pd.DataFrame:
    """DIAGNOSTIC ONLY: Logit refitted on the hold-out, using the development-fitted
    preprocessing so the scales match, next to the development coefficients."""
    design = pipeline["preprocess"].transform(holdout[config.MAIN_MODEL_FEATURES])
    fit = sm.Logit(holdout[config.TARGET_COL].to_numpy(), sm.add_constant(design)).fit(disp=0)
    names = model.feature_names(pipeline)
    table = pd.DataFrame({
        "feature": names,
        "holdout_coefficient": fit.params[1:],
        "holdout_std_error": fit.bse[1:],
        "holdout_p_value": fit.pvalues[1:],
    })
    dev = dev_coefficients.set_index("feature")
    table["dev_coefficient"] = table["feature"].map(dev["sm_coefficient"])
    table["dev_std_error"] = table["feature"].map(dev["sm_std_error"])
    table["same_sign"] = np.sign(table["holdout_coefficient"]) == np.sign(table["dev_coefficient"])
    half = 1.96
    table["ci_overlap"] = (
        (table["holdout_coefficient"] - half * table["holdout_std_error"]
         <= table["dev_coefficient"] + half * table["dev_std_error"])
        & (table["dev_coefficient"] - half * table["dev_std_error"]
           <= table["holdout_coefficient"] + half * table["holdout_std_error"])
    )
    return table


def population_results(label: str, y: pd.Series, p: pd.Series, with_ci: bool) -> tuple[dict, dict, pd.DataFrame]:
    """Discrimination row, calibration row and (optionally) bootstrap CIs for one population."""
    metrics = {"population": label, "n_loans": len(y), **validation.discrimination_metrics(y, p)}
    calibration = {
        "population": label,
        **validation.calibration_in_the_large(y, p),
        **validation.calibration_intercept_slope(y, p),
        **validation.hosmer_lemeshow(y, p),
    }
    ci = validation.bootstrap_ci(y, p).assign(population=label) if with_ci else pd.DataFrame()
    return metrics, calibration, ci


def segment_auc(holdout: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Hold-out AUC per level of each segment column (levels with both outcomes only)."""
    rows = []
    for col in columns:
        for level, grp in holdout.dropna(subset=[col]).groupby(col):
            if grp[config.TARGET_COL].nunique() == 2:
                rows.append({
                    "variable": col, "level": level,
                    "auc": float(roc_auc_score(grp[config.TARGET_COL], grp["pd"])),
                })
    return pd.DataFrame(rows)


def run_stage6(
    db_path: Path | str = config.DUCKDB_PATH,
    output_dir: Path = config.ARTIFACTS_DIR,
    model_path: Path = config.MODEL_PATH,
    cv_metrics_path: Path | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Score with the frozen model -> store pd_scores -> metrics, calibration, criteria,
    decile and segment tables -> save to `output_dir` (tests pass a temporary folder).
    """
    cv_metrics_path = cv_metrics_path or output_dir / config.MODEL_CV_METRICS_PATH.name
    cv = pd.read_csv(cv_metrics_path)
    cv_auc = float(cv[(cv["model"] == "main") & (cv["fold"] == "mean")]["auc"].iloc[0])
    dev_coefficients = pd.read_csv(output_dir / config.MODEL_COEFFICIENTS_PATH.name)

    pipeline = load_frozen_model(model_path)
    con = db.connect(db_path)
    try:
        data = load_scoring_data(con)
        data["pd"] = score(pipeline, data)
        write_pd_scores(con, data, data["pd"])

        sql_params = {"sample": config.SAMPLE_HOLDOUT}
        deciles = db.run_query(
            con, "validation_calibration_deciles", n_bins=config.CALIBRATION_N_BINS, **sql_params
        )
        feature_deciles = db.run_query(
            con, "validation_feature_deciles", n_bins=config.CALIBRATION_N_BINS, **sql_params
        )
        segments = db.run_query(
            con, "validation_segments", min_size=config.SQL_MIN_SEGMENT_SIZE, **sql_params
        )
        scope = db.run_query(con, "validation_scope", equi=config.EQUI_LEVEL)
    finally:
        con.close()

    y_col = config.TARGET_COL
    dev = data[data["sample"] == config.SAMPLE_DEVELOPMENT]
    holdout = data[data["sample"] == config.SAMPLE_HOLDOUT]
    populations = {"development_in_sample": dev, HOLDOUT_POPULATION: holdout}
    metric_rows, calibration_rows, intervals = [], [], []
    for label, frame in populations.items():
        m, c, ci = population_results(label, frame[y_col], frame["pd"], with_ci=label == HOLDOUT_POPULATION)
        metric_rows.append(m)
        calibration_rows.append(c)
        if not ci.empty:
            intervals.append(ci)
    metrics = pd.DataFrame(metric_rows)
    metrics = pd.concat([metrics, pd.DataFrame([{"population": "cv_mean_development", **cv[
        (cv["model"] == "main") & (cv["fold"] == "mean")][["auc", "gini", "ks", "brier"]].iloc[0].to_dict()}])],
        ignore_index=True)
    calibration = pd.DataFrame(calibration_rows)
    confidence_intervals = pd.concat(intervals, ignore_index=True)

    holdout_cal = calibration[calibration["population"] == HOLDOUT_POPULATION].iloc[0]
    criteria = validation.assess_criteria(
        cv_auc=cv_auc,
        holdout_auc=float(metrics.loc[metrics["population"] == HOLDOUT_POPULATION, "auc"].iloc[0]),
        binomial_p=float(holdout_cal["binomial_p"]),
        slope=float(holdout_cal["calibration_slope"]),
        max_decile_gap=float(deciles["gap"].abs().max()),
    )
    segments = segments.merge(segment_auc(holdout, config.VALIDATION_SEGMENT_COLUMNS), how="left", on=["variable", "level"])
    stability = coefficient_stability(pipeline, holdout, dev_coefficients)

    tables = {
        "scope": scope, "metrics": metrics, "confidence_intervals": confidence_intervals,
        "calibration": calibration, "criteria": criteria,
        "calibration_deciles": deciles, "feature_deciles": feature_deciles,
        "segments": segments, "coefficient_stability": stability,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.round(6).to_csv(output_dir / f"{config.VALIDATION_ARTIFACT_PREFIX}{name}.csv", index=False)
    if verbose:
        _print_report(tables)
    return tables


def _print_report(tables: dict[str, pd.DataFrame]) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for name, table in tables.items():
            print(f"\n=== {name} ===")
            print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    run_stage6()
