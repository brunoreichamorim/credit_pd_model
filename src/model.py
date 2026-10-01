"""Stage 5 baseline logistic regression PD model (decision log D-012, D-022 to D-024).

Fitted on the DuckDB development sample only, within the model's scope
(`model_scope_dataset` view: no credit_type = EQUI, D-026; `sample_split` from Stage 4). The hold-out sample is never read here; it is reserved for Stage 6's
single final evaluation (D-013).

    load_development_data     in-scope development rows (D-026), ordered by ID
    load_screening_data       ALL development rows plus credit_type, for the EQUI comparisons
    check_rare_levels         fails loudly if a categorical level is below the
                              D-023 rare-level threshold (a safeguard, not a rule
                              expected to trigger -- see docs/decision_log.md)
    build_preprocessor        the main model's ColumnTransformer (D-023)
    build_pipeline            preprocessor + LogisticRegression
    cross_validate_model      5-fold stratified CV: metrics and coefficients per fold
    out_of_fold_predictions   PD per row from the fold model that did not see it (Stage 7 check)
    fit_final_model           fit on the whole development sample; save to disk
    coefficient_table         sklearn + statsmodels coefficients, with expected signs
    run_leakage_demo          option C: full model + indicators-only ablation (D-024)

Run from the project root (after `python -m src.db`):

    python -m src.model
"""

import warnings
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src import db
from src import eda
from src import validation

MISSING_LEVEL = "<missing>"  # same label as the EDA / screening tables use

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_development_data(
    con: duckdb.DuckDBPyConnection, view: str = config.MODEL_SCOPE_VIEW
) -> pd.DataFrame:
    """Read the development-sample rows of `view`. Never reads the hold-out sample.
    The default view is the main model's scope, which leaves out credit_type = EQUI (D-026).

    Rows are ordered by ID: the CV folds are assigned by row position, so without a
    fixed order the folds (and every CV metric) would change between runs.
    """
    return con.execute(
        f"SELECT * FROM {view} WHERE sample = $sample ORDER BY {config.ID_COL}",
        {"sample": config.SAMPLE_DEVELOPMENT},
    ).df()


def load_screening_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """All development rows of model_dataset (EQUI included) plus credit_type, ordered
    by ID. credit_type is used only to compare results in and outside EQUI (D-017),
    never as a feature.
    """
    return con.execute(
        f"SELECT m.*, l.credit_type FROM {config.MODEL_DATASET_VIEW} AS m "
        f"JOIN {config.LOANS_TABLE} AS l USING ({config.ID_COL}) "
        f"WHERE m.sample = $sample ORDER BY {config.ID_COL}",
        {"sample": config.SAMPLE_DEVELOPMENT},
    ).df()


def check_rare_levels(
    df: pd.DataFrame,
    features: list[str] = config.CATEGORICAL_FEATURES,
    min_share: float = config.RARE_LEVEL_MIN_SHARE,
) -> None:
    """Raise ValueError if any non-missing level of `features` is below `min_share`
    of the development sample (D-023). This is a safeguard: as of Stage 5 no kept
    level is this rare, so the check is expected to pass, not to trigger merging code.
    It uses level frequencies only, never Status, so it cannot leak target information.
    """
    for col in features:
        shares = df[col].dropna().value_counts(normalize=True)
        rare = shares[shares < min_share]
        if not rare.empty:
            raise ValueError(
                f"Rare level(s) in '{col}' below {min_share:.0%} of development rows: "
                f"{rare.round(4).to_dict()}. Decide how to group them (docs/decision_log.md) "
                "before fitting."
            )


# ---------------------------------------------------------------------------
# Preprocessing (D-023): fitted only on training data / training folds
# ---------------------------------------------------------------------------


def _log_pipeline(impute: bool) -> Pipeline:
    """log -> (median impute) -> standardise. Missing values, if any, stay missing
    through the log step and are imputed on the log scale.
    """
    steps = [("log", FunctionTransformer(np.log, feature_names_out="one-to-one"))]
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median")))
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _main_categorical_pipeline() -> Pipeline:
    """Most-frequent impute -> one-hot encode with the fixed reference levels (D-023).

    handle_unknown="error": a level never seen in training (for example a new
    loan_type in the hold-out) raises instead of being scored silently as the
    reference level.
    """
    reference = [config.REFERENCE_LEVELS[c] for c in config.CATEGORICAL_FEATURES]
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop=reference, handle_unknown="error", sparse_output=False)),
    ])


def build_preprocessor() -> ColumnTransformer:
    """The main model's preprocessing (D-023): log + standardise the two numeric
    features, a missing indicator for income_clean, and one-hot encode the four
    categorical features with the most frequent level as reference. Only the listed
    columns are used; any other column in the input is dropped.
    """
    return ColumnTransformer(
        [
            ("income_log", _log_pipeline(impute=True), [config.INCOME_CLEAN_COL]),
            ("income_missing", MissingIndicator(features="all"), [config.INCOME_CLEAN_COL]),
            ("loan_amount_log", _log_pipeline(impute=False), ["loan_amount"]),
            ("cat", _main_categorical_pipeline(), config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def clean_feature_name(name: str) -> str:
    """'income_log__income_clean' -> 'income_clean'; 'cat__loan_type_type2' ->
    'loan_type_type2'; the missing-indicator branch becomes '<col>_missing'.
    """
    prefix, _, rest = name.partition("__")
    if prefix == "income_missing":
        return f"{config.INCOME_CLEAN_COL}_missing"
    return rest


def feature_names(pipeline: Pipeline) -> list[str]:
    """Readable names of the fitted pipeline's model inputs, in coefficient order."""
    return [clean_feature_name(n) for n in pipeline["preprocess"].get_feature_names_out()]


def build_pipeline() -> Pipeline:
    """preprocessor + logistic regression. The main model is unpenalised (D-012):
    `C=np.inf` gives plain maximum-likelihood coefficients (sklearn's recommended
    way to fit unpenalised, `penalty=None` being deprecated), comparable with the
    statsmodels fit in `coefficient_table`. `class_weight` is left at None, so
    predicted probabilities stay calibrated to the development default rate.
    """
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=config.LOGIT_MAX_ITER)
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


# ---------------------------------------------------------------------------
# Cross-validation (D-013): stratified folds within the development sample only
# ---------------------------------------------------------------------------


def _cv_folds() -> StratifiedKFold:
    """The D-013 folds. Shared by every CV function, so Stage 5's CV metrics and
    Stage 7's out-of-fold check always use the same folds."""
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)


def cross_validate_model(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """5-fold stratified CV (config.CV_FOLDS, D-013). Returns (metrics, coefficients),
    one row per fold in each, plus a 'mean' row in the metrics table. Each fold fits
    a fresh, unfitted copy of `pipeline` on its training rows only, so imputation,
    scaling and encoding never see the fold's validation rows.
    """
    folds = _cv_folds()
    metric_rows, coef_rows = [], []

    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y), start=1):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = fold_pipeline.predict_proba(X.iloc[val_idx])[:, 1]

        metric_rows.append({"fold": fold, **validation.discrimination_metrics(y.iloc[val_idx], proba)})
        for name, coef in zip(feature_names(fold_pipeline), fold_pipeline["model"].coef_.ravel()):
            coef_rows.append({"fold": fold, "feature": name, "coefficient": coef})

    metrics_df = pd.DataFrame(metric_rows)
    mean_row = {"fold": "mean", **metrics_df.drop(columns="fold").mean().to_dict()}
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)
    return metrics_df, pd.DataFrame(coef_rows)


def out_of_fold_predictions(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """PD for every row from the fold model that did NOT see it, on the same folds as
    `cross_validate_model`. Returns (pd, fold number 1..CV_FOLDS), in the row order of X.
    """
    pd_oof = np.zeros(len(y))
    fold_of_row = np.zeros(len(y), dtype=int)
    for fold, (train_idx, val_idx) in enumerate(_cv_folds().split(X, y), start=1):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        pd_oof[val_idx] = fold_pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        fold_of_row[val_idx] = fold
    return pd_oof, fold_of_row


def fit_final_model(X: pd.DataFrame, y: pd.Series, path: Path = config.MODEL_PATH) -> Pipeline:
    """Fit the main pipeline on the full development sample and save it to `path`
    (gitignored: reproducible from this code and the raw data)."""
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return pipeline


def check_expected_sign_coverage(names: list[str]) -> None:
    """Raise if the fitted feature names and config.EXPECTED_SIGNS differ. Without this,
    a renamed or new feature would silently get "no expected sign" and the sign check
    would quietly skip it.
    """
    fitted, expected = set(names), set(config.EXPECTED_SIGNS)
    if fitted != expected:
        raise ValueError(
            "Fitted feature names do not match config.EXPECTED_SIGNS. "
            f"Missing from EXPECTED_SIGNS: {sorted(fitted - expected)}. "
            f"Not produced by the model: {sorted(expected - fitted)}."
        )


def coefficient_table(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv_coefficients: pd.DataFrame
) -> pd.DataFrame:
    """Coefficients of a pipeline already fitted on the full development sample:
    sklearn's estimate, an odds ratio, the expected sign from credit sense (D-022),
    and statsmodels' standard error / p-value fitted on the same design matrix for a
    second, independent estimate (both are unpenalised MLE, so they should closely
    agree). `cv_sign_share` is the share of the CV folds (D-013) whose coefficient has
    the same sign as this full-sample fit -- a stability check.
    """
    names = feature_names(pipeline)
    check_expected_sign_coverage(names)
    sk_coefs = pipeline["model"].coef_.ravel()

    design = pipeline["preprocess"].transform(X)
    sm_model = sm.Logit(y.to_numpy(), sm.add_constant(design)).fit(disp=0)
    # sm_model params: [const, feature_1, ..., feature_k], matching `names` order

    table = pd.DataFrame({
        "feature": names,
        "coefficient": sk_coefs,
        "odds_ratio": np.exp(sk_coefs),
        "sm_coefficient": sm_model.params[1:],
        "sm_std_error": sm_model.bse[1:],
        "sm_p_value": sm_model.pvalues[1:],
    })
    table["expected_sign"] = table["feature"].map(config.EXPECTED_SIGNS)
    table["sign_matches_expected"] = [
        None if pd.isna(expected) else bool(np.sign(coef) == expected)
        for coef, expected in zip(table["coefficient"], table["expected_sign"])
    ]
    full_sign = dict(zip(table["feature"], np.sign(table["coefficient"])))
    same_sign = np.sign(cv_coefficients["coefficient"]) == cv_coefficients["feature"].map(full_sign)
    table["cv_sign_share"] = table["feature"].map(same_sign.groupby(cv_coefficients["feature"]).mean())
    return table


# ---------------------------------------------------------------------------
# Screening table (D-022): every candidate feature, kept or screened out
# ---------------------------------------------------------------------------


def screening_table(screening_data: pd.DataFrame) -> pd.DataFrame:
    """Information Value in and outside credit_type = EQUI for all 16 D-017
    candidates, plus the D-022 keep/drop decision and its reason. Computed on the
    whole development sample (see D-022 on what this means for the CV estimate).
    """
    outside_equi = screening_data[screening_data["credit_type"] != config.EQUI_LEVEL]
    rows = []
    for feature in config.MAIN_MODEL_CANDIDATE_FEATURES:
        kept = feature in config.MAIN_MODEL_FEATURES
        rows.append({
            "feature": feature,
            "iv_all": eda.information_value(screening_data, feature),
            "iv_outside_equi": eda.information_value(outside_equi, feature),
            "kept": kept,
            "reason": "D-022 kept: passes screening" if kept else config.SCREENED_OUT_FEATURES[feature],
        })
    return pd.DataFrame(rows).sort_values("iv_outside_equi", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Leakage demonstration (D-024): never used on the hold-out, grades or dashboard
# ---------------------------------------------------------------------------


def _as_category_with_missing(X: pd.DataFrame) -> pd.DataFrame:
    """Turn every value into a string label and missing values into MISSING_LEVEL, so
    missingness becomes its own one-hot level (numeric codes such as term included).
    """
    return X.astype("object").where(X.notna(), MISSING_LEVEL).astype(str)


def _demo_categorical_pipeline() -> Pipeline:
    """Missing kept as its own level -> one-hot. The first level is dropped as an
    arbitrary reference; unseen levels in a validation fold (the rare term values)
    are encoded as all zeros, which is acceptable for a demonstration-only model.
    """
    return Pipeline([
        ("missing_as_level", FunctionTransformer(_as_category_with_missing, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])


def _demo_logit() -> LogisticRegression:
    """Default L2 penalty: with near-perfect separation an unpenalised fit would not
    converge (D-024)."""
    return LogisticRegression(solver="lbfgs", max_iter=config.LOGIT_MAX_ITER)


def _leakage_full_pipeline() -> Pipeline:
    """The main model's own preprocessing plus every D-017-excluded field, with their
    missingness kept visible (D-024): numeric fields get a missing indicator next to
    the median-imputed value, categorical fields keep missing as a level.
    """
    extra = [
        (
            "extra_numeric",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            config.LEAKAGE_DEMO_NUMERIC_FEATURES,
        ),
        ("extra_missing", MissingIndicator(features="all"), config.LEAKAGE_DEMO_NUMERIC_FEATURES),
        ("extra_cat", _demo_categorical_pipeline(), config.LEAKAGE_DEMO_CATEGORICAL_FEATURES),
    ]
    preprocessor = ColumnTransformer(
        build_preprocessor().transformers + extra, remainder="drop", verbose_feature_names_out=True
    )
    return Pipeline([("preprocess", preprocessor), ("model", _demo_logit())])


def _leakage_ablation_pipeline() -> Pipeline:
    """Only whether each D-017 pricing/leakage field is missing, plus credit_type
    (D-024). Shows how much of the full demo model's skill is pure missingness.
    """
    preprocessor = ColumnTransformer(
        [
            ("missing", MissingIndicator(features="all"), config.LEAKAGE_ABLATION_MISSINGNESS_FEATURES),
            ("credit_type", _demo_categorical_pipeline(), ["credit_type"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("preprocess", preprocessor), ("model", _demo_logit())])


def run_leakage_demo(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """CV metrics (development sample only) for the full leakage-demonstration model
    and the indicators-only ablation (D-024). Compare against the main model's own
    CV metrics, computed by `run_stage5`. The demo uses ALL development rows (EQUI
    included), because EQUI is part of the leakage it demonstrates; the main model's CV
    covers its in-scope rows only (D-026). Never touches the hold-out sample; never
    used for risk grades or the dashboard.
    """
    dev = load_development_data(con, view=config.LEAKAGE_DEMO_VIEW)
    y = dev[config.TARGET_COL]
    full_X = dev[config.MAIN_MODEL_FEATURES + config.LEAKAGE_DEMO_EXTRA_FEATURES]
    ablation_X = dev[config.LEAKAGE_ABLATION_MISSINGNESS_FEATURES + ["credit_type"]]

    results = []
    with warnings.catch_warnings():
        # near-perfect separation can stop lbfgs early; expected for this demo only
        warnings.simplefilter("ignore", ConvergenceWarning)
        for label, pipeline, X in [
            ("leakage_full", _leakage_full_pipeline(), full_X),
            ("leakage_ablation_indicators_only", _leakage_ablation_pipeline(), ablation_X),
        ]:
            metrics, _ = cross_validate_model(pipeline, X, y)
            metrics.insert(0, "model", label)
            results.append(metrics)
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_stage5(
    db_path: Path | str = config.DUCKDB_PATH,
    output_dir: Path = config.ARTIFACTS_DIR,
    model_path: Path = config.MODEL_PATH,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Screen -> fit + CV the main model -> leakage demo -> save artifacts to
    `output_dir` and the fitted model to `model_path` (tests pass a temporary folder).
    """
    con = db.connect(db_path)
    try:
        dev = load_development_data(con)
        screening = screening_table(load_screening_data(con))

        check_rare_levels(dev)
        X, y = dev[config.MAIN_MODEL_FEATURES], dev[config.TARGET_COL]

        cv_metrics, cv_coefficients = cross_validate_model(build_pipeline(), X, y)
        final_model = fit_final_model(X, y, model_path)
        coefficients = coefficient_table(final_model, X, y, cv_coefficients)

        cv_metrics.insert(0, "model", "main")
        all_cv_metrics = pd.concat([cv_metrics, run_leakage_demo(con)], ignore_index=True)
    finally:
        con.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    screening.to_csv(output_dir / config.MODEL_SCREENING_PATH.name, index=False)
    all_cv_metrics.to_csv(output_dir / config.MODEL_CV_METRICS_PATH.name, index=False)
    coefficients.to_csv(output_dir / config.MODEL_COEFFICIENTS_PATH.name, index=False)

    if verbose:
        _print_report(screening, all_cv_metrics, coefficients)
    return {"screening": screening, "cv_metrics": all_cv_metrics, "coefficients": coefficients}


def _print_report(screening, cv_metrics, coefficients) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("=== Screening (D-022) ===")
        print(screening.to_string(index=False))
        print("\n=== CV metrics (mean rows) ===")
        print(cv_metrics[cv_metrics["fold"] == "mean"].to_string(index=False))
        print("\n=== Coefficients (main model) ===")
        print(coefficients.round(4).to_string(index=False))


if __name__ == "__main__":
    run_stage5()
