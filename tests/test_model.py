"""Tests for the Stage 5 baseline model (src/model.py).

Two groups:

* Unit tests on a small synthetic dataset (built through src.db, so a toy DuckDB
  with model_dataset behaves exactly like the real one). They run anywhere.
* Tests on the real data, skipped automatically if data/raw/Loan_Default.csv is
  absent (D-002). They check the Stage 5 screening facts and CV results.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src import data_processing as dp
from src import db
from src import eda
from src import model

# ---------------------------------------------------------------------------
# Unit tests on a small synthetic dataset
# ---------------------------------------------------------------------------

N_TOY = 200
RNG = np.random.RandomState(0)


@pytest.fixture
def toy_processed() -> pd.DataFrame:
    """200 processed rows with a genuine but noisy relationship to Status, so the
    unpenalised logistic regression converges without perfect separation.
    """
    ids = np.arange(1, N_TOY + 1)
    raw = pd.DataFrame({col: ["x"] * N_TOY for col in config.EXPECTED_RAW_COLUMNS})
    raw[config.ID_COL] = ids
    raw[config.YEAR_COL] = 2019

    income = RNG.lognormal(mean=8.5, sigma=0.6, size=N_TOY)
    loan_amount = RNG.lognormal(mean=12.0, sigma=0.4, size=N_TOY)
    lpsm = RNG.random(N_TOY) < 0.15
    neg_amm = RNG.random(N_TOY) < 0.15
    loan_type = RNG.choice(["type1", "type2", "type3"], N_TOY, p=[0.7, 0.2, 0.1])
    loan_purpose = RNG.choice(["p1", "p2", "p3", "p4"], N_TOY, p=[0.25, 0.1, 0.35, 0.3])

    score = (
        -0.6 * (np.log(income) - 8.5)
        + 1.2 * lpsm
        + 1.0 * neg_amm
        + 0.5 * (loan_type == "type2")
        + RNG.normal(scale=0.5, size=N_TOY)
    )
    status = (score > np.quantile(score, 0.6)).astype(int)  # ~40% default, noisy

    raw[config.TARGET_COL] = status
    raw["income"] = income
    raw.loc[RNG.random(N_TOY) < 0.2, "income"] = np.nan  # missing income, D-008
    raw["loan_amount"] = loan_amount
    raw["lump_sum_payment"] = np.where(lpsm, "lpsm", "not_lpsm")
    raw["Neg_ammortization"] = np.where(neg_amm, "neg_amm", "not_neg")
    raw["loan_type"] = loan_type
    raw["loan_purpose"] = loan_purpose
    for col in ["rate_of_interest", "Interest_rate_spread", "Upfront_charges",
                "Credit_Score", "dtir1", "total_units", "LTV"]:
        raw[col] = 1.0
    raw["property_value"] = 500_000.0
    return dp.apply_cleaning_rules(raw)


@pytest.fixture
def toy_con(toy_processed, tmp_path):
    parquet = tmp_path / "loans_clean.parquet"
    dp.save_processed_data(toy_processed, parquet)
    con = db.connect(tmp_path / "toy.duckdb")
    db.load_loans(con, parquet)
    db.assign_sample_split(con)
    db.create_model_dataset_view(con)
    db.create_leakage_demo_view(con)
    yield con
    con.close()


@pytest.fixture
def toy_dev(toy_con) -> pd.DataFrame:
    return model.load_development_data(toy_con)


def test_loader_returns_only_development_rows(toy_dev):
    assert len(toy_dev) > 0
    assert set(toy_dev["sample"]) == {config.SAMPLE_DEVELOPMENT}
    assert list(toy_dev.columns) == [config.ID_COL, "sample", config.TARGET_COL] + config.MAIN_MODEL_CANDIDATE_FEATURES


def test_loaders_return_rows_ordered_by_id(toy_con):
    # CV folds are assigned by row position, so a fixed row order keeps them reproducible
    for df in [model.load_development_data(toy_con), model.load_screening_data(toy_con)]:
        assert df[config.ID_COL].is_monotonic_increasing


def test_check_rare_levels_passes_on_the_toy_data(toy_dev):
    model.check_rare_levels(toy_dev)  # does not raise: no level is this rare by design


def test_check_rare_levels_raises_on_a_rare_level(toy_dev):
    df = toy_dev.copy()
    df.loc[df.index[0], "loan_type"] = "rare_type"  # 1 out of ~140 development rows
    with pytest.raises(ValueError, match="Rare level"):
        model.check_rare_levels(df)


def test_log_pipeline_applies_log_before_scaling():
    from sklearn.preprocessing import StandardScaler

    X = pd.DataFrame({"loan_amount": [10.0, 100.0, 1_000.0, 5_000.0]})
    out = model._log_pipeline(impute=False).fit_transform(X)
    expected = StandardScaler().fit_transform(np.log(X))
    np.testing.assert_allclose(out, expected)


def test_preprocessor_drops_reference_levels_and_adds_missing_indicator(toy_dev):
    X = toy_dev[config.MAIN_MODEL_FEATURES]
    preprocessor = model.build_preprocessor()
    preprocessor.fit(X)
    names = [model.clean_feature_name(n) for n in preprocessor.get_feature_names_out()]

    for feature, reference in config.REFERENCE_LEVELS.items():
        assert f"{feature}_{reference}" not in names, f"reference level for {feature} was not dropped"
    assert f"{config.INCOME_CLEAN_COL}_missing" in names


def test_imputation_statistics_are_learned_on_training_rows_only(toy_dev):
    X = toy_dev[config.MAIN_MODEL_FEATURES]
    half = len(X) // 2
    train, rest = X.iloc[:half], X.iloc[half:]
    # construct training and full data so their income medians clearly differ
    train = train.copy()
    train[config.INCOME_CLEAN_COL] = np.linspace(1_000, 2_000, len(train))
    full = pd.concat([train, rest.assign(**{config.INCOME_CLEAN_COL: np.linspace(5_000, 6_000, len(rest))})])

    preprocessor = model.build_preprocessor()
    preprocessor.fit(train)
    fitted_median = preprocessor.named_transformers_["income_log"].named_steps["impute"].statistics_[0]

    assert fitted_median == pytest.approx(np.log(train[config.INCOME_CLEAN_COL]).median())
    assert fitted_median != pytest.approx(np.log(full[config.INCOME_CLEAN_COL]).median())


def test_main_pipeline_never_sees_leakage_columns(toy_con):
    # even when handed the leakage-demo view, the main preprocessor uses only its own columns
    X = model.load_development_data(toy_con, view=config.LEAKAGE_DEMO_VIEW)
    preprocessor = model.build_preprocessor().fit(X)
    used = {col for name, _, cols in preprocessor.transformers_ if name != "remainder" for col in cols}
    assert used == set(config.MAIN_MODEL_FEATURES)
    assert preprocessor.remainder == "drop"


def test_unseen_level_raises_at_scoring(toy_dev):
    # D-023: a level never seen in training must not be scored silently as the reference
    X, y = toy_dev[config.MAIN_MODEL_FEATURES], toy_dev[config.TARGET_COL]
    pipeline = model.build_pipeline().fit(X, y)
    new = X.head(1).copy()
    new["loan_type"] = "type9"
    with pytest.raises(ValueError):
        pipeline.predict_proba(new)


def test_expected_sign_coverage_matches_fitted_names(toy_dev):
    X, y = toy_dev[config.MAIN_MODEL_FEATURES], toy_dev[config.TARGET_COL]
    names = model.feature_names(model.build_pipeline().fit(X, y))
    model.check_expected_sign_coverage(names)  # the real names match EXPECTED_SIGNS exactly
    with pytest.raises(ValueError, match="EXPECTED_SIGNS"):
        model.check_expected_sign_coverage(names + ["loan_type_type9"])
    with pytest.raises(ValueError, match="EXPECTED_SIGNS"):
        model.check_expected_sign_coverage(names[1:])


def test_coefficient_table_reports_every_feature(toy_dev):
    X, y = toy_dev[config.MAIN_MODEL_FEATURES], toy_dev[config.TARGET_COL]
    _, cv_coefficients = model.cross_validate_model(model.build_pipeline(), X, y)
    pipeline = model.build_pipeline().fit(X, y)
    table = model.coefficient_table(pipeline, X, y, cv_coefficients)
    assert set(table["feature"]) == set(config.EXPECTED_SIGNS)
    assert table["cv_sign_share"].between(0, 1).all()
    # two independent MLE fits; on 200 rows lbfgs stops within ~0.01 of statsmodels
    np.testing.assert_allclose(table["coefficient"], table["sm_coefficient"], atol=0.05)


def test_leakage_full_model_keeps_missingness_visible(toy_con):
    # D-024: the full demo model must see the missingness that carries the leakage
    dev = model.load_development_data(toy_con, view=config.LEAKAGE_DEMO_VIEW)
    X = dev[config.MAIN_MODEL_FEATURES + config.LEAKAGE_DEMO_EXTRA_FEATURES].copy()
    X.loc[X.index[:5], "Interest_rate_spread"] = np.nan
    X.loc[X.index[:5], "age"] = np.nan
    preprocessor = model._leakage_full_pipeline()["preprocess"].fit(X)
    names = list(preprocessor.get_feature_names_out())
    assert "extra_missing__missingindicator_Interest_rate_spread" in names
    # missing age is its own level (here it sorts first and is the dropped reference,
    # so it shows as all-zero age columns): missing rows must be distinguishable
    design = pd.DataFrame(preprocessor.transform(X), columns=names)
    age_cols = [n for n in names if n.startswith("extra_cat__age_")]
    missing = X["age"].isna().to_numpy()
    assert age_cols
    assert not np.array_equal(design.loc[missing, age_cols].iloc[0], design.loc[~missing, age_cols].iloc[0])
    # built on the main model's own preprocessing, not a copy that could drift from it
    assert set(model.build_preprocessor().fit(X).get_feature_names_out()) <= set(names)


def test_run_stage5_writes_only_to_the_given_folders(toy_con, tmp_path):
    db_path = tmp_path / "toy.duckdb"
    toy_con.close()
    committed = {p: p.stat().st_mtime_ns for p in config.ARTIFACTS_DIR.glob("model_*.csv")}
    out = tmp_path / "out"
    model.run_stage5(db_path, output_dir=out, model_path=out / "model.joblib", verbose=False)
    for path in [config.MODEL_SCREENING_PATH, config.MODEL_CV_METRICS_PATH, config.MODEL_COEFFICIENTS_PATH]:
        assert (out / path.name).exists()
    assert (out / "model.joblib").exists()
    assert {p: p.stat().st_mtime_ns for p in config.ARTIFACTS_DIR.glob("model_*.csv")} == committed


def test_predicted_probabilities_lie_in_unit_interval(toy_dev):
    X, y = toy_dev[config.MAIN_MODEL_FEATURES], toy_dev[config.TARGET_COL]
    pipeline = model.build_pipeline()
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_cross_validate_model_returns_one_row_per_fold_plus_mean(toy_dev):
    X, y = toy_dev[config.MAIN_MODEL_FEATURES], toy_dev[config.TARGET_COL]
    metrics, coefficients = model.cross_validate_model(model.build_pipeline(), X, y)
    assert list(metrics["fold"]) == [1, 2, 3, 4, 5, "mean"]
    assert {"auc", "gini", "ks", "brier"} <= set(metrics.columns)
    assert set(coefficients["fold"]) == {1, 2, 3, 4, 5}


def test_information_value_is_higher_for_a_more_predictive_feature(toy_dev):
    # lump_sum_payment was built into the toy score; total_units is pure noise (constant, in fact)
    assert eda.information_value(toy_dev, "lump_sum_payment") > eda.information_value(toy_dev, "total_units")


# ---------------------------------------------------------------------------
# Tests on the real data (skipped if the raw file has not been downloaded)
# ---------------------------------------------------------------------------

needs_raw_data = pytest.mark.skipif(
    not config.RAW_DATA_PATH.exists(),
    reason="data/raw/Loan_Default.csv not present (it is not committed, see D-002)",
)


@pytest.fixture(scope="module")
def real_build(tmp_path_factory):
    """Rebuild everything from the raw CSV into a temporary folder; nothing committed is touched."""
    folder = tmp_path_factory.mktemp("stage5")
    parquet = folder / "loans_clean.parquet"
    dp.run_pipeline(output_path=parquet, report_dir=folder, verbose=False)
    db_path = folder / "credit_risk.duckdb"
    db.build_database(db_path, parquet, folder, verbose=False)
    results = model.run_stage5(db_path, output_dir=folder, model_path=folder / "pd_model.joblib", verbose=False)
    return db_path, results


@pytest.fixture(scope="module")
def real_stage5(real_build):
    return real_build[1]


@pytest.fixture(scope="module")
def real_screening_data(real_build):
    con = db.connect(real_build[0])
    try:
        return model.load_screening_data(con)
    finally:
        con.close()


@needs_raw_data
def test_real_development_sample_size(real_screening_data):
    assert len(real_screening_data) == 104_069


@needs_raw_data
def test_real_screening_matches_stage5_decisions(real_stage5):
    # D-022: term and co-applicant_credit_type are dropped for cause, not for low IV
    screening = real_stage5["screening"].set_index("feature")
    assert not screening.loc["term", "kept"]
    assert not screening.loc["co-applicant_credit_type", "kept"]
    assert screening.loc["term", "iv_outside_equi"] > config.IV_MIN  # dropped despite passing IV
    # the 6 kept features all clear the IV_MIN screen outside EQUI
    kept = screening.loc[screening["kept"]]
    assert set(kept.index) == set(config.MAIN_MODEL_FEATURES)
    assert (kept["iv_outside_equi"] >= config.IV_MIN).all()


@needs_raw_data
def test_real_term_300_neg_amm_cell_is_the_anomaly(real_screening_data):
    # the finding behind dropping `term` (D-022): reproduced directly, not just via IV
    dev = real_screening_data
    outside_equi = dev[dev["credit_type"] != config.EQUI_LEVEL]
    cell = outside_equi[(outside_equi["term"] == 300) & (outside_equi["Neg_ammortization"] == "neg_amm")]
    assert len(cell) == 579
    assert cell[config.TARGET_COL].mean() == pytest.approx(0.914, abs=0.001)


@needs_raw_data
def test_real_cv_auc_ranks_main_below_leakage_models(real_stage5):
    # D-024: both leakage models look (near) perfect, far above the honest main model.
    # The full model sees the same missingness as the ablation, so it is at least as good.
    cv = real_stage5["cv_metrics"].set_index(["model", "fold"])
    main_auc = cv.loc[("main", "mean"), "auc"]
    full_auc = cv.loc[("leakage_full", "mean"), "auc"]
    ablation_auc = cv.loc[("leakage_ablation_indicators_only", "mean"), "auc"]
    assert 0.5 < main_auc < 0.75
    assert full_auc > 0.99 and ablation_auc > 0.99


@needs_raw_data
def test_real_cv_metrics_are_reproducible(real_build, tmp_path):
    # same data, same seed -> identical folds and metrics (rows are loaded ordered by ID)
    db_path, first = real_build
    again = model.run_stage5(db_path, output_dir=tmp_path, model_path=tmp_path / "m.joblib", verbose=False)
    pd.testing.assert_frame_equal(first["cv_metrics"], again["cv_metrics"])


@needs_raw_data
def test_real_unambiguous_coefficient_signs_match_expectation(real_stage5):
    # income_clean, lump_sum_payment and Neg_ammortization have an unambiguous credit
    # rationale (D-022); loan_type/loan_purpose levels and loan_amount are informational
    # only, and any mismatch there is reported for discussion, not asserted here.
    coefficients = real_stage5["coefficients"].set_index("feature")
    for feature in [config.INCOME_CLEAN_COL, "lump_sum_payment_lpsm", "Neg_ammortization_neg_amm"]:
        assert coefficients.loc[feature, "sign_matches_expected"]
