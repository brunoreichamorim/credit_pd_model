"""Guard the agreed data-quality thresholds and project paths against silent changes.

If one of these tests fails, a documented decision (docs/decision_log.md) was
changed in code. That may be intentional, but the decision log must be updated too.
"""

from src import config


def test_paths_live_inside_project():
    for path in [config.RAW_DATA_PATH, config.CLEAN_DATA_PATH, config.DUCKDB_PATH,
                 config.SQL_DIR, config.ARTIFACTS_DIR, config.DQ_REPORT_DIR, config.FIGURES_DIR]:
        assert config.PROJECT_ROOT in path.parents


def test_raw_data_location_is_as_documented():
    assert config.RAW_DATA_PATH.relative_to(config.PROJECT_ROOT).as_posix() == "data/raw/Loan_Default.csv"


def test_expected_schema():
    cols = config.EXPECTED_RAW_COLUMNS
    assert len(cols) == 34
    assert len(set(cols)) == 34, "duplicate column names in expected schema"
    assert config.ID_COL in cols
    assert config.TARGET_COL in cols
    assert set(config.PRICING_COLUMNS_UNDER_INVESTIGATION) <= set(cols)


def test_agreed_data_quality_thresholds():
    # D-005, D-009 and D-008 in docs/decision_log.md
    assert config.PROPERTY_VALUE_MIN_VALID == 10_000
    assert config.RATE_OF_INTEREST_MIN_EXCLUSIVE == 0.0
    assert config.INCOME_MIN_EXCLUSIVE == 0.0


def test_main_model_feature_scope_covers_every_column_once():
    # D-017: each processed column is a candidate, excluded with a reason, or the target
    from src.data_processing import processed_columns

    candidates = config.MAIN_MODEL_CANDIDATE_FEATURES
    excluded = set(config.EXCLUDED_FROM_MAIN_MODEL)
    assert len(candidates) == len(set(candidates)) == 16
    assert not set(candidates) & excluded
    assert set(candidates) | excluded | {config.TARGET_COL} == set(processed_columns())


def test_hold_out_settings():
    # D-013: 70% development/training set, 30% final hold-out test set
    assert config.TEST_SIZE == 0.30
    assert isinstance(config.RANDOM_SEED, int)


def test_cross_validation_settings():
    # D-013: 5-fold CV performed only within the development/training set
    assert config.CV_FOLDS == 5
    assert isinstance(config.CV_FOLDS, int)
