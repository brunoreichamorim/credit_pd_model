"""Tests for the Stage 6 runner (src/holdout.py).

Unit tests use the small synthetic dataset from tests/test_model.py. Tests on the real
data skip automatically if data/raw/Loan_Default.csv is absent (D-002).
"""

import pandas as pd
import pytest

from src import config
from src import db
from src import holdout
from src import model
from tests.test_model import toy_con, toy_processed  # noqa: F401  (pytest fixtures)


@pytest.fixture
def toy_stage6(toy_con, tmp_path):  # noqa: F811
    """Run Stage 5 then Stage 6 on the toy database, writing only to tmp_path."""
    toy_con.close()
    out = tmp_path / "out"
    model_path = out / "model.joblib"
    model.run_stage5(tmp_path / "toy.duckdb", output_dir=out, model_path=model_path, verbose=False)
    before = model_path.read_bytes()
    tables = holdout.run_stage6(tmp_path / "toy.duckdb", output_dir=out, model_path=model_path, verbose=False)
    return tables, out, model_path, before, tmp_path / "toy.duckdb"


def test_run_stage6_writes_every_artifact_to_the_given_folder(toy_stage6):
    tables, out, *_ = toy_stage6
    for name in tables:
        assert (out / f"{config.VALIDATION_ARTIFACT_PREFIX}{name}.csv").exists()


def test_run_stage6_does_not_change_the_frozen_model(toy_stage6):
    _, _, model_path, before, _ = toy_stage6
    assert model_path.read_bytes() == before


def test_pd_scores_has_every_id_once_with_valid_probabilities(toy_stage6):
    *_, db_path = toy_stage6
    con = db.connect(db_path)
    try:
        scores = con.execute(f"SELECT * FROM {config.PD_SCORES_TABLE}").df()
        n_loans = con.execute(f"SELECT count(*) FROM {config.MODEL_SCOPE_VIEW}").fetchone()[0]
    finally:
        con.close()
    assert len(scores) == n_loans
    assert scores[config.ID_COL].is_unique
    assert scores["pd"].between(0, 1).all()


def test_criteria_table_has_one_status_per_criterion(toy_stage6):
    tables, *_ = toy_stage6
    assert len(tables["criteria"]) == 4
    assert set(tables["criteria"]["status"]) <= {"green", "amber", "red"}


def test_calibration_deciles_cover_the_whole_holdout(toy_stage6):
    tables, *_ = toy_stage6
    deciles = tables["calibration_deciles"]
    holdout_rows = tables["metrics"].set_index("population").loc[holdout.HOLDOUT_POPULATION, "n_loans"]
    assert deciles["n_loans"].sum() == holdout_rows


def test_coefficient_stability_has_a_row_per_model_feature(toy_stage6):
    tables, *_ = toy_stage6
    assert len(tables["coefficient_stability"]) == len(config.EXPECTED_SIGNS)


def test_missing_model_file_fails_with_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.model"):
        holdout.load_frozen_model(tmp_path / "absent.joblib")


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

needs_real_outputs = pytest.mark.skipif(
    not (config.RAW_DATA_PATH.exists() and config.DUCKDB_PATH.exists() and config.MODEL_PATH.exists()),
    reason="needs the raw CSV and the Stage 4/5 outputs (not available on GitHub)",
)


@needs_real_outputs
def test_real_run_scores_the_documented_holdout(tmp_path):
    for path in [config.MODEL_CV_METRICS_PATH, config.MODEL_COEFFICIENTS_PATH]:
        (tmp_path / path.name).write_bytes(path.read_bytes())
    tables = holdout.run_stage6(output_dir=tmp_path, verbose=False)
    metrics = tables["metrics"].set_index("population")
    # D-026: EQUI loans (4,620 hold-out, 10,678 development) are out of scope
    assert metrics.loc[holdout.HOLDOUT_POPULATION, "n_loans"] == 44_601 - 4_620
    assert metrics.loc["development_in_sample", "n_loans"] == 104_069 - 10_678
    assert len(tables["criteria"]) == 4
    scope = tables["scope"].set_index(["sample", "scope"])["n_loans"]
    assert scope.sum() == config.EXPECTED_N_ROWS
    con = db.connect()
    try:
        n_scores = con.execute(f"SELECT count(*) FROM {config.PD_SCORES_TABLE}").fetchone()[0]
        assert n_scores == config.EXPECTED_N_ROWS - 10_678 - 4_620
    finally:
        con.close()
