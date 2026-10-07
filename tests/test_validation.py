"""Tests for src/validation.py, on small hand-checkable examples."""

import numpy as np
import pandas as pd
import pytest

from src import config

from src import validation


def test_gini_equals_2_auc_minus_1():
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.35, 0.8]  # one pair out of order -> AUC = 0.75
    assert validation.gini(y, p) == pytest.approx(2 * 0.75 - 1)


def test_gini_is_zero_for_a_random_score():
    y = [0, 1, 0, 1]
    p = [0.5, 0.5, 0.5, 0.5]
    assert validation.gini(y, p) == pytest.approx(0.0)


def test_gini_is_one_for_perfect_separation():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert validation.gini(y, p) == pytest.approx(1.0)


def test_ks_statistic_on_perfectly_separated_scores():
    # non-defaults all score below defaults -> the CDFs never overlap, KS = 1
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert validation.ks_statistic(y, p) == pytest.approx(1.0)


def test_ks_statistic_on_identical_scores_is_zero():
    y = [0, 1, 0, 1]
    p = [0.5, 0.5, 0.5, 0.5]
    assert validation.ks_statistic(y, p) == pytest.approx(0.0)


def test_brier_score_matches_manual_computation():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.1, 0.9])
    expected = np.mean((p - y) ** 2)
    metrics = validation.discrimination_metrics(y, p)
    assert metrics["brier"] == pytest.approx(expected)


def test_discrimination_metrics_returns_all_four_keys():
    y = [0, 0, 1, 1]
    p = [0.2, 0.3, 0.6, 0.9]
    metrics = validation.discrimination_metrics(y, p)
    assert set(metrics) == {"auc", "gini", "ks", "brier"}
    assert metrics["gini"] == pytest.approx(2 * metrics["auc"] - 1)


# ---------------------------------------------------------------------------
# Stage 6: calibration, bootstrap and criteria (D-025), on simulated data
# ---------------------------------------------------------------------------

def _simulated(n=20_000, seed=1, shift=0.0, stretch=1.0):
    """True PDs p; outcomes drawn from p; predictions are p distorted on the logit scale."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.6, n)
    y = (rng.random(n) < p).astype(int)
    logit = np.log(p / (1 - p))
    pred = 1 / (1 + np.exp(-(stretch * logit + shift)))
    return y, pred


def test_calibration_in_the_large_detects_a_shifted_mean():
    y, good = _simulated()
    _, shifted = _simulated(shift=0.5)
    assert validation.calibration_in_the_large(y, good)["binomial_p"] > 0.01
    assert validation.calibration_in_the_large(y, shifted)["binomial_p"] < 1e-6


def test_calibration_slope_is_about_one_when_calibrated_and_below_one_when_overconfident():
    y, good = _simulated()
    _, overconfident = _simulated(stretch=1.6)
    assert validation.calibration_intercept_slope(y, good)["calibration_slope"] == pytest.approx(1.0, abs=0.1)
    assert validation.calibration_intercept_slope(y, overconfident)["calibration_slope"] < 0.8


def test_hosmer_lemeshow_p_value_small_only_when_miscalibrated():
    y, good = _simulated()
    _, bad = _simulated(shift=0.4)
    assert validation.hosmer_lemeshow(y, good)["hl_p_value"] > 0.001
    assert validation.hosmer_lemeshow(y, bad)["hl_p_value"] < 1e-6
    assert validation.hosmer_lemeshow(y, good)["hl_df"] == config.CALIBRATION_N_BINS - 2


def test_bootstrap_ci_is_reproducible_and_contains_the_estimate():
    y, pred = _simulated(n=2_000)
    first = validation.bootstrap_ci(y, pred, n_resamples=50, seed=3)
    second = validation.bootstrap_ci(y, pred, n_resamples=50, seed=3)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == {"auc", "gini", "ks", "brier"}
    assert ((first["ci_lower"] <= first["estimate"]) & (first["estimate"] <= first["ci_upper"])).all()


def test_traffic_light_boundaries_are_inclusive():
    assert validation.traffic_light(0.02, 0.02, 0.05, higher_is_better=False) == "green"
    assert validation.traffic_light(0.05, 0.02, 0.05, higher_is_better=False) == "amber"
    assert validation.traffic_light(0.0501, 0.02, 0.05, higher_is_better=False) == "red"
    assert validation.traffic_light(0.05, 0.05, 0.01, higher_is_better=True) == "green"
    assert validation.traffic_light(0.01, 0.05, 0.01, higher_is_better=True) == "amber"
    assert validation.traffic_light(0.009, 0.05, 0.01, higher_is_better=True) == "red"


def test_range_light():
    green, amber = (0.9, 1.1), (0.8, 1.2)
    assert validation.range_light(1.1, green, amber) == "green"
    assert validation.range_light(0.85, green, amber) == "amber"
    assert validation.range_light(1.3, green, amber) == "red"


def test_assess_criteria_gives_one_status_per_criterion():
    table = validation.assess_criteria(0.65, 0.64, 0.5, 1.0, 0.01)
    assert len(table) == 4
    assert set(table["status"]) == {"green"}
    worst = validation.assess_criteria(0.65, 0.55, 0.001, 0.5, 0.2)
    assert set(worst["status"]) == {"red"}
