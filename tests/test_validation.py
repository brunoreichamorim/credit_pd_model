"""Tests for src/validation.py, on small hand-checkable examples."""

import numpy as np
import pytest

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
