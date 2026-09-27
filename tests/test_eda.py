"""Tests for the Stage 3 EDA helpers (src/eda.py), on small synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src import eda


@pytest.fixture
def toy() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan],
        "a": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0],
        config.TARGET_COL: [0, 0, 1, 1, 1, 0],
    })


def test_default_rate_by_bin_ignores_missing_values(toy):
    table = eda.default_rate_by_bin(toy, "x", n_bins=2)
    assert table["n"].sum() == 4  # the two NaN rows are excluded
    assert table["default_rate"].tolist() == [0.0, 1.0]


def test_default_rate_by_bin_merges_duplicate_edges():
    df = pd.DataFrame({"t": [360.0] * 8 + [180.0, 240.0], config.TARGET_COL: [0] * 9 + [1]})
    table = eda.default_rate_by_bin(df, "t", n_bins=10)
    assert len(table) < 10
    assert table["n"].sum() == 10


def test_missingness_patterns_counts_every_combination(toy):
    table = eda.missingness_patterns(toy, ["x", "a"]).set_index(["x", "a"])
    assert table["n"].sum() == len(toy)
    both_present = table.loc[(False, False)]
    assert both_present["n"] == 2 and both_present["n_default"] == 1
    assert table.loc[(True, False), "default_rate"] == 0.5
    assert table.loc[(False, True), "n"] == 2


def test_default_rate_by_category_includes_missing_level():
    df = pd.DataFrame({"c": ["A", "A", "B", None], config.TARGET_COL: [1, 0, 0, 1]})
    table = eda.default_rate_by_category(df, "c").set_index("c")
    assert table["n"].sum() == len(df)
    assert table.loc["A", "n"] == 2
    assert table.loc["A", "default_rate"] == 0.5
    assert table.loc["<missing>", "default_rate"] == 1.0


def test_single_feature_auc_perfect_and_reversed(toy):
    assert eda.single_feature_auc(toy, "x") == 1.0  # x ranks the 4 non-missing rows perfectly
    reversed_toy = toy.assign(x=-toy["x"])
    assert eda.single_feature_auc(reversed_toy, "x") == 0.0
