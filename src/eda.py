"""Stage 3 exploratory-analysis helpers, used by notebooks/01_eda.ipynb.

Only small, reusable calculations live here, so they can be tested and reused later
(for example when screening features in Stage 5). Plots stay in the notebook.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config


def default_rate_by_bin(df: pd.DataFrame, col: str, n_bins: int = 10) -> pd.DataFrame:
    """Default rate per quantile bin of a numeric column, using non-missing values only.

    Bins with duplicate edges are merged, so a column with few distinct values
    (for example `term`) returns fewer than `n_bins` rows.
    """
    present = df.loc[df[col].notna(), [col, config.TARGET_COL]]
    bins = pd.qcut(present[col], n_bins, duplicates="drop")
    table = present.groupby(bins, observed=True)[config.TARGET_COL].agg(n="size", default_rate="mean")
    return table.rename_axis("bin").reset_index()


def missingness_patterns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Count rows and defaults for every combination of missing / present in `cols`.

    Each row of the result is one pattern (True = missing). This shows whether a
    combination of missing fields separates the target, which a column-by-column
    missingness table cannot show.
    """
    pattern = df[cols].isna()
    grouped = pattern.assign(**{config.TARGET_COL: df[config.TARGET_COL]}).groupby(cols)
    table = grouped[config.TARGET_COL].agg(n="size", n_default="sum", default_rate="mean")
    return table.sort_values("n", ascending=False).reset_index()


def default_rate_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rows and default rate for each level of `col`; missing values form their own level."""
    levels = df[col].astype("object").where(df[col].notna(), "<missing>")
    table = df.groupby(levels)[config.TARGET_COL].agg(n="size", default_rate="mean")
    return table.rename_axis(col).sort_values("n", ascending=False).reset_index()


def single_feature_auc(df: pd.DataFrame, col: str) -> float:
    """ROC AUC of a raw numeric column used directly as a score (non-missing rows only).

    0.5 means no ranking power; values below 0.5 mean higher values go with fewer defaults.
    """
    present = df.loc[df[col].notna(), [col, config.TARGET_COL]]
    return float(roc_auc_score(present[config.TARGET_COL], present[col]))


def _levels(df: pd.DataFrame, col: str, n_bins: int) -> pd.Series:
    """Turn a column into levels for a screening table: quantile bins for numerics with
    many distinct values, categories otherwise. Missing values form their own level,
    so a column's missingness (a leakage indicator in this dataset, D-017) is screened
    along with its values, not silently dropped.

    A numeric column is only binned if it has clearly more distinct values than
    `n_bins` (config.SCREENING_BIN_DISTINCT_FACTOR times, a rule of thumb): `term`
    takes 25 discrete values and is treated as a category, not binned, so a small
    group like `term = 300` is not diluted into a decile dominated by `term = 360`.
    """
    values = df[col]
    max_distinct = config.SCREENING_BIN_DISTINCT_FACTOR * n_bins
    if pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > max_distinct:
        values = pd.qcut(values, n_bins, duplicates="drop").astype(str)
        values = values.where(df[col].notna(), np.nan)
    return values.astype("object").where(df[col].notna(), "<missing>")


def information_value(df: pd.DataFrame, col: str, n_bins: int = config.SCREENING_N_BINS) -> float:
    """Information Value (IV) of one feature against Status: a standard univariate
    screen (HEURISTIC thresholds, see D-022). Categories are used as levels; numeric
    columns are split into quantile bins first. A small constant avoids division by
    zero when a level has no goods or no bads.

    Rules of thumb: < 0.02 not predictive, 0.02-0.1 weak, 0.1-0.3 medium, > 0.3 strong.
    These are HEURISTIC labels, not a regulatory standard.
    """
    levels = _levels(df, col, n_bins)
    grouped = df.groupby(levels)[config.TARGET_COL].agg(bad="sum", n="size")
    good = grouped["n"] - grouped["bad"]
    bad_rate = (grouped["bad"] + 0.5) / (grouped["bad"].sum() + 0.5)
    good_rate = (good + 0.5) / (good.sum() + 0.5)
    return float(((good_rate - bad_rate) * np.log(good_rate / bad_rate)).sum())
