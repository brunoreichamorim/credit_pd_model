"""Stage 3 exploratory-analysis helpers, used by notebooks/01_eda.ipynb.

Only small, reusable calculations live here, so they can be tested and reused later
(for example when screening features in Stage 5). Plots stay in the notebook.
"""

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
