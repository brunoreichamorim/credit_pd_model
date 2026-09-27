"""Discrimination metrics shared by Stage 5 (cross-validation) and Stage 6 (validation).

Only small, reusable metric functions live here, kept separate from src/model.py so
Stage 6 can extend this module (calibration, PSI) without touching the model code.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def gini(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Gini coefficient = 2 * AUC - 1. Ranges from -1 to 1; 0 means no ranking power."""
    return float(2 * roc_auc_score(y_true, y_pred) - 1)


def ks_statistic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Kolmogorov-Smirnov statistic: the largest gap between the cumulative default
    and non-default distributions of the predicted score. Ranges from 0 to 1.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(np.max(tpr - fpr))


def discrimination_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """AUC, Gini, KS and Brier score for one set of predictions."""
    return {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "gini": gini(y_true, y_pred),
        "ks": ks_statistic(y_true, y_pred),
        "brier": float(brier_score_loss(y_true, y_pred)),
    }
