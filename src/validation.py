"""Discrimination metrics shared by Stage 5 (cross-validation) and Stage 6 (validation).

Only small, reusable metric functions live here, kept separate from src/model.py so
Stage 6 can extend this module (calibration, PSI) without touching the model code.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest, chi2
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from src import config


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


# ---------------------------------------------------------------------------
# Stage 6: calibration, bootstrap intervals and pass/fail criteria (D-025)
# ---------------------------------------------------------------------------

_PD_CLIP = 1e-6  # keeps logit(pd) finite


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _PD_CLIP, 1 - _PD_CLIP)
    return np.log(p / (1 - p))


def calibration_in_the_large(y_true, pd_pred) -> dict[str, float]:
    """Mean PD vs observed default rate, with a two-sided binomial test of the observed
    number of defaults against the mean PD."""
    y = np.asarray(y_true)
    mean_pd = float(np.mean(pd_pred))
    test = binomtest(int(y.sum()), len(y), mean_pd)
    return {
        "n_loans": len(y),
        "mean_pd": mean_pd,
        "observed_rate": float(y.mean()),
        "binomial_p": float(test.pvalue),
    }


def calibration_intercept_slope(y_true, pd_pred) -> dict[str, float]:
    """Calibration slope: coefficient of logit(PD) in a logistic regression of the
    outcome on logit(PD) (1 = perfect). Intercept: the shift needed with logit(PD)
    as a fixed offset (0 = perfect)."""
    y = np.asarray(y_true)
    score = _logit(pd_pred)
    slope_fit = sm.GLM(y, sm.add_constant(score), family=sm.families.Binomial()).fit()
    intercept_fit = sm.GLM(
        y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=score
    ).fit()
    return {
        "calibration_intercept": float(intercept_fit.params[0]),
        "calibration_slope": float(slope_fit.params[1]),
    }


def hosmer_lemeshow(y_true, pd_pred, n_bins: int = config.CALIBRATION_N_BINS) -> dict[str, float]:
    """Hosmer-Lemeshow test over `n_bins` equal-count PD bins (df = n_bins - 2).
    Reported without a pass/fail judgement: with tens of thousands of loans it
    rejects even trivial misfit (D-025)."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(pd_pred, dtype=float)})
    frame = frame.sort_values("p", kind="stable")
    chunks = np.array_split(np.arange(len(frame)), n_bins)
    frame["bin"] = np.concatenate([np.full(len(c), i) for i, c in enumerate(chunks)])
    grouped = frame.groupby("bin").agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum"))
    pbar = grouped["expected"] / grouped["n"]
    stat = float((((grouped["observed"] - grouped["expected"]) ** 2) / (grouped["n"] * pbar * (1 - pbar))).sum())
    df = n_bins - 2
    return {"hl_statistic": stat, "hl_df": df, "hl_p_value": float(chi2.sf(stat, df))}


def bootstrap_ci(
    y_true,
    pd_pred,
    n_resamples: int = config.BOOTSTRAP_N_RESAMPLES,
    level: float = config.BOOTSTRAP_CI_LEVEL,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Percentile bootstrap interval for AUC, Gini, KS and Brier: resample loans with
    replacement, recompute the metrics each time."""
    y, p = np.asarray(y_true), np.asarray(pd_pred, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        draws.append(discrimination_metrics(y[idx], p[idx]))
    draws = pd.DataFrame(draws)
    point = discrimination_metrics(y, p)
    lower, upper = (1 - level) / 2, 1 - (1 - level) / 2
    return pd.DataFrame({
        "metric": list(point),
        "estimate": list(point.values()),
        "ci_lower": draws.quantile(lower).to_numpy(),
        "ci_upper": draws.quantile(upper).to_numpy(),
    })


def traffic_light(value: float, green: float, amber: float, higher_is_better: bool) -> str:
    """'green' if `value` is within the green limit (inclusive), 'amber' if within
    the amber limit (inclusive), else 'red'."""
    if higher_is_better:
        return "green" if value >= green else "amber" if value >= amber else "red"
    return "green" if value <= green else "amber" if value <= amber else "red"


def range_light(value: float, green: tuple[float, float], amber: tuple[float, float]) -> str:
    """Traffic light for a value that must lie inside a range (both ends inclusive)."""
    if green[0] <= value <= green[1]:
        return "green"
    return "amber" if amber[0] <= value <= amber[1] else "red"


def assess_criteria(
    cv_auc: float, holdout_auc: float, binomial_p: float, slope: float, max_decile_gap: float
) -> pd.DataFrame:
    """The D-025 pass/fail table: one row per criterion with its value and status."""
    auc_drop = cv_auc - holdout_auc
    rows = [
        ("auc_drop_cv_minus_holdout", auc_drop, traffic_light(auc_drop, *config.CRITERION_AUC_DROP, False)),
        ("binomial_p", binomial_p, traffic_light(binomial_p, *config.CRITERION_BINOMIAL_P, True)),
        ("calibration_slope", slope, range_light(slope, config.CRITERION_SLOPE_GREEN, config.CRITERION_SLOPE_AMBER)),
        ("max_decile_gap", max_decile_gap, traffic_light(max_decile_gap, *config.CRITERION_DECILE_GAP, False)),
    ]
    return pd.DataFrame(rows, columns=["criterion", "value", "status"])
