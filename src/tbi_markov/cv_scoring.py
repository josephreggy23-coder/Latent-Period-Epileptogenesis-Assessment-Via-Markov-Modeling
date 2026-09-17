"""Stratified cross-validation scoring utilities.

The pivot toward supervised classification on LFP features with PTZ
assay ground truth requires proper evaluation of binary classifiers
on small, imbalanced datasets. This module provides scoring utilities
that handle the specific challenges of this setting:

1. **Class imbalance**: positive rates vary across injury arms, and
   some folds may have very few positives. All scorers use stratified
   splitting and handle degenerate folds gracefully.

2. **Small sample sizes**: with ~10-30 fish per arm, each fold has
   very few samples. The module provides bootstrap confidence intervals
   and reports per-fold scores so the user can assess stability.

3. **Proper scoring rules**: in addition to AUC and balanced accuracy,
   this module computes the Brier score (a proper scoring rule that
   penalizes overconfident predictions) and calibration error.

Mathematical notes
------------------
**Brier score** for binary outcomes:

    BS = (1/n) * sum_i (p_i - y_i)^2

where p_i is the predicted probability and y_i in {0, 1}. BS = 0 for
perfect probabilistic predictions, BS = 0.25 for constant p = 0.5.

**Calibration error** (binned):

    CE = sum_b (n_b / n) * |mean(y in bin_b) - mean(p in bin_b)|

where the probability range [0, 1] is split into equal-width bins.
A well-calibrated model has CE near 0.

**Net reclassification improvement (NRI)**: when comparing two models
(e.g., HMM-derived features vs. static landmark), NRI quantifies how
often the new model correctly reclassifies cases and controls:

    NRI = (P(up|event) - P(down|event)) + (P(down|nonevent) - P(up|nonevent))

where "up" and "down" refer to risk category changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class FoldScore:
    """Metrics from a single cross-validation fold."""

    fold: int
    n_train: int
    n_test: int
    n_positive_test: int
    auc: float
    average_precision: float
    brier: float
    balanced_accuracy: float


@dataclass(frozen=True)
class CVResult:
    """Aggregated cross-validation results with uncertainty."""

    folds: list[FoldScore]
    mean_auc: float
    std_auc: float
    mean_ap: float
    std_ap: float
    mean_brier: float
    std_brier: float
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    ci_auc: tuple[float, float]

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def summary_dict(self) -> dict:
        """Return a flat dictionary suitable for JSON serialization."""
        return {
            "n_folds": self.n_folds,
            "auc_mean": round(self.mean_auc, 4),
            "auc_std": round(self.std_auc, 4),
            "auc_ci_low": round(self.ci_auc[0], 4),
            "auc_ci_high": round(self.ci_auc[1], 4),
            "ap_mean": round(self.mean_ap, 4),
            "ap_std": round(self.std_ap, 4),
            "brier_mean": round(self.mean_brier, 4),
            "brier_std": round(self.std_brier, 4),
            "balanced_accuracy_mean": round(self.mean_balanced_accuracy, 4),
            "balanced_accuracy_std": round(self.std_balanced_accuracy, 4),
        }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC, returning NaN if the fold is degenerate."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision, returning NaN if degenerate."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def score_fold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_train: np.ndarray,
    fold_index: int,
    threshold: float = 0.5,
) -> FoldScore:
    """Score a single fold's predictions.

    Parameters
    ----------
    y_true : ndarray, shape (n_test,)
        Binary ground-truth labels.
    y_score : ndarray, shape (n_test,)
        Predicted probabilities of the positive class.
    y_train : ndarray, shape (n_train,)
        Training labels (for reporting sample sizes).
    fold_index : int
        Zero-based fold number.
    threshold : float
        Decision threshold for balanced accuracy.

    Returns
    -------
    FoldScore with all metrics computed.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    y_pred = (y_score >= threshold).astype(int)

    return FoldScore(
        fold=fold_index,
        n_train=len(y_train),
        n_test=len(y_true),
        n_positive_test=int(y_true.sum()),
        auc=_safe_auc(y_true, y_score),
        average_precision=_safe_ap(y_true, y_score),
        brier=float(brier_score_loss(y_true, y_score)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
    )


def aggregate_folds(
    fold_scores: list[FoldScore],
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> CVResult:
    """Aggregate per-fold scores into summary statistics.

    Uses bootstrap resampling of fold-level AUC values to compute a
    confidence interval, since the normal approximation is unreliable
    with 5 or fewer folds.

    Parameters
    ----------
    fold_scores : list of FoldScore
        One entry per cross-validation fold.
    confidence : float
        Confidence level for the bootstrap interval (default 0.95).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for the bootstrap.

    Returns
    -------
    CVResult with means, standard deviations, and CI.
    """
    if not fold_scores:
        raise ValueError("fold_scores must not be empty")

    aucs = np.array([f.auc for f in fold_scores])
    aps = np.array([f.average_precision for f in fold_scores])
    briers = np.array([f.brier for f in fold_scores])
    ba = np.array([f.balanced_accuracy for f in fold_scores])

    # Drop NaN folds (degenerate splits) for summary statistics
    valid_aucs = aucs[np.isfinite(aucs)]
    valid_aps = aps[np.isfinite(aps)]

    # Bootstrap CI for AUC
    ci_low, ci_high = _bootstrap_ci(
        valid_aucs, confidence=confidence, n_bootstrap=n_bootstrap, seed=seed
    )

    return CVResult(
        folds=fold_scores,
        mean_auc=float(np.nanmean(aucs)),
        std_auc=float(np.nanstd(aucs, ddof=1)) if len(valid_aucs) > 1 else 0.0,
        mean_ap=float(np.nanmean(aps)),
        std_ap=float(np.nanstd(aps, ddof=1)) if len(valid_aps) > 1 else 0.0,
        mean_brier=float(np.mean(briers)),
        std_brier=float(np.std(briers, ddof=1)) if len(briers) > 1 else 0.0,
        mean_balanced_accuracy=float(np.mean(ba)),
        std_balanced_accuracy=float(np.std(ba, ddof=1)) if len(ba) > 1 else 0.0,
        ci_auc=(ci_low, ci_high),
    )


def _bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    if len(values) < 2:
        mean = float(values[0]) if len(values) == 1 else np.nan
        return (mean, mean)

    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1 - alpha)),
    )


def calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error (ECE) for binary predictions.

    Partitions the predicted probability range [0, 1] into n_bins
    equal-width bins and computes the weighted average of the absolute
    difference between observed frequency and mean predicted probability
    within each bin:

        ECE = sum_b (n_b / n) * |freq(y=1 in bin_b) - mean(p in bin_b)|

    Parameters
    ----------
    y_true : ndarray
        Binary ground-truth labels.
    y_score : ndarray
        Predicted probabilities of the positive class.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        Expected calibration error.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)

    ece = 0.0
    for lower in np.linspace(0.0, 1.0, n_bins, endpoint=False):
        upper = lower + 1.0 / n_bins
        if upper >= 1.0:
            in_bin = (y_score >= lower) & (y_score <= upper)
        else:
            in_bin = (y_score >= lower) & (y_score < upper)

        if np.any(in_bin):
            bin_freq = float(np.mean(y_true[in_bin]))
            bin_pred = float(np.mean(y_score[in_bin]))
            ece += (np.sum(in_bin) / n) * abs(bin_freq - bin_pred)

    return ece


def net_reclassification_improvement(
    y_true: np.ndarray,
    risk_old: np.ndarray,
    risk_new: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Net reclassification improvement comparing two risk models.

    NRI quantifies how often the new model correctly moves events up
    in risk and non-events down, relative to the old model:

        NRI_events = P(up | event) - P(down | event)
        NRI_nonevents = P(down | nonevent) - P(up | nonevent)
        NRI = NRI_events + NRI_nonevents

    Parameters
    ----------
    y_true : ndarray
        Binary ground-truth labels.
    risk_old : ndarray
        Predicted probabilities from the reference model.
    risk_new : ndarray
        Predicted probabilities from the new model.
    threshold : float
        Decision threshold for risk categories.

    Returns
    -------
    dict with "nri_events", "nri_nonevents", and "nri_total".
    """
    y_true = np.asarray(y_true, dtype=int)
    risk_old = np.asarray(risk_old, dtype=float)
    risk_new = np.asarray(risk_new, dtype=float)

    cat_old = (risk_old >= threshold).astype(int)
    cat_new = (risk_new >= threshold).astype(int)

    events = y_true == 1
    nonevents = y_true == 0

    n_events = events.sum()
    n_nonevents = nonevents.sum()

    if n_events == 0 or n_nonevents == 0:
        return {"nri_events": np.nan, "nri_nonevents": np.nan, "nri_total": np.nan}

    up_events = float(np.sum((cat_new > cat_old) & events)) / n_events
    down_events = float(np.sum((cat_new < cat_old) & events)) / n_events
    nri_events = up_events - down_events

    down_nonevents = float(np.sum((cat_new < cat_old) & nonevents)) / n_nonevents
    up_nonevents = float(np.sum((cat_new > cat_old) & nonevents)) / n_nonevents
    nri_nonevents = down_nonevents - up_nonevents

    return {
        "nri_events": round(nri_events, 4),
        "nri_nonevents": round(nri_nonevents, 4),
        "nri_total": round(nri_events + nri_nonevents, 4),
    }
