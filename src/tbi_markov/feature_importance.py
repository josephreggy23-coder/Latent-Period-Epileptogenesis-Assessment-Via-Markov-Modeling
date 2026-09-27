"""Feature importance for the LFP classification pipeline.

Multiple complementary views of feature importance are needed because
no single method captures the full picture:

- **Permutation importance** is model-agnostic and detects both linear
  and nonlinear dependencies, but spreads importance across correlated
  features (correlation blindness).

- **Drop-column importance** retrains without each feature, capturing
  the unique contribution even when features are correlated, at
  the cost of O(p) model fits.

- **Bootstrap stability selection** (Meinshausen & Buhlmann, 2010)
  assesses whether a feature is reliably selected across resamples,
  guarding against selection instability on small datasets.

- **Correlation-adjusted importance** redistributes raw importance
  scores to account for inter-feature correlations, concentrating
  credit on the least-redundant features.

- **Borda rank aggregation** produces a consensus ranking across
  methods, reducing the influence of any single method's pathology.

Mathematical notes
------------------
**Permutation importance** for feature j:

    I_j = (1/R) * sum_{r=1}^{R} [s(y, f(X)) - s(y, f(X_perm_j^r))]

where s is the scoring metric (AUC), f is the fitted model, and
X_perm_j^r is the data with column j randomly permuted in repeat r.

**Drop-column importance** for feature j:

    D_j = s_full - s_{-j}

where s_full is the CV score with all features and s_{-j} is the
CV score with feature j removed and the model retrained.

**Bootstrap stability** for feature j:

    Pi_j = (1/B) * sum_{b=1}^{B} I(|beta_j^b| > 0)

where beta_j^b is the coefficient of feature j in bootstrap b.
Stability > 0.6 is a common threshold (Meinshausen & Buhlmann).

**Correlation adjustment**:

    adjusted_i = raw_i * (1 - max_corr_i) / mean(1 - max_corr)

where max_corr_i = max_{k != i} |corr(x_i, x_k)|. This down-weights
features that are highly correlated with others and up-weights
those with unique variance contributions.

**Borda rank aggregation**: each method ranks features 0 (best) to
p-1 (worst); the Borda score is the mean rank across methods, and
the consensus ranking sorts by Borda score (lower is better).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------

@dataclass(frozen=True)
class PermutationImportanceResult:
    """Result of permutation importance analysis.

    Attributes
    ----------
    importances_mean : ndarray, shape (n_features,)
        Mean decrease in scoring metric when each feature is permuted,
        averaged over n_repeats.
    importances_std : ndarray, shape (n_features,)
        Standard deviation of the decrease across repeats.
    importances_raw : ndarray, shape (n_repeats, n_features)
        Per-repeat importance values.
    """

    importances_mean: np.ndarray
    importances_std: np.ndarray
    importances_raw: np.ndarray


@dataclass(frozen=True)
class DropColumnResult:
    """Result of drop-column (leave-one-feature-out) importance.

    Attributes
    ----------
    importances : ndarray, shape (n_features,)
        Decrease in CV score when each feature is removed and the
        model is retrained.  Positive values mean the feature helps.
    full_score : float
        CV score with all features included.
    reduced_scores : ndarray, shape (n_features,)
        CV score with each feature removed.
    """

    importances: np.ndarray
    full_score: float
    reduced_scores: np.ndarray


@dataclass(frozen=True)
class BootstrapStabilityResult:
    """Result of bootstrap feature stability analysis.

    Attributes
    ----------
    selection_frequencies : ndarray, shape (n_features,)
        Fraction of bootstrap resamples in which each feature has a
        non-zero coefficient (Pi_j in the stability selection
        literature).
    n_bootstrap : int
        Number of bootstrap resamples performed.
    coefficient_matrix : ndarray, shape (n_bootstrap, n_features)
        Fitted coefficients in each bootstrap resample.
    """

    selection_frequencies: np.ndarray
    n_bootstrap: int
    coefficient_matrix: np.ndarray


@dataclass(frozen=True)
class ImportanceSummary:
    """Consensus feature ranking from multiple importance methods.

    Attributes
    ----------
    feature_indices : ndarray, shape (n_features,)
        Feature indices sorted by consensus rank (best first).
    borda_scores : ndarray, shape (n_features,)
        Mean rank across methods for each feature (lower is better).
    per_method_ranks : dict[str, ndarray]
        Ranks assigned by each individual method.
    """

    feature_indices: np.ndarray
    borda_scores: np.ndarray
    per_method_ranks: dict[str, np.ndarray]


# ------------------------------------------------------------------
# Permutation importance
# ------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC, returning 0.5 if degenerate."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 30,
    seed: int = 42,
) -> PermutationImportanceResult:
    """Permutation importance for a fitted classifier.

    For each feature j, shuffle its values across samples and measure
    the drop in AUC.  This is model-agnostic and detects both linear
    and nonlinear dependencies.

    Parameters
    ----------
    model
        A fitted classifier with a ``predict_proba`` method.
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Binary labels.
    n_repeats : int
        Number of permutation repeats per feature.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    PermutationImportanceResult
        Importance scores with standard deviations across repeats.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)

    n_samples, n_features = X.shape
    baseline_score = _safe_auc(y, model.predict_proba(X)[:, 1])

    raw = np.empty((n_repeats, n_features))
    for j in range(n_features):
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_score = _safe_auc(y, model.predict_proba(X_perm)[:, 1])
            raw[r, j] = baseline_score - perm_score

    return PermutationImportanceResult(
        importances_mean=raw.mean(axis=0),
        importances_std=raw.std(axis=0, ddof=1) if n_repeats > 1 else np.zeros(n_features),
        importances_raw=raw,
    )


# ------------------------------------------------------------------
# Drop-column importance
# ------------------------------------------------------------------

def drop_column_importance(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    scoring_fn=None,
) -> DropColumnResult:
    """Leave-one-feature-out importance via retraining.

    More expensive than permutation importance but avoids its
    correlation blindness: when two features carry the same signal,
    dropping one forces the model to rely on the other, correctly
    attributing non-zero importance to both.

    Parameters
    ----------
    model_factory : callable
        A zero-argument callable that returns an unfitted classifier
        with ``fit`` and ``predict_proba`` methods.
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Binary labels.
    cv_splits : list of (train_idx, test_idx) tuples
        Pre-computed cross-validation split indices.
    scoring_fn : callable, optional
        Callable(y_true, y_score) -> float.  Defaults to AUC.

    Returns
    -------
    DropColumnResult
        Importance of each feature measured by the score decrease
        when that feature is removed and the model retrained.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if scoring_fn is None:
        scoring_fn = _safe_auc

    n_features = X.shape[1]

    def _cv_score(X_sub: np.ndarray) -> float:
        scores = []
        for train_idx, test_idx in cv_splits:
            clf = model_factory()
            clf.fit(X_sub[train_idx], y[train_idx])
            prob = clf.predict_proba(X_sub[test_idx])[:, 1]
            scores.append(scoring_fn(y[test_idx], prob))
        return float(np.mean(scores))

    full_score = _cv_score(X)

    reduced_scores = np.empty(n_features)
    for j in range(n_features):
        mask = np.ones(n_features, dtype=bool)
        mask[j] = False
        reduced_scores[j] = _cv_score(X[:, mask])

    return DropColumnResult(
        importances=full_score - reduced_scores,
        full_score=full_score,
        reduced_scores=reduced_scores,
    )


# ------------------------------------------------------------------
# Bootstrap stability selection
# ------------------------------------------------------------------

def bootstrap_feature_stability(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> BootstrapStabilityResult:
    """Bootstrap selection stability (Meinshausen & Buhlmann, 2010).

    Fit the model on bootstrap resamples and track which features
    have non-zero coefficients.  The selection frequency Pi_j is
    the fraction of bootstraps in which feature j is selected.

    A stability threshold of 0.6 is commonly used: features with
    Pi_j > 0.6 are considered reliably selected.

    Parameters
    ----------
    model_factory : callable
        A zero-argument callable that returns an unfitted model with
        ``fit`` and ``coef_`` attributes (e.g., logistic regression
        with L1/elastic-net penalty).
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Binary labels.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapStabilityResult
        Selection frequencies and per-bootstrap coefficients.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)

    n_samples, n_features = X.shape
    coef_matrix = np.zeros((n_bootstrap, n_features))

    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        # Ensure both classes are present in the resample.
        if len(np.unique(y[idx])) < 2:
            continue
        clf = model_factory()
        clf.fit(X[idx], y[idx])
        coefs = np.asarray(clf.coef_).ravel()
        coef_matrix[b, :len(coefs)] = coefs

    selection_freq = np.mean(np.abs(coef_matrix) > 0, axis=0)

    return BootstrapStabilityResult(
        selection_frequencies=selection_freq,
        n_bootstrap=n_bootstrap,
        coefficient_matrix=coef_matrix,
    )


# ------------------------------------------------------------------
# Correlation-adjusted importance
# ------------------------------------------------------------------

def correlation_adjusted_importance(
    importance_scores: np.ndarray,
    feature_correlations: np.ndarray,
) -> np.ndarray:
    """Adjust raw importance scores for inter-feature correlations.

    When features are correlated, standard importance measures spread
    credit across the correlated group.  This function redistributes
    importance proportional to the unique variance each feature
    contributes:

        adjusted_i = raw_i * (1 - max_corr_i) / mean(1 - max_corr)

    where max_corr_i = max_{k != i} |corr(x_i, x_k)|.

    Features with low correlation to others are up-weighted; features
    that are highly correlated with at least one other feature are
    down-weighted.  The adjustment preserves the sum of importance
    scores.

    Parameters
    ----------
    importance_scores : ndarray, shape (n_features,)
        Raw importance scores from any method.
    feature_correlations : ndarray, shape (n_features, n_features)
        Pairwise absolute correlation matrix.  The diagonal is
        ignored (set to 0 or 1 internally).

    Returns
    -------
    ndarray, shape (n_features,)
        Correlation-adjusted importance scores.
    """
    importance_scores = np.asarray(importance_scores, dtype=float)
    feature_correlations = np.asarray(feature_correlations, dtype=float)

    n = len(importance_scores)
    if feature_correlations.shape != (n, n):
        raise ValueError(
            f"Correlation matrix shape {feature_correlations.shape} does not "
            f"match {n} features"
        )

    # Zero the diagonal so a feature's self-correlation is excluded.
    corr = np.abs(feature_correlations.copy())
    np.fill_diagonal(corr, 0.0)

    max_corr = corr.max(axis=1)  # shape (n_features,)
    uniqueness = 1.0 - max_corr
    mean_uniqueness = np.mean(uniqueness)

    if mean_uniqueness == 0.0:
        # All features perfectly correlated; return raw scores unchanged.
        return importance_scores.copy()

    adjusted = importance_scores * uniqueness / mean_uniqueness
    return adjusted


# ------------------------------------------------------------------
# Consensus ranking via Borda count
# ------------------------------------------------------------------

def importance_summary(
    importance_dict: dict[str, np.ndarray],
) -> ImportanceSummary:
    """Combine multiple importance measures via Borda rank aggregation.

    Each method produces an importance vector of length p.  Features
    are ranked within each method (rank 0 = most important), and the
    Borda score is the mean rank across methods.  The consensus
    ranking sorts by Borda score (lower is better); ties are broken
    by the original feature index.

    Parameters
    ----------
    importance_dict : dict[str, ndarray]
        Mapping from method name to importance scores, each of shape
        (n_features,).  Higher scores mean more important.

    Returns
    -------
    ImportanceSummary
        Consensus ranking with per-method ranks and Borda scores.
    """
    if not importance_dict:
        raise ValueError("importance_dict must not be empty")

    methods = list(importance_dict.keys())
    arrays = [np.asarray(importance_dict[m], dtype=float) for m in methods]
    n_features = len(arrays[0])

    for i, arr in enumerate(arrays):
        if len(arr) != n_features:
            raise ValueError(
                f"All importance arrays must have the same length; "
                f"'{methods[0]}' has {n_features} but '{methods[i]}' has {len(arr)}"
            )

    # Rank each method: highest importance gets rank 0.
    per_method_ranks: dict[str, np.ndarray] = {}
    rank_matrix = np.empty((len(methods), n_features))

    for i, (name, scores) in enumerate(zip(methods, arrays)):
        # argsort of -scores gives indices from highest to lowest.
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(n_features, dtype=float)
        ranks[order] = np.arange(n_features, dtype=float)
        per_method_ranks[name] = ranks
        rank_matrix[i] = ranks

    borda_scores = rank_matrix.mean(axis=0)
    # Sort by Borda score (ascending = best first), stable for tie-breaking.
    consensus_order = np.argsort(borda_scores, kind="stable")

    return ImportanceSummary(
        feature_indices=consensus_order,
        borda_scores=borda_scores,
        per_method_ranks=per_method_ranks,
    )
