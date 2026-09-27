"""Tests for feature importance module.

Uses synthetic data where feature importance is known by construction:
informative features are generated with a clear relationship to the
target, while noise features are random.  This lets us verify that
each importance method correctly identifies the signal.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from tbi_markov.feature_importance import (
    BootstrapStabilityResult,
    DropColumnResult,
    ImportanceSummary,
    PermutationImportanceResult,
    bootstrap_feature_stability,
    correlation_adjusted_importance,
    drop_column_importance,
    importance_summary,
    permutation_importance,
)


# ------------------------------------------------------------------
# Helpers: synthetic data with known structure
# ------------------------------------------------------------------

def _make_data(
    n: int = 200,
    n_informative: int = 2,
    n_noise: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Create binary classification data with known informative features.

    Features 0..n_informative-1 carry the signal; the rest are noise.
    Returns X, y, n_informative, n_noise.
    """
    rng = np.random.default_rng(seed)
    n_features = n_informative + n_noise
    X = rng.standard_normal((n, n_features))

    # Target is a logistic function of the first n_informative features.
    logits = X[:, :n_informative].sum(axis=1)
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < prob).astype(int)

    return X, y, n_informative, n_noise


def _make_model_factory(seed: int = 42):
    """Return a factory for L1-regularized logistic regression."""
    def factory():
        return LogisticRegression(
            l1_ratio=1.0,
            solver="saga",
            C=1.0,
            max_iter=2000,
            random_state=seed,
        )
    return factory


def _fit_model(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Fit and return a logistic regression model."""
    clf = LogisticRegression(
        l1_ratio=0.0,
        solver="lbfgs",
        C=1.0,
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf


# ------------------------------------------------------------------
# Permutation importance
# ------------------------------------------------------------------

class TestPermutationImportance:
    def test_returns_correct_type(self) -> None:
        X, y, _, _ = _make_data()
        model = _fit_model(X, y)
        result = permutation_importance(model, X, y, n_repeats=5)
        assert isinstance(result, PermutationImportanceResult)

    def test_shape_matches_features(self) -> None:
        X, y, _, _ = _make_data(n_informative=2, n_noise=3)
        model = _fit_model(X, y)
        result = permutation_importance(model, X, y, n_repeats=5)
        assert result.importances_mean.shape == (5,)
        assert result.importances_std.shape == (5,)
        assert result.importances_raw.shape == (5, 5)

    def test_informative_features_rank_higher(self) -> None:
        """Informative features should have higher importance than noise."""
        X, y, n_info, n_noise = _make_data(n=300, n_informative=2, n_noise=3)
        model = _fit_model(X, y)
        result = permutation_importance(model, X, y, n_repeats=20)

        info_importance = result.importances_mean[:n_info].min()
        noise_importance = result.importances_mean[n_info:].max()
        assert info_importance > noise_importance

    def test_noise_features_near_zero(self) -> None:
        """Pure noise features should have importance near zero."""
        X, y, n_info, _ = _make_data(n=300, n_informative=2, n_noise=3)
        model = _fit_model(X, y)
        result = permutation_importance(model, X, y, n_repeats=20)

        for j in range(n_info, X.shape[1]):
            assert abs(result.importances_mean[j]) < 0.05

    def test_reproducible_with_same_seed(self) -> None:
        X, y, _, _ = _make_data()
        model = _fit_model(X, y)
        r1 = permutation_importance(model, X, y, seed=0)
        r2 = permutation_importance(model, X, y, seed=0)
        np.testing.assert_array_equal(r1.importances_mean, r2.importances_mean)

    def test_different_seed_gives_different_results(self) -> None:
        X, y, _, _ = _make_data()
        model = _fit_model(X, y)
        r1 = permutation_importance(model, X, y, seed=0)
        r2 = permutation_importance(model, X, y, seed=99)
        assert not np.allclose(r1.importances_raw, r2.importances_raw)


# ------------------------------------------------------------------
# Drop-column importance
# ------------------------------------------------------------------

class TestDropColumnImportance:
    def test_returns_correct_type(self) -> None:
        X, y, _, _ = _make_data(n=100)
        factory = _make_model_factory()
        splits = list(
            (train, test)
            for train, test in _simple_splits(X, y)
        )
        result = drop_column_importance(factory, X, y, splits)
        assert isinstance(result, DropColumnResult)

    def test_informative_features_have_positive_importance(self) -> None:
        """Removing informative features should decrease the score."""
        X, y, n_info, _ = _make_data(n=200, n_informative=2, n_noise=3)
        factory = _make_model_factory()
        splits = list(_simple_splits(X, y))
        result = drop_column_importance(factory, X, y, splits)

        for j in range(n_info):
            assert result.importances[j] > 0

    def test_full_score_is_scalar(self) -> None:
        X, y, _, _ = _make_data(n=100)
        factory = _make_model_factory()
        splits = list(_simple_splits(X, y))
        result = drop_column_importance(factory, X, y, splits)
        assert isinstance(result.full_score, float)

    def test_importances_equal_full_minus_reduced(self) -> None:
        X, y, _, _ = _make_data(n=100)
        factory = _make_model_factory()
        splits = list(_simple_splits(X, y))
        result = drop_column_importance(factory, X, y, splits)
        np.testing.assert_allclose(
            result.importances,
            result.full_score - result.reduced_scores,
        )

    def test_custom_scoring_fn(self) -> None:
        """A custom scoring function should be used instead of AUC."""
        X, y, _, _ = _make_data(n=100)
        factory = _make_model_factory()
        splits = list(_simple_splits(X, y))

        def neg_brier(y_true, y_score):
            return -float(np.mean((y_true - y_score) ** 2))

        result = drop_column_importance(factory, X, y, splits, scoring_fn=neg_brier)
        assert isinstance(result, DropColumnResult)
        # Score should be negative (negative Brier)
        assert result.full_score < 0


def _simple_splits(X, y, n_splits=3, seed=42):
    """Helper: generate stratified CV splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return skf.split(X, y)


# ------------------------------------------------------------------
# Bootstrap stability
# ------------------------------------------------------------------

class TestBootstrapStability:
    def test_returns_correct_type(self) -> None:
        X, y, _, _ = _make_data(n=100)
        factory = _make_model_factory()
        result = bootstrap_feature_stability(factory, X, y, n_bootstrap=20)
        assert isinstance(result, BootstrapStabilityResult)

    def test_frequencies_between_zero_and_one(self) -> None:
        X, y, _, _ = _make_data(n=100)
        factory = _make_model_factory()
        result = bootstrap_feature_stability(factory, X, y, n_bootstrap=20)
        assert np.all(result.selection_frequencies >= 0.0)
        assert np.all(result.selection_frequencies <= 1.0)

    def test_clear_signal_has_high_stability(self) -> None:
        """A strong signal feature should be selected in most bootstraps."""
        rng = np.random.default_rng(42)
        n = 200
        # Feature 0 is strongly predictive; features 1-4 are noise.
        X = rng.standard_normal((n, 5))
        logits = 3.0 * X[:, 0]
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)

        factory = _make_model_factory()
        result = bootstrap_feature_stability(factory, X, y, n_bootstrap=50)
        assert result.selection_frequencies[0] > 0.6

    def test_noise_features_have_low_stability(self) -> None:
        """Pure noise features should rarely be selected by L1."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 5))
        logits = 3.0 * X[:, 0]
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)

        factory = _make_model_factory()
        result = bootstrap_feature_stability(factory, X, y, n_bootstrap=50)

        # At least some noise features should have low stability.
        noise_max = result.selection_frequencies[1:].min()
        assert noise_max < result.selection_frequencies[0]

    def test_coefficient_matrix_shape(self) -> None:
        X, y, _, _ = _make_data(n=100)
        factory = _make_model_factory()
        n_boot = 15
        result = bootstrap_feature_stability(factory, X, y, n_bootstrap=n_boot)
        assert result.coefficient_matrix.shape == (n_boot, X.shape[1])

    def test_reproducible_with_same_seed(self) -> None:
        X, y, _, _ = _make_data(n=100)
        factory = _make_model_factory()
        r1 = bootstrap_feature_stability(factory, X, y, n_bootstrap=20, seed=7)
        r2 = bootstrap_feature_stability(factory, X, y, n_bootstrap=20, seed=7)
        np.testing.assert_array_equal(
            r1.selection_frequencies, r2.selection_frequencies
        )


# ------------------------------------------------------------------
# Correlation-adjusted importance
# ------------------------------------------------------------------

class TestCorrelationAdjustedImportance:
    def test_uncorrelated_features_unchanged(self) -> None:
        """With identity correlation matrix, adjustment is a no-op."""
        scores = np.array([0.5, 0.3, 0.2])
        corr = np.eye(3)
        adjusted = correlation_adjusted_importance(scores, corr)
        np.testing.assert_allclose(adjusted, scores)

    def test_correlated_feature_downweighted(self) -> None:
        """A feature correlated with another should lose importance."""
        scores = np.array([0.5, 0.5, 0.5])
        corr = np.array([
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        adjusted = correlation_adjusted_importance(scores, corr)

        # Features 0 and 1 (correlated) should be down-weighted
        # relative to feature 2 (uncorrelated).
        assert adjusted[2] > adjusted[0]
        assert adjusted[2] > adjusted[1]

    def test_mismatched_shape_raises(self) -> None:
        scores = np.array([0.5, 0.3])
        corr = np.eye(3)
        with pytest.raises(ValueError, match="does not match"):
            correlation_adjusted_importance(scores, corr)

    def test_all_perfectly_correlated_returns_raw(self) -> None:
        """If all features are perfectly correlated, return raw scores."""
        scores = np.array([0.5, 0.3])
        corr = np.ones((2, 2))
        adjusted = correlation_adjusted_importance(scores, corr)
        np.testing.assert_allclose(adjusted, scores)

    def test_adjustment_formula(self) -> None:
        """Verify the adjustment matches the documented formula."""
        scores = np.array([0.8, 0.4, 0.2])
        corr = np.array([
            [1.0, 0.6, 0.2],
            [0.6, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])

        adjusted = correlation_adjusted_importance(scores, corr)

        # Manual computation.
        abs_corr = np.abs(corr.copy())
        np.fill_diagonal(abs_corr, 0.0)
        max_corr = abs_corr.max(axis=1)  # [0.6, 0.6, 0.3]
        uniqueness = 1.0 - max_corr       # [0.4, 0.4, 0.7]
        mean_u = np.mean(uniqueness)       # 0.5
        expected = scores * uniqueness / mean_u

        np.testing.assert_allclose(adjusted, expected)


# ------------------------------------------------------------------
# Importance summary (Borda count)
# ------------------------------------------------------------------

class TestImportanceSummary:
    def test_returns_correct_type(self) -> None:
        d = {
            "method_a": np.array([0.9, 0.1, 0.5]),
            "method_b": np.array([0.8, 0.2, 0.4]),
        }
        result = importance_summary(d)
        assert isinstance(result, ImportanceSummary)

    def test_unanimous_ranking(self) -> None:
        """When all methods agree, the consensus should match."""
        d = {
            "perm": np.array([0.9, 0.1, 0.5]),
            "drop": np.array([0.8, 0.2, 0.4]),
            "stab": np.array([0.7, 0.3, 0.6]),
        }
        # All methods rank feature 0 first, feature 2 second, feature 1 last.
        result = importance_summary(d)
        np.testing.assert_array_equal(result.feature_indices, [0, 2, 1])

    def test_borda_scores_are_mean_ranks(self) -> None:
        """Borda score should equal mean rank across methods."""
        d = {
            "a": np.array([0.5, 0.9, 0.1]),  # ranks: 1, 0, 2
            "b": np.array([0.8, 0.2, 0.6]),  # ranks: 0, 2, 1
        }
        result = importance_summary(d)

        # Feature 0: mean(1, 0) = 0.5
        # Feature 1: mean(0, 2) = 1.0
        # Feature 2: mean(2, 1) = 1.5
        np.testing.assert_allclose(result.borda_scores, [0.5, 1.0, 1.5])

    def test_single_method(self) -> None:
        """With one method, Borda scores equal that method's ranks."""
        d = {"only": np.array([0.3, 0.9, 0.1, 0.7])}
        result = importance_summary(d)
        # Ranking: [1] > [3] > [0] > [2]  =>  ranks: 2, 0, 3, 1
        np.testing.assert_array_equal(result.feature_indices, [1, 3, 0, 2])

    def test_disagreeing_methods(self) -> None:
        """When methods disagree, Borda picks the compromise."""
        d = {
            "a": np.array([1.0, 0.0, 0.5]),  # ranks: 0, 2, 1
            "b": np.array([0.0, 1.0, 0.5]),  # ranks: 2, 0, 1
        }
        result = importance_summary(d)
        # Feature 0: mean(0, 2) = 1.0
        # Feature 1: mean(2, 0) = 1.0
        # Feature 2: mean(1, 1) = 1.0
        # All tied; stable sort means original order [0, 1, 2].
        np.testing.assert_allclose(result.borda_scores, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result.feature_indices, [0, 1, 2])

    def test_per_method_ranks_present(self) -> None:
        d = {
            "perm": np.array([0.9, 0.1]),
            "drop": np.array([0.2, 0.8]),
        }
        result = importance_summary(d)
        assert "perm" in result.per_method_ranks
        assert "drop" in result.per_method_ranks

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            importance_summary({})

    def test_mismatched_lengths_raises(self) -> None:
        d = {
            "a": np.array([0.5, 0.3]),
            "b": np.array([0.5, 0.3, 0.1]),
        }
        with pytest.raises(ValueError, match="same length"):
            importance_summary(d)
