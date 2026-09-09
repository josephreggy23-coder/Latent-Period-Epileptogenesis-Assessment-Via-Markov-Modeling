"""Tests for the permutation test module.

Covers distance metrics, the permutation engine's statistical
properties, and the convenience statistic functions.
"""

import numpy as np
import pytest

from tbi_markov.permutation_test import (
    PermutationResult,
    absolute_mean_difference,
    mean_difference,
    permutation_test,
    stationary_l1_distance,
    transition_matrix_l1_distance,
)


# ------------------------------------------------------------------
# Distance metrics
# ------------------------------------------------------------------
class TestStationaryL1Distance:
    def test_identical_distributions_give_zero(self) -> None:
        pi = np.array([0.25, 0.25, 0.5])
        assert stationary_l1_distance(pi, pi) == 0.0

    def test_disjoint_distributions_give_two(self) -> None:
        """Maximally different distributions: all mass in different states."""
        pi_a = np.array([1.0, 0.0, 0.0])
        pi_b = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(stationary_l1_distance(pi_a, pi_b), 2.0)

    def test_known_value(self) -> None:
        pi_a = np.array([0.6, 0.4])
        pi_b = np.array([0.3, 0.7])
        # |0.6-0.3| + |0.4-0.7| = 0.3 + 0.3 = 0.6
        np.testing.assert_allclose(stationary_l1_distance(pi_a, pi_b), 0.6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            stationary_l1_distance(np.array([0.5, 0.5]), np.array([1.0]))


class TestTransitionMatrixL1Distance:
    def test_identical_matrices_give_zero(self) -> None:
        P = np.array([[0.9, 0.1], [0.3, 0.7]])
        assert transition_matrix_l1_distance(P, P) == 0.0

    def test_known_value(self) -> None:
        P_a = np.array([[1.0, 0.0], [0.0, 1.0]])
        P_b = np.array([[0.5, 0.5], [0.5, 0.5]])
        # Row 0: |1-0.5| + |0-0.5| = 1.0
        # Row 1: |0-0.5| + |1-0.5| = 1.0
        # Mean = 1.0
        np.testing.assert_allclose(
            transition_matrix_l1_distance(P_a, P_b), 1.0
        )

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            transition_matrix_l1_distance(
                np.eye(2), np.eye(3)
            )


# ------------------------------------------------------------------
# Permutation test engine
# ------------------------------------------------------------------
class TestPermutationTest:
    def test_clearly_different_groups_give_small_p(self) -> None:
        """Two well-separated groups should produce a significant p-value."""
        rng = np.random.default_rng(0)
        group_a = rng.normal(5.0, 0.5, size=30)
        group_b = rng.normal(0.0, 0.5, size=30)

        result = permutation_test(
            group_a, group_b,
            statistic_fn=absolute_mean_difference,
            n_permutations=499,
            seed=42,
        )
        assert isinstance(result, PermutationResult)
        assert result.p_value < 0.01

    def test_identical_groups_give_large_p(self) -> None:
        """Two draws from the same distribution should not be significant."""
        rng = np.random.default_rng(7)
        group_a = rng.normal(0.0, 1.0, size=20)
        group_b = rng.normal(0.0, 1.0, size=20)

        result = permutation_test(
            group_a, group_b,
            statistic_fn=absolute_mean_difference,
            n_permutations=499,
            seed=42,
        )
        assert result.p_value > 0.05

    def test_p_value_is_valid_probability(self) -> None:
        rng = np.random.default_rng(99)
        a = rng.normal(size=10)
        b = rng.normal(size=10)

        result = permutation_test(
            a, b,
            statistic_fn=mean_difference,
            n_permutations=199,
            seed=0,
        )
        assert 0.0 < result.p_value <= 1.0

    def test_null_distribution_length(self) -> None:
        n_perm = 50
        result = permutation_test(
            np.ones(5), np.zeros(5),
            statistic_fn=mean_difference,
            n_permutations=n_perm,
            seed=0,
        )
        assert len(result.null_distribution) == n_perm
        assert result.n_permutations == n_perm

    def test_p_value_never_zero(self) -> None:
        """The conservative +1 correction ensures p > 0."""
        # Maximally separated groups: observed should exceed all permutations.
        group_a = np.full(20, 100.0)
        group_b = np.full(20, -100.0)

        result = permutation_test(
            group_a, group_b,
            statistic_fn=mean_difference,
            n_permutations=99,
            seed=0,
        )
        # With the +1 correction: p = 1/100 = 0.01, never 0.
        assert result.p_value > 0.0
        np.testing.assert_allclose(result.p_value, 1 / 100)

    def test_invalid_n_permutations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            permutation_test(
                np.ones(5), np.ones(5),
                statistic_fn=mean_difference,
                n_permutations=0,
            )

    def test_multidimensional_values(self) -> None:
        """The engine should work with per-subject feature vectors."""
        rng = np.random.default_rng(42)
        a = rng.normal(loc=3.0, size=(15, 4))
        b = rng.normal(loc=0.0, size=(15, 4))

        def row_mean_diff(x, y):
            return float(np.abs(np.mean(x) - np.mean(y)))

        result = permutation_test(
            a, b,
            statistic_fn=row_mean_diff,
            n_permutations=199,
            seed=0,
        )
        assert result.p_value < 0.05


# ------------------------------------------------------------------
# Convenience statistics
# ------------------------------------------------------------------
class TestConvenienceStatistics:
    def test_mean_difference_sign(self) -> None:
        a = np.array([10.0, 12.0, 11.0])
        b = np.array([1.0, 2.0, 3.0])
        assert mean_difference(a, b) > 0.0
        assert mean_difference(b, a) < 0.0

    def test_absolute_mean_difference_symmetric(self) -> None:
        a = np.array([5.0, 6.0])
        b = np.array([1.0, 2.0])
        assert absolute_mean_difference(a, b) == absolute_mean_difference(b, a)

    def test_mean_difference_known_value(self) -> None:
        a = np.array([4.0, 6.0])  # mean 5
        b = np.array([1.0, 3.0])  # mean 2
        np.testing.assert_allclose(mean_difference(a, b), 3.0)
