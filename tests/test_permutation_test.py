"""Tests for the permutation test module.

Covers distance metrics (stationary L1, transition matrix L1),
the permutation engine's statistical properties (p-value bounds,
null-distribution length, determinism, detection power), and the
convenience statistic functions (mean_difference, absolute_mean_difference).
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
# stationary_l1_distance
# ------------------------------------------------------------------
class TestStationaryL1Distance:
    """Tests for L1 distance between probability vectors."""

    def test_identical_distributions_give_zero(self) -> None:
        pi = np.array([0.25, 0.25, 0.5])
        assert stationary_l1_distance(pi, pi) == 0.0

    def test_maximally_different_gives_two(self) -> None:
        """[1, 0] vs [0, 1] should give L1 = 2."""
        pi_a = np.array([1.0, 0.0])
        pi_b = np.array([0.0, 1.0])
        np.testing.assert_allclose(stationary_l1_distance(pi_a, pi_b), 2.0)

    def test_disjoint_three_state(self) -> None:
        """Maximally different 3-state distributions also give 2."""
        pi_a = np.array([1.0, 0.0, 0.0])
        pi_b = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(stationary_l1_distance(pi_a, pi_b), 2.0)

    def test_known_value(self) -> None:
        pi_a = np.array([0.6, 0.4])
        pi_b = np.array([0.3, 0.7])
        # |0.6-0.3| + |0.4-0.7| = 0.3 + 0.3 = 0.6
        np.testing.assert_allclose(stationary_l1_distance(pi_a, pi_b), 0.6)

    def test_known_value_three_state(self) -> None:
        pi_a = np.array([0.5, 0.3, 0.2])
        pi_b = np.array([0.1, 0.6, 0.3])
        # |0.5-0.1| + |0.3-0.6| + |0.2-0.3| = 0.4 + 0.3 + 0.1 = 0.8
        np.testing.assert_allclose(stationary_l1_distance(pi_a, pi_b), 0.8)

    def test_result_in_valid_range(self) -> None:
        """L1 distance between probability vectors is always in [0, 2]."""
        rng = np.random.default_rng(12)
        for _ in range(50):
            raw = rng.dirichlet(np.ones(5))
            raw2 = rng.dirichlet(np.ones(5))
            d = stationary_l1_distance(raw, raw2)
            assert 0.0 <= d <= 2.0 + 1e-12

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            stationary_l1_distance(np.array([0.5, 0.5]), np.array([1.0]))

    def test_shape_mismatch_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            stationary_l1_distance(
                np.array([0.25, 0.25, 0.25, 0.25]),
                np.array([0.5, 0.5]),
            )

    def test_symmetry(self) -> None:
        pi_a = np.array([0.7, 0.2, 0.1])
        pi_b = np.array([0.1, 0.3, 0.6])
        assert stationary_l1_distance(pi_a, pi_b) == stationary_l1_distance(
            pi_b, pi_a
        )


# ------------------------------------------------------------------
# transition_matrix_l1_distance
# ------------------------------------------------------------------
class TestTransitionMatrixL1Distance:
    """Tests for row-averaged L1 distance between transition matrices."""

    def test_identical_matrices_give_zero(self) -> None:
        P = np.array([[0.9, 0.1], [0.3, 0.7]])
        assert transition_matrix_l1_distance(P, P) == 0.0

    def test_known_analytic_value(self) -> None:
        P_a = np.array([[1.0, 0.0], [0.0, 1.0]])  # identity
        P_b = np.array([[0.5, 0.5], [0.5, 0.5]])  # uniform rows
        # Row 0: |1-0.5| + |0-0.5| = 1.0
        # Row 1: |0-0.5| + |1-0.5| = 1.0
        # Mean = (1.0 + 1.0) / 2 = 1.0
        np.testing.assert_allclose(
            transition_matrix_l1_distance(P_a, P_b), 1.0
        )

    def test_known_analytic_value_asymmetric(self) -> None:
        P_a = np.array([[0.8, 0.2], [0.4, 0.6]])
        P_b = np.array([[0.6, 0.4], [0.1, 0.9]])
        # Row 0: |0.8-0.6| + |0.2-0.4| = 0.2 + 0.2 = 0.4
        # Row 1: |0.4-0.1| + |0.6-0.9| = 0.3 + 0.3 = 0.6
        # Mean = (0.4 + 0.6) / 2 = 0.5
        np.testing.assert_allclose(
            transition_matrix_l1_distance(P_a, P_b), 0.5
        )

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            transition_matrix_l1_distance(np.eye(2), np.eye(3))

    def test_symmetry(self) -> None:
        P_a = np.array([[0.9, 0.1], [0.3, 0.7]])
        P_b = np.array([[0.5, 0.5], [0.8, 0.2]])
        assert transition_matrix_l1_distance(
            P_a, P_b
        ) == transition_matrix_l1_distance(P_b, P_a)

    def test_three_state_known_value(self) -> None:
        P_a = np.eye(3)
        P_b = np.ones((3, 3)) / 3.0
        # Each row: |1 - 1/3| + |0 - 1/3| + |0 - 1/3| = 2/3 + 1/3 + 1/3 = 4/3
        # Mean = 4/3
        np.testing.assert_allclose(
            transition_matrix_l1_distance(P_a, P_b), 4.0 / 3.0
        )


# ------------------------------------------------------------------
# mean_difference
# ------------------------------------------------------------------
class TestMeanDifference:
    """Tests for the mean_difference convenience statistic."""

    def test_known_value(self) -> None:
        a = np.array([4.0, 6.0])  # mean 5
        b = np.array([1.0, 3.0])  # mean 2
        np.testing.assert_allclose(mean_difference(a, b), 3.0)

    def test_swap_groups_negates_result(self) -> None:
        a = np.array([10.0, 12.0, 11.0])
        b = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            mean_difference(a, b), -mean_difference(b, a)
        )

    def test_sign_positive_when_a_larger(self) -> None:
        a = np.array([10.0, 12.0, 11.0])
        b = np.array([1.0, 2.0, 3.0])
        assert mean_difference(a, b) > 0.0

    def test_sign_negative_when_b_larger(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 12.0, 11.0])
        assert mean_difference(a, b) < 0.0

    def test_equal_groups_give_zero(self) -> None:
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert mean_difference(a, b) == 0.0


# ------------------------------------------------------------------
# absolute_mean_difference
# ------------------------------------------------------------------
class TestAbsoluteMeanDifference:
    """Tests for the absolute_mean_difference convenience statistic."""

    def test_always_non_negative(self) -> None:
        rng = np.random.default_rng(77)
        for _ in range(20):
            a = rng.normal(size=10)
            b = rng.normal(size=10)
            assert absolute_mean_difference(a, b) >= 0.0

    def test_magnitude_matches_mean_difference(self) -> None:
        a = np.array([4.0, 6.0])
        b = np.array([1.0, 3.0])
        np.testing.assert_allclose(
            absolute_mean_difference(a, b), abs(mean_difference(a, b))
        )

    def test_symmetric(self) -> None:
        a = np.array([5.0, 6.0])
        b = np.array([1.0, 2.0])
        assert absolute_mean_difference(a, b) == absolute_mean_difference(
            b, a
        )

    def test_known_value(self) -> None:
        a = np.array([10.0, 20.0])  # mean 15
        b = np.array([3.0, 7.0])  # mean 5
        np.testing.assert_allclose(absolute_mean_difference(a, b), 10.0)


# ------------------------------------------------------------------
# Permutation test engine -- mechanics
# ------------------------------------------------------------------
class TestPermutationTestMechanics:
    """Tests for the permutation test engine's structural properties."""

    def test_p_value_bounds(self) -> None:
        """p-value must be in [1/(1+B), 1] by construction."""
        n_perm = 199
        rng = np.random.default_rng(99)
        a = rng.normal(size=10)
        b = rng.normal(size=10)

        result = permutation_test(
            a, b,
            statistic_fn=mean_difference,
            n_permutations=n_perm,
            seed=0,
        )
        min_p = 1.0 / (1 + n_perm)
        assert min_p <= result.p_value <= 1.0

    def test_minimum_p_value_achievable(self) -> None:
        """With maximally separated groups, p should equal 1/(1+B)."""
        n_perm = 99
        group_a = np.full(20, 100.0)
        group_b = np.full(20, -100.0)

        result = permutation_test(
            group_a, group_b,
            statistic_fn=mean_difference,
            n_permutations=n_perm,
            seed=0,
        )
        expected_min_p = 1.0 / (1 + n_perm)
        np.testing.assert_allclose(result.p_value, expected_min_p)

    def test_p_value_never_zero(self) -> None:
        """The conservative +1 correction ensures p > 0."""
        group_a = np.full(20, 100.0)
        group_b = np.full(20, -100.0)
        result = permutation_test(
            group_a, group_b,
            statistic_fn=mean_difference,
            n_permutations=99,
            seed=0,
        )
        assert result.p_value > 0.0

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

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed should produce identical null distributions."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        result1 = permutation_test(
            a, b, statistic_fn=mean_difference, n_permutations=100, seed=42
        )
        result2 = permutation_test(
            a, b, statistic_fn=mean_difference, n_permutations=100, seed=42
        )
        np.testing.assert_array_equal(
            result1.null_distribution, result2.null_distribution
        )
        assert result1.observed == result2.observed
        assert result1.p_value == result2.p_value

    def test_different_seed_gives_different_null(self) -> None:
        """Different seeds should (almost surely) produce different nulls."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        result1 = permutation_test(
            a, b, statistic_fn=mean_difference, n_permutations=100, seed=42
        )
        result2 = permutation_test(
            a, b, statistic_fn=mean_difference, n_permutations=100, seed=99
        )
        # Observed is the same (same data, same statistic), but nulls differ.
        assert result1.observed == result2.observed
        assert not np.array_equal(
            result1.null_distribution, result2.null_distribution
        )

    def test_result_is_dataclass(self) -> None:
        result = permutation_test(
            np.ones(5), np.zeros(5),
            statistic_fn=mean_difference,
            n_permutations=10,
            seed=0,
        )
        assert isinstance(result, PermutationResult)
        assert hasattr(result, "observed")
        assert hasattr(result, "null_distribution")
        assert hasattr(result, "p_value")
        assert hasattr(result, "n_permutations")


# ------------------------------------------------------------------
# Permutation test -- no difference (H0 true)
# ------------------------------------------------------------------
class TestPermutationTestNoDifference:
    """When groups are drawn from the same distribution, p should be large."""

    def test_same_distribution_not_significant(self) -> None:
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

    def test_identical_values_high_p(self) -> None:
        """If both groups are literally the same values, p should be 1."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = permutation_test(
            a, b,
            statistic_fn=absolute_mean_difference,
            n_permutations=199,
            seed=0,
        )
        # Observed = 0, and every permutation also gives 0 => all >= 0
        # p = (1 + 199) / (1 + 199) = 1.0
        np.testing.assert_allclose(result.p_value, 1.0)


# ------------------------------------------------------------------
# Permutation test -- clear signal (H1 true)
# ------------------------------------------------------------------
class TestPermutationTestClearSignal:
    """When groups are well-separated, p should be near the minimum."""

    def test_shifted_groups_significant(self) -> None:
        rng = np.random.default_rng(0)
        group_a = rng.normal(5.0, 0.5, size=30)
        group_b = rng.normal(0.0, 0.5, size=30)

        result = permutation_test(
            group_a, group_b,
            statistic_fn=absolute_mean_difference,
            n_permutations=499,
            seed=42,
        )
        assert result.p_value < 0.01

    def test_p_near_minimum_for_extreme_separation(self) -> None:
        """Constant groups far apart should yield minimum p = 1/(1+B)."""
        n_perm = 499
        group_a = np.full(15, 1000.0)
        group_b = np.full(15, -1000.0)

        result = permutation_test(
            group_a, group_b,
            statistic_fn=absolute_mean_difference,
            n_permutations=n_perm,
            seed=0,
        )
        expected_min = 1.0 / (1 + n_perm)
        np.testing.assert_allclose(result.p_value, expected_min)

    def test_observed_exceeds_all_permuted(self) -> None:
        """For extreme separation the observed stat exceeds every null value."""
        group_a = np.full(20, 100.0)
        group_b = np.full(20, -100.0)

        result = permutation_test(
            group_a, group_b,
            statistic_fn=mean_difference,
            n_permutations=199,
            seed=0,
        )
        assert result.observed > np.max(result.null_distribution)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------
class TestPermutationTestEdgeCases:
    """Edge-case coverage for the permutation test engine."""

    def test_n_permutations_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            permutation_test(
                np.ones(5), np.ones(5),
                statistic_fn=mean_difference,
                n_permutations=0,
            )

    def test_n_permutations_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            permutation_test(
                np.ones(5), np.ones(5),
                statistic_fn=mean_difference,
                n_permutations=-5,
            )

    def test_single_permutation(self) -> None:
        """n_permutations=1 should work and return a length-1 null."""
        result = permutation_test(
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            statistic_fn=mean_difference,
            n_permutations=1,
            seed=0,
        )
        assert len(result.null_distribution) == 1
        assert result.n_permutations == 1
        # p-value is either 1/2 or 2/2 = 1.0
        assert result.p_value in (0.5, 1.0)

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

    def test_unequal_group_sizes(self) -> None:
        """Permutation test should handle groups of different sizes."""
        rng = np.random.default_rng(33)
        a = rng.normal(10.0, 1.0, size=5)
        b = rng.normal(0.0, 1.0, size=25)

        result = permutation_test(
            a, b,
            statistic_fn=absolute_mean_difference,
            n_permutations=199,
            seed=0,
        )
        assert result.p_value < 0.05
