"""Tests for transition matrix diagnostics.

These tests verify stationary distribution computation, spectral gap,
mixing time, detailed balance, and mean first passage times using
transition matrices with known analytic properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from tbi_markov.transition_analysis import (
    DetailedBalanceResult,
    TransitionSummary,
    _validate_stochastic,
    check_detailed_balance,
    mean_first_passage_times,
    mixing_time,
    spectral_gap,
    stationary_distribution,
    summarize_transitions,
)


# ---- Known transition matrices -----------------------------------------

def _uniform_chain(K: int) -> np.ndarray:
    """Doubly stochastic matrix: every row and column sums to 1.
    Stationary distribution is uniform 1/K."""
    return np.ones((K, K)) / K


def _identity_chain(K: int) -> np.ndarray:
    """Identity: each state is absorbing. Spectral gap = 0."""
    return np.eye(K)


def _two_state_symmetric(p: float) -> np.ndarray:
    """Symmetric 2-state chain: stay with prob p, switch with prob 1-p.
    Stationary = (0.5, 0.5), spectral gap = 2*(1-p)."""
    return np.array([[p, 1 - p], [1 - p, p]])


def _three_state_directed() -> np.ndarray:
    """Directed 3-state chain: 0 -> 1 -> 2 with high probability.
    This chain violates detailed balance."""
    return np.array([
        [0.1, 0.8, 0.1],
        [0.05, 0.1, 0.85],
        [0.6, 0.15, 0.25],
    ])


# ---- Validation --------------------------------------------------------

def test_validate_rejects_nonsquare():
    with pytest.raises(ValueError, match="square"):
        _validate_stochastic(np.ones((2, 3)))


def test_validate_rejects_negative():
    P = np.array([[1.0, 0.0], [-0.5, 1.5]])
    with pytest.raises(ValueError, match="non-negative"):
        _validate_stochastic(P)


def test_validate_rejects_nonunit_rows():
    P = np.array([[0.5, 0.3], [0.4, 0.6]])
    with pytest.raises(ValueError, match="sum to 1"):
        _validate_stochastic(P)


# ---- Stationary distribution -------------------------------------------

def test_stationary_uniform_chain():
    """A doubly stochastic matrix has uniform stationary distribution."""
    pi = stationary_distribution(_uniform_chain(4))
    assert pi.shape == (4,)
    np.testing.assert_allclose(pi, 0.25, atol=1e-10)


def test_stationary_symmetric_two_state():
    """Symmetric 2-state chain has pi = (0.5, 0.5)."""
    pi = stationary_distribution(_two_state_symmetric(0.7))
    np.testing.assert_allclose(pi, [0.5, 0.5], atol=1e-10)


def test_stationary_sums_to_one():
    pi = stationary_distribution(_three_state_directed())
    assert pi.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(pi >= 0)


def test_stationary_is_fixed_point():
    """pi @ P should equal pi."""
    P = _three_state_directed()
    pi = stationary_distribution(P)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-10)


def test_stationary_single_state():
    pi = stationary_distribution(np.array([[1.0]]))
    np.testing.assert_allclose(pi, [1.0])


# ---- Spectral gap ------------------------------------------------------

def test_spectral_gap_identity():
    """Identity matrix has second eigenvalue 1, so gap = 0."""
    gap = spectral_gap(_identity_chain(3))
    assert gap == pytest.approx(0.0, abs=1e-10)


def test_spectral_gap_uniform():
    """Fully mixing matrix has all non-leading eigenvalues 0, gap = 1."""
    gap = spectral_gap(_uniform_chain(4))
    assert gap == pytest.approx(1.0, abs=1e-10)


def test_spectral_gap_symmetric_two_state():
    """For symmetric 2-state with stay prob p, gap = 2*(1-p)."""
    p = 0.8
    gap = spectral_gap(_two_state_symmetric(p))
    expected = 2 * (1 - p)
    assert gap == pytest.approx(expected, abs=1e-10)


def test_spectral_gap_between_zero_and_one():
    gap = spectral_gap(_three_state_directed())
    assert 0.0 < gap < 1.0


# ---- Mixing time -------------------------------------------------------

def test_mixing_time_uniform_is_small():
    """A fully mixing chain should reach stationarity very quickly."""
    t = mixing_time(_uniform_chain(4), eps=0.01)
    assert t < 10.0


def test_mixing_time_identity_is_infinite():
    """Identity matrix never mixes."""
    t = mixing_time(_identity_chain(3))
    assert t == float("inf")


def test_mixing_time_positive():
    t = mixing_time(_three_state_directed(), eps=0.01)
    assert t > 0


def test_mixing_time_decreases_with_larger_eps():
    """Relaxing the threshold should give a shorter mixing time."""
    P = _three_state_directed()
    t_strict = mixing_time(P, eps=0.001)
    t_loose = mixing_time(P, eps=0.1)
    assert t_loose <= t_strict


# ---- Detailed balance ---------------------------------------------------

def test_detailed_balance_symmetric_chain():
    """A symmetric 2-state chain satisfies detailed balance."""
    result = check_detailed_balance(_two_state_symmetric(0.7))
    assert isinstance(result, DetailedBalanceResult)
    assert result.is_reversible is True
    assert result.max_imbalance < 1e-10


def test_detailed_balance_directed_chain():
    """A directed chain should violate detailed balance."""
    result = check_detailed_balance(_three_state_directed())
    assert result.is_reversible is False
    assert result.max_imbalance > 1e-6


def test_detailed_balance_flow_matrix_sums():
    """Row sums of the flow matrix should equal the stationary distribution."""
    P = _three_state_directed()
    result = check_detailed_balance(P)
    pi = stationary_distribution(P)
    row_sums = result.flow_matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, pi, atol=1e-10)


def test_detailed_balance_imbalance_antisymmetric():
    """The imbalance matrix should be antisymmetric (skew-symmetric)."""
    result = check_detailed_balance(_three_state_directed())
    np.testing.assert_allclose(
        result.imbalance_matrix, -result.imbalance_matrix.T, atol=1e-14
    )


# ---- Mean first passage times ------------------------------------------

def test_mfpt_diagonal_is_zero():
    """Mean first passage time from a state to itself should be 0."""
    M = mean_first_passage_times(_three_state_directed())
    np.testing.assert_allclose(np.diag(M), 0.0, atol=1e-10)


def test_mfpt_positive_off_diagonal():
    """Off-diagonal entries should be positive for an ergodic chain."""
    M = mean_first_passage_times(_three_state_directed())
    K = M.shape[0]
    for i in range(K):
        for j in range(K):
            if i != j:
                assert M[i, j] > 0, f"M[{i},{j}] should be positive"


def test_mfpt_symmetric_two_state():
    """For a symmetric 2-state chain with stay p, MFPT = 1/(1-p)."""
    p = 0.9
    P = _two_state_symmetric(p)
    M = mean_first_passage_times(P)
    expected = 1.0 / (1.0 - p)
    assert M[0, 1] == pytest.approx(expected, rel=1e-8)
    assert M[1, 0] == pytest.approx(expected, rel=1e-8)


def test_mfpt_shape():
    K = 4
    M = mean_first_passage_times(_uniform_chain(K))
    assert M.shape == (K, K)


# ---- Summary -----------------------------------------------------------

def test_summarize_returns_all_fields():
    P = _three_state_directed()
    result = summarize_transitions(P)
    assert isinstance(result, TransitionSummary)
    assert result.stationary.shape == (3,)
    assert 0.0 < result.spectral_gap < 1.0
    assert result.mixing_time > 0
    assert isinstance(result.detailed_balance, DetailedBalanceResult)
    assert result.mfpt.shape == (3, 3)
    assert result.latent_period_steps > 0


def test_summarize_with_severity_order():
    """Severity reordering should permute the transition matrix."""
    P = _three_state_directed()
    # Reverse the state ordering.
    severity = np.array([2, 1, 0])
    result = summarize_transitions(P, severity_order=severity)

    # The latent period should reflect the reordered chain.
    assert result.latent_period_steps > 0
    assert result.stationary.sum() == pytest.approx(1.0)


def test_summarize_two_state():
    P = _two_state_symmetric(0.8)
    result = summarize_transitions(P)
    assert result.latent_period_steps > 0
    assert result.detailed_balance.is_reversible is True
