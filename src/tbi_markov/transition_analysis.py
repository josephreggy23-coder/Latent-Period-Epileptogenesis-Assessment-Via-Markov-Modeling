"""Transition matrix diagnostics for HMM-derived Markov chains.

After the HMM is fit, its transition matrix encodes the day-to-day
dynamics of the latent epileptogenesis states.  This module computes
properties of that matrix that answer questions central to the
biological interpretation:

1. **Stationary distribution** -- the long-run state occupancy if the
   chain ran indefinitely.  If the highest-severity state dominates
   the stationary distribution for TBI fish but not shams, that is
   evidence of an absorbing epileptogenic attractor.

2. **Mixing time** -- how many daily transitions until the chain
   approximately reaches stationarity.  A short mixing time means state
   assignment decorrelates quickly across days (the chain forgets its
   initial condition); a long mixing time means early state assignment
   carries predictive weight for later states, which is the scenario
   where the HMM adds value over a single-time-point classifier.

3. **Detailed balance** -- whether the transition matrix is reversible.
   A reversible chain has no net probability current between states.
   Epileptogenesis is directional (baseline -> irritable -> seizure),
   so a violation of detailed balance in the TBI-fit model -- and its
   absence in the sham-fit model -- is a signature of the disease
   process.

4. **Mean first passage times** -- the expected number of transitions
   to reach state j starting from state i.  For the severity-ordered
   states, the first passage time from baseline to the highest severity
   state is a Markov estimate of the latent period.

All functions accept a row-stochastic transition matrix P where
P[i, j] = Pr(state j at t+1 | state i at t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------
# Stationary distribution
# ------------------------------------------------------------------
def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a row-stochastic matrix.

    Solves pi @ P = pi subject to sum(pi) = 1 via the left eigenvector
    of P associated with eigenvalue 1.

    Parameters
    ----------
    P : ndarray, shape (K, K)
        Row-stochastic transition matrix.

    Returns
    -------
    pi : ndarray, shape (K,)
        Stationary probability vector, non-negative and summing to 1.

    Raises
    ------
    ValueError
        If P is not square or not row-stochastic.
    """
    _validate_stochastic(P)
    K = P.shape[0]

    # Solve (P^T - I) @ pi = 0 with the constraint sum(pi) = 1.
    # Replace the last equation with sum(pi) = 1.
    A = (P.T - np.eye(K)).copy()
    A[-1, :] = 1.0
    b = np.zeros(K)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)

    # Clamp numerical noise.
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    return pi


# ------------------------------------------------------------------
# Spectral gap and mixing time
# ------------------------------------------------------------------
def spectral_gap(P: np.ndarray) -> float:
    """Spectral gap: 1 minus the second-largest eigenvalue magnitude.

    The spectral gap controls the rate of convergence to the stationary
    distribution.  A gap near 0 means very slow mixing; a gap near 1
    means rapid mixing.
    """
    _validate_stochastic(P)
    eigenvalues = np.linalg.eigvals(P)
    mags = np.sort(np.abs(eigenvalues))[::-1]

    if len(mags) < 2:
        return 1.0
    # The largest eigenvalue of a stochastic matrix is 1.
    return float(1.0 - mags[1])


def mixing_time(P: np.ndarray, eps: float = 0.01) -> float:
    """Estimated mixing time in number of transitions.

    The mixing time t_mix(eps) is the smallest t such that the total
    variation distance from stationarity is below eps for any starting
    state.  Using the spectral-gap bound:

        t_mix(eps) <= (1 / gap) * log(1 / (eps * pi_min))

    where gap is the spectral gap and pi_min is the smallest stationary
    probability.  This is a conservative upper bound.

    Parameters
    ----------
    P : ndarray, shape (K, K)
        Row-stochastic transition matrix.
    eps : float
        Total-variation threshold (default 0.01).

    Returns
    -------
    Estimated number of transitions to reach eps-stationarity.
    Returns inf if the spectral gap is zero.
    """
    _validate_stochastic(P)
    gap = spectral_gap(P)
    if gap < 1e-15:
        return float("inf")

    pi = stationary_distribution(P)
    pi_min = float(np.min(pi[pi > 0]))
    return float(np.log(1.0 / (eps * pi_min)) / gap)


# ------------------------------------------------------------------
# Detailed balance (reversibility)
# ------------------------------------------------------------------
@dataclass
class DetailedBalanceResult:
    """Outcome of a detailed-balance check."""

    is_reversible: bool
    max_imbalance: float
    imbalance_matrix: np.ndarray
    flow_matrix: np.ndarray


def check_detailed_balance(
    P: np.ndarray,
    tol: float = 1e-6,
) -> DetailedBalanceResult:
    """Test whether the Markov chain satisfies detailed balance.

    Detailed balance holds if pi[i] * P[i,j] = pi[j] * P[j,i] for all
    i, j.  The flow matrix F[i,j] = pi[i] * P[i,j] gives the
    equilibrium probability current from i to j; the imbalance matrix
    is F - F^T.

    Parameters
    ----------
    P : ndarray, shape (K, K)
        Row-stochastic transition matrix.
    tol : float
        Absolute tolerance for imbalance. If the maximum absolute value
        of the imbalance matrix is below tol, the chain is considered
        reversible.

    Returns
    -------
    DetailedBalanceResult
        ``is_reversible`` is True if detailed balance holds within tol.
        ``max_imbalance`` is the largest |F[i,j] - F[j,i]|.
        ``imbalance_matrix`` is F - F^T.
        ``flow_matrix`` is the equilibrium flow pi[i] * P[i,j].
    """
    _validate_stochastic(P)
    pi = stationary_distribution(P)
    flow = pi[:, None] * P
    imbalance = flow - flow.T
    max_imb = float(np.max(np.abs(imbalance)))

    return DetailedBalanceResult(
        is_reversible=(max_imb < tol),
        max_imbalance=max_imb,
        imbalance_matrix=imbalance,
        flow_matrix=flow,
    )


# ------------------------------------------------------------------
# Mean first passage times
# ------------------------------------------------------------------
def mean_first_passage_times(P: np.ndarray) -> np.ndarray:
    """Compute the matrix of mean first passage times.

    M[i, j] is the expected number of transitions to reach state j
    starting from state i.  By convention M[i, i] = 0.

    The computation uses the fundamental matrix approach:

        M[i,j] = (Z[j,j] - Z[i,j]) / pi[j]

    where Z = (I - P + Pi)^{-1} is the fundamental matrix and Pi is
    the matrix with each row equal to the stationary distribution.

    Parameters
    ----------
    P : ndarray, shape (K, K)
        Row-stochastic transition matrix.

    Returns
    -------
    M : ndarray, shape (K, K)
        Mean first passage time matrix.
    """
    _validate_stochastic(P)
    K = P.shape[0]
    pi = stationary_distribution(P)
    Pi = np.tile(pi, (K, 1))

    # Fundamental matrix.
    Z = np.linalg.inv(np.eye(K) - P + Pi)

    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                M[i, j] = (Z[j, j] - Z[i, j]) / pi[j]

    return M


# ------------------------------------------------------------------
# Summary for severity-ordered states
# ------------------------------------------------------------------
@dataclass
class TransitionSummary:
    """Complete transition-matrix diagnostic summary."""

    stationary: np.ndarray
    spectral_gap: float
    mixing_time: float
    detailed_balance: DetailedBalanceResult
    mfpt: np.ndarray
    latent_period_steps: float


def summarize_transitions(
    P: np.ndarray,
    severity_order: np.ndarray | None = None,
    eps: float = 0.01,
) -> TransitionSummary:
    """Compute all transition diagnostics for a fitted HMM.

    Parameters
    ----------
    P : ndarray, shape (K, K)
        Row-stochastic transition matrix.
    severity_order : ndarray, optional
        Permutation mapping raw HMM state indices to severity-ordered
        indices (output of ``common.severity_mapping``).  If given, the
        transition matrix is reordered so that state 0 is the least
        severe (baseline) and state K-1 is the most severe.
    eps : float
        Total-variation threshold for mixing time.

    Returns
    -------
    TransitionSummary
        All diagnostics, including the latent-period estimate (mean
        first passage time from state 0 to state K-1 in the
        severity-ordered chain).
    """
    _validate_stochastic(P)

    if severity_order is not None:
        # Reorder to severity scale.
        P = P[np.ix_(severity_order, severity_order)]

    pi = stationary_distribution(P)
    gap = spectral_gap(P)
    t_mix = mixing_time(P, eps=eps)
    db = check_detailed_balance(P)
    mfpt = mean_first_passage_times(P)

    # Latent period: mean transitions from baseline (0) to worst (K-1).
    K = P.shape[0]
    latent_steps = float(mfpt[0, K - 1]) if K > 1 else 0.0

    return TransitionSummary(
        stationary=pi,
        spectral_gap=gap,
        mixing_time=t_mix,
        detailed_balance=db,
        mfpt=mfpt,
        latent_period_steps=latent_steps,
    )


# ------------------------------------------------------------------
# Hitting probabilities
# ------------------------------------------------------------------
def hitting_probabilities(P: np.ndarray, target: int) -> np.ndarray:
    """Probability of eventually reaching a target state from each state.

    For an irreducible chain every state is reached with probability 1.
    For a reducible chain (or one made absorbing by removing transitions
    out of the target), the hitting probability can be strictly less
    than 1 from some states.

    This solves the system h[i] = P[i, target] + sum_{j != target} P[i,j] h[j]
    for all i != target, with h[target] = 1.

    In the epileptogenesis context, setting the target to the
    highest-severity state answers: "starting from each baseline or
    intermediate state, what fraction of animals eventually reach the
    seizure state under this transition model?"

    Parameters
    ----------
    P : ndarray, shape (K, K)
        Row-stochastic transition matrix.
    target : int
        Index of the target state.

    Returns
    -------
    h : ndarray, shape (K,)
        h[i] = probability of eventually reaching ``target`` from state i.
        h[target] = 1 by definition.

    Raises
    ------
    ValueError
        If P is not square/stochastic or target is out of range.
    """
    _validate_stochastic(P)
    K = P.shape[0]
    if not 0 <= target < K:
        raise ValueError(f"target must be in [0, {K - 1}], got {target}")

    if K == 1:
        return np.array([1.0])

    # Indices of non-target states.
    others = [i for i in range(K) if i != target]
    n = len(others)

    # Build the system (I - Q) h_others = P_others[:,target]
    # where Q = P[others][:,others] is the sub-matrix of transitions
    # among non-target states.
    Q = P[np.ix_(others, others)]
    b = P[others, target]

    A_sys = np.eye(n) - Q

    # The system is singular when some non-target states form a closed
    # communicating class that cannot reach the target.  In that case
    # use least-squares, which yields h = 0 for the unreachable states
    # (the minimum-norm solution of a consistent under-determined
    # sub-system) and the correct probabilities for reachable states.
    try:
        h_others = np.linalg.solve(A_sys, b)
    except np.linalg.LinAlgError:
        h_others, _, _, _ = np.linalg.lstsq(A_sys, b, rcond=None)

    h = np.empty(K)
    h[target] = 1.0
    for idx, state in enumerate(others):
        h[state] = float(np.clip(h_others[idx], 0.0, 1.0))

    return h


def expected_hitting_time(P: np.ndarray, target: int) -> np.ndarray:
    """Expected number of steps to reach a target state from each state.

    Solves k[i] = 1 + sum_{j != target} P[i,j] k[j] for i != target,
    with k[target] = 0.

    This is related to but distinct from the mean first passage time
    matrix: it gives the column for a single target state without
    computing the full matrix, and handles reducible chains by returning
    inf when the target is not reachable.

    Parameters
    ----------
    P : ndarray, shape (K, K)
        Row-stochastic transition matrix.
    target : int
        Index of the target state.

    Returns
    -------
    k : ndarray, shape (K,)
        k[i] = expected steps to reach ``target`` from state i.
        k[target] = 0.  Returns inf for states that cannot reach target.
    """
    _validate_stochastic(P)
    K = P.shape[0]
    if not 0 <= target < K:
        raise ValueError(f"target must be in [0, {K - 1}], got {target}")

    if K == 1:
        return np.array([0.0])

    others = [i for i in range(K) if i != target]
    n = len(others)
    Q = P[np.ix_(others, others)]
    A_sys = np.eye(n) - Q

    # Check whether the system is solvable (target is reachable).
    try:
        k_others = np.linalg.solve(A_sys, np.ones(n))
    except np.linalg.LinAlgError:
        k_others = np.full(n, np.inf)

    k = np.empty(K)
    k[target] = 0.0
    for idx, state in enumerate(others):
        val = k_others[idx]
        k[state] = float(val) if np.isfinite(val) and val >= 0 else np.inf

    return k


# ------------------------------------------------------------------
# Validation helper
# ------------------------------------------------------------------
def _validate_stochastic(P: np.ndarray) -> None:
    """Check that P is a valid row-stochastic matrix."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Transition matrix must be square.")
    if P.shape[0] == 0:
        raise ValueError("Transition matrix must have at least one state.")
    if np.any(P < -1e-10):
        raise ValueError("Transition probabilities must be non-negative.")
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"Rows must sum to 1. Got row sums: {row_sums}"
        )
