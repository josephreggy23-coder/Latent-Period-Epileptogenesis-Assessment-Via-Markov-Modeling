"""Permutation tests for comparing Markov chain properties between groups.

After fitting separate HMMs to TBI and sham cohorts, the next question
is whether the observed differences in transition dynamics are
statistically significant or could arise from chance partitioning of
the fish into two groups.

Permutation framework
---------------------
Under the null hypothesis, the group label (TBI vs. sham) carries no
information about the latent epileptogenesis dynamics.  A permutation
test proceeds as follows:

1. Compute a test statistic T_obs from the real group assignments
   (e.g., the difference in spectral gaps, or the L1 distance between
   stationary distributions).

2. For B permutations, randomly shuffle the group labels among fish
   while keeping the LFP sequences intact, refit the HMMs (or
   recompute the statistic from precomputed per-fish quantities),
   and record the permuted statistic T_b.

3. The p-value is the fraction of permuted statistics at least as
   extreme as T_obs:

       p = (1 + sum_{b=1}^{B} I(T_b >= T_obs)) / (1 + B)

   The +1 in numerator and denominator avoids a p-value of exactly
   zero and accounts for the observed data as one of the (B+1)
   equally likely arrangements under the null.

Test statistics
---------------
Several scalar summaries of the transition matrix are biologically
interpretable:

- **Stationary distribution distance**: the L1 (total variation)
  distance between the stationary distributions of the two groups.
  A large distance means the long-run state occupancy differs --
  e.g., TBI fish spend more time in the highest-severity state.

- **Spectral gap difference**: the difference in spectral gaps.
  A smaller spectral gap in TBI means slower mixing, consistent
  with the chain being pulled toward an absorbing seizure state.

- **MFPT difference**: the difference in mean first passage times
  from baseline to the highest-severity state.  A shorter passage
  time in TBI means faster progression through latent states.

- **Reversibility difference**: the difference in max imbalance
  from the detailed balance check.  Larger imbalance in TBI means
  stronger directional probability current (baseline -> seizure),
  which is the signature of a disease process.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------
# Distance metrics for transition matrix properties
# ------------------------------------------------------------------
def stationary_l1_distance(
    pi_a: np.ndarray,
    pi_b: np.ndarray,
) -> float:
    """Total variation distance between two stationary distributions.

    The L1 distance between probability vectors equals twice the total
    variation distance:

        d_TV(pi_a, pi_b) = (1/2) * ||pi_a - pi_b||_1

    We return the full L1 norm (range [0, 2]) for interpretability.

    Parameters
    ----------
    pi_a, pi_b
        Stationary probability vectors (must sum to 1).

    Returns
    -------
    L1 distance in [0, 2].
    """
    pi_a = np.asarray(pi_a, dtype=float)
    pi_b = np.asarray(pi_b, dtype=float)
    if pi_a.shape != pi_b.shape:
        raise ValueError(
            f"Stationary vectors must have the same shape: "
            f"{pi_a.shape} vs {pi_b.shape}"
        )
    return float(np.sum(np.abs(pi_a - pi_b)))


def transition_matrix_l1_distance(
    P_a: np.ndarray,
    P_b: np.ndarray,
) -> float:
    """Row-averaged L1 distance between two transition matrices.

    For K-state matrices, this computes:

        d(P_a, P_b) = (1/K) * sum_i ||P_a[i,:] - P_b[i,:]||_1

    Each row difference is the L1 distance between the conditional
    distributions for state i, so the average gives a per-state
    measure of how differently the two chains behave.

    Parameters
    ----------
    P_a, P_b
        Row-stochastic transition matrices of equal dimension.

    Returns
    -------
    Mean per-row L1 distance in [0, 2].
    """
    P_a = np.asarray(P_a, dtype=float)
    P_b = np.asarray(P_b, dtype=float)
    if P_a.shape != P_b.shape:
        raise ValueError(
            f"Transition matrices must have the same shape: "
            f"{P_a.shape} vs {P_b.shape}"
        )
    K = P_a.shape[0]
    row_dists = np.sum(np.abs(P_a - P_b), axis=1)
    return float(np.mean(row_dists))


# ------------------------------------------------------------------
# Permutation test engine
# ------------------------------------------------------------------
@dataclass
class PermutationResult:
    """Result of a two-group permutation test.

    Attributes
    ----------
    observed : float
        The test statistic computed from the real group assignments.
    null_distribution : ndarray, shape (n_permutations,)
        The test statistic under each random permutation of labels.
    p_value : float
        One-sided p-value: fraction of permuted statistics >= observed,
        with the conservative +1/(1+B) correction.
    n_permutations : int
        Number of permutations performed.
    """

    observed: float
    null_distribution: np.ndarray
    p_value: float
    n_permutations: int


def permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    statistic_fn,
    n_permutations: int = 999,
    seed: int = 42,
) -> PermutationResult:
    """Two-sample permutation test with a user-supplied test statistic.

    Parameters
    ----------
    values_a
        Per-subject values for group A (e.g., TBI fish state sequences
        or per-fish summary statistics), shape (n_a, ...).
    values_b
        Per-subject values for group B (e.g., sham fish), shape (n_b, ...).
    statistic_fn
        Callable(group_a, group_b) -> float.  Computes the test
        statistic from two arrays of per-subject values.  The function
        is called once with the real partition and n_permutations times
        with shuffled partitions.
    n_permutations
        Number of random permutations (default 999, giving a minimum
        achievable p-value of 1/1000 with the +1 correction).
    seed
        Random seed for reproducibility.

    Returns
    -------
    PermutationResult
        Contains the observed statistic, null distribution, and p-value.

    Notes
    -----
    The p-value uses the conservative formula:

        p = (1 + #{T_perm >= T_obs}) / (1 + B)

    which guarantees a valid (though slightly conservative) p-value
    even when the true p is near 0.  See Phipson & Smyth (2010),
    "Permutation P-values Should Never Be Zero."
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    n_a = len(values_a)
    n_total = n_a + len(values_b)

    pooled = np.concatenate([values_a, values_b], axis=0)
    rng = np.random.default_rng(seed)

    # Observed statistic.
    t_obs = float(statistic_fn(values_a, values_b))

    # Null distribution.
    null = np.empty(n_permutations)
    for b in range(n_permutations):
        perm = rng.permutation(n_total)
        perm_a = pooled[perm[:n_a]]
        perm_b = pooled[perm[n_a:]]
        null[b] = statistic_fn(perm_a, perm_b)

    # Conservative p-value (Phipson & Smyth 2010).
    n_extreme = int(np.sum(null >= t_obs))
    p_value = (1 + n_extreme) / (1 + n_permutations)

    return PermutationResult(
        observed=t_obs,
        null_distribution=null,
        p_value=p_value,
        n_permutations=n_permutations,
    )


# ------------------------------------------------------------------
# Convenience: common two-group comparison statistics
# ------------------------------------------------------------------
def mean_difference(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Difference in group means (A - B).

    A positive value when A = TBI means the TBI group has a higher
    average of whatever per-fish quantity was passed in.
    """
    return float(np.mean(group_a) - np.mean(group_b))


def absolute_mean_difference(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Absolute difference in group means.

    Two-sided alternative: detects any shift in location regardless
    of direction.
    """
    return float(np.abs(np.mean(group_a) - np.mean(group_b)))
