"""Tests for the Task 5 primary dose-ordering result and negative controls.

Uses small synthetic cohorts with a known planted dose-state relationship
(or, for the null check, a deliberately absent one) rather than the measured
workbooks, so the statistics themselves are pinned against a known answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tbi_markov.common import GROUPS
from tbi_markov.dose_ordering import (
    leave_one_arm_out,
    primary_dose_ordering_test,
)


def _make_outcomes(n_per_arm: int = 15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for arm_index, group in enumerate(GROUPS):
        for member in range(n_per_arm):
            rows.append({"fish_id": f"F{arm_index}{member:03d}", "group": group})
    return pd.DataFrame(rows)


def test_primary_test_recovers_a_planted_monotone_relationship():
    """State index rises deterministically with arm index; rho must be
    strongly positive and the permutation p-value must be small."""
    outcomes = _make_outcomes()
    rng = np.random.default_rng(1)
    state_index = pd.DataFrame(
        {
            "fish_id": outcomes["fish_id"],
            "mean_state_index": [
                GROUPS.index(group) + rng.normal(0, 0.15)
                for group in outcomes["group"]
            ],
        }
    )
    result = primary_dose_ordering_test(outcomes, state_index, seed=7)
    assert result["spearman_rho"] > 0.8
    assert result["one_sided_permutation_p"] < 0.01
    assert result["bootstrap_95ci"][0] > 0.5


def test_primary_test_finds_no_signal_when_state_index_is_random():
    """With no planted relationship, rho should be near zero and the
    permutation p-value should not support the directional prediction."""
    outcomes = _make_outcomes()
    rng = np.random.default_rng(2)
    state_index = pd.DataFrame(
        {
            "fish_id": outcomes["fish_id"],
            "mean_state_index": rng.normal(0, 1, len(outcomes)),
        }
    )
    result = primary_dose_ordering_test(outcomes, state_index, seed=9)
    assert abs(result["spearman_rho"]) < 0.25
    assert result["one_sided_permutation_p"] > 0.05


def test_leave_one_arm_out_returns_one_entry_per_arm_with_correct_counts():
    outcomes = _make_outcomes(n_per_arm=10)
    state_index = pd.DataFrame(
        {
            "fish_id": outcomes["fish_id"],
            "mean_state_index": [GROUPS.index(group) for group in outcomes["group"]],
        }
    )
    result = leave_one_arm_out(outcomes, state_index)
    assert set(result) == set(GROUPS)
    for arm in GROUPS:
        assert result[arm]["n_fish"] == 30  # 40 total - 10 in the excluded arm
