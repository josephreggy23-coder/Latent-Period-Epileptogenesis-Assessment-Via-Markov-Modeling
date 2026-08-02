"""Tests for the Task 3 reduced feature layer.

The allowlist, the log1p transform, and the severity ordering are exercised
against synthetic inputs with a known, hand-computed ground truth rather than
the measured workbooks, so these tests run without the source data and pin
exact expected values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tbi_markov.common import FEATURES, LOG1P_FEATURES, NONNEGATIVE_LFP_FEATURES, transform_feature_values
from tbi_markov.modeling import RISING_FEATURES, FALLING_FEATURES, severity_mapping


def test_feature_allowlist_is_exactly_the_preregistered_four_columns():
    """Locked in docs/PREREGISTRATION.md: three concepts, four columns."""
    assert FEATURES == (
        "lfp_variance_uv2",
        "lfp_kurtosis",
        "lfp_seizure_event_rate_per_h",
        "lfp_fourth_power_mean_uv4",
    )
    dropped = {"lfp_mean_uv", "lfp_skewness", "lfp_ica_complexity"}
    assert not dropped & set(FEATURES)
    # All four are log1p'd and checked non-negative; nothing is exempt.
    assert set(LOG1P_FEATURES) == set(FEATURES)
    assert set(NONNEGATIVE_LFP_FEATURES) == set(FEATURES)


def test_transform_feature_values_matches_hand_computed_log1p():
    """Synthetic frame with known constants; expected output is log1p by hand."""
    known_values = {
        "lfp_variance_uv2": 3.0,
        "lfp_kurtosis": 7.0,
        "lfp_seizure_event_rate_per_h": 1.5,
        "lfp_fourth_power_mean_uv4": 12.0,
    }
    frame = pd.DataFrame({feature: [value] for feature, value in known_values.items()})
    transformed = transform_feature_values(frame)

    expected = np.array([[np.log1p(known_values[feature]) for feature in FEATURES]])
    np.testing.assert_allclose(transformed, expected)


def test_transform_feature_values_is_monotone_increasing():
    """log1p preserves ordering: a known-larger raw value stays larger after transform."""
    frame = pd.DataFrame(
        {feature: [1.0, 5.0, 25.0] for feature in FEATURES}
    )
    transformed = transform_feature_values(frame)
    assert np.all(np.diff(transformed, axis=0) > 0)


def test_severity_mapping_recovers_known_synthetic_ordering():
    """Three synthetic states with a known severity rank, planted in scrambled
    raw-label order, must be recovered in the correct order by severity_mapping.

    Ground truth: state "low" < state "mid" < state "high" on every rising
    feature. The raw HMM label order is arbitrary (label non-identifiability),
    so the planted rows are shuffled before being handed to the function under
    test, and the function must still recover low < mid < high.
    """
    low = np.array([0.1, 0.1, 0.1, 0.1])
    mid = np.array([1.0, 1.0, 1.0, 1.0])
    high = np.array([3.0, 3.0, 3.0, 3.0])
    # Scrambled raw order: raw state 0 = mid, raw state 1 = high, raw state 2 = low.
    means = np.vstack([mid, high, low])
    ground_truth_severity_rank_by_raw = np.array([1, 2, 0])  # mid=1, high=2, low=0

    raw_to_severity, severity_to_raw, _ = severity_mapping(means)

    np.testing.assert_array_equal(raw_to_severity, ground_truth_severity_rank_by_raw)
    # severity_to_raw must be the inverse permutation: raw index 2 (low) is
    # severity-rank 0, raw index 0 (mid) is rank 1, raw index 1 (high) is rank 2.
    np.testing.assert_array_equal(severity_to_raw, [2, 0, 1])


def test_all_rising_features_are_prespecified_and_none_fall():
    """Locked choice: with the 1/f and line-length fallbacks, every remaining
    feature is expected to rise with severity, so there is no falling feature.
    """
    assert set(RISING_FEATURES) == set(FEATURES)
    assert FALLING_FEATURES == ()
