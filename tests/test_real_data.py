"""Tests for the real-recording ingestion path.

These tests exercise the real-data contract on a small hand-built fixture so
they run without the source workbooks, which are large and may not be
redistributable. A separate test skips automatically when the real workbooks
are present so the actual dataset is checked when it is available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tbi_markov.common import (
    FEATURES,
    TARGET,
    TRUTH_STATE,
    validate_dataset,
)
from tbi_markov.modeling import state_recovery_metrics
from tbi_markov.real_data import (
    DEFAULT_BEHAVIOR_WORKBOOK,
    DEFAULT_LFP_WORKBOOK,
    build_real_tables,
    validate_real_tables,
)


def _synthetic_like_frame(is_synthetic: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fish = ["F1", "F2"]
    rows = []
    for fish_id in fish:
        for dpf in (4, 5, 6):
            row = {
                "fish_id": fish_id,
                "group": "sham",
                "batch": 1,
                "tbi_dpf": 3,
                "dpf": dpf,
                "days_post_tbi": dpf - 3,
                "electrode_shift_pct": 5.0,
                "rms_noise_mv": 0.05,
                "qc_pass": True,
                "measured_peak_pressure_kpa": 0.0,
                "cumulative_pressure_burden_kpa_hits": 0.0,
                "is_synthetic": is_synthetic,
            }
            for index, feature in enumerate(FEATURES):
                row[feature] = float(index + dpf)
            rows.append(row)
    lfp = pd.DataFrame(rows)
    outcomes = pd.DataFrame(
        {
            "fish_id": fish,
            "group": ["sham", "sham"],
            "batch": [1, 1],
            "survived_to_6dpf": [True, True],
            "cumulative_pressure_burden_kpa_hits": [0.0, 0.0],
            TARGET: [0, 1],
            "is_synthetic": is_synthetic,
        }
    )
    dlc = pd.DataFrame(
        {
            "fish_id": fish,
            "dpf": [6, 6],
            "dlc_mean_keypoint_likelihood": [0.95, 0.95],
            "dlc_tracking_qc_pass": [True, True],
            "is_synthetic": is_synthetic,
        }
    )
    return lfp, outcomes, dlc


def test_real_tables_rejected_when_marked_synthetic():
    """A real load must refuse rows still flagged as simulator output."""
    lfp, outcomes, dlc = _synthetic_like_frame(is_synthetic=True)
    with pytest.raises(ValueError, match="real"):
        validate_dataset(
            lfp, outcomes, dlc, expect_synthetic=False, require_truth=False
        )


def test_synthetic_load_still_rejects_real_rows():
    """The synthetic benchmark must refuse measured rows, so the two never mix."""
    lfp, outcomes, dlc = _synthetic_like_frame(is_synthetic=False)
    lfp[TRUTH_STATE] = 0
    with pytest.raises(ValueError, match="synthetic"):
        validate_dataset(lfp, outcomes, dlc)


def test_real_load_does_not_require_planted_truth():
    """Real data has no hidden_state_TRUTH and must still validate."""
    lfp, outcomes, dlc = _synthetic_like_frame(is_synthetic=False)
    assert TRUTH_STATE not in lfp.columns
    validate_dataset(
        lfp, outcomes, dlc, expect_synthetic=False, require_truth=False
    )


def test_state_recovery_is_none_without_truth():
    """State recovery must report absence, never a proxy score."""
    scored = pd.DataFrame(
        {"fish_id": ["F1"], "predicted_state": [1], "expected_state": [1.0]}
    )
    assert state_recovery_metrics(scored) is None


def test_state_recovery_scores_when_truth_present():
    scored = pd.DataFrame(
        {
            "fish_id": ["F1", "F2"],
            "predicted_state": [0, 2],
            TRUTH_STATE: [0, 2],
        }
    )
    metrics = state_recovery_metrics(scored)
    assert metrics is not None
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


_WORKBOOKS_PRESENT = DEFAULT_LFP_WORKBOOK.exists() and DEFAULT_BEHAVIOR_WORKBOOK.exists()


@pytest.mark.skipif(not _WORKBOOKS_PRESENT, reason="real source workbooks not present")
def test_real_workbooks_normalize_and_validate():
    """When the real workbooks are available, the full contract must hold."""
    lfp, outcomes, behavior = build_real_tables()
    validate_real_tables(lfp, outcomes, behavior)

    assert not lfp["is_synthetic"].any()
    assert not outcomes["is_synthetic"].any()
    assert TRUTH_STATE not in lfp.columns

    # The endpoint must be behavioural and carry both classes.
    assert outcomes[TARGET].nunique() == 2
    # One behaviour row per LFP session, including zero-event sessions.
    assert len(behavior) == len(lfp)
    assert behavior["dlc_behavior_abnormality_index"].notna().all()
    # Zero-event sessions must be present rather than silently dropped.
    assert (behavior["n_scored_events"] == 0).any()
    # No LFP feature may leak into the behaviour table.
    assert not set(FEATURES) & set(behavior.columns)


@pytest.mark.skipif(not _WORKBOOKS_PRESENT, reason="real source workbooks not present")
def test_real_endpoint_is_independent_of_lfp_features():
    """The forecast target must not be derivable from the model's own inputs."""
    lfp, outcomes, _ = build_real_tables()
    merged = lfp.merge(outcomes[["fish_id", TARGET]], on="fish_id")
    # The endpoint comes from the behavioural log, so it is constant within a
    # fish while the LFP features vary across that fish's sessions. Fish with an
    # unresolved endpoint contribute no distinct value at all (nunique == 0).
    per_fish = merged.groupby("fish_id")[TARGET].nunique()
    assert per_fish.isin([0, 1]).all()
    assert np.isfinite(merged[list(FEATURES)].to_numpy(float)).all()


@pytest.mark.skipif(not _WORKBOOKS_PRESENT, reason="real source workbooks not present")
def test_unobserved_fish_get_na_endpoint_not_zero():
    """A fish never observed at 6 dpf has an unknown endpoint, not a negative one.

    Coding absence as 0 would pad the negative class with animals that were
    never actually checked, inflating both the negative rate and the apparent
    discrimination.
    """
    lfp, outcomes, _ = build_real_tables()
    observed = set(lfp.loc[lfp["dpf"] == 6, "fish_id"])
    unresolved = set(outcomes.loc[outcomes[TARGET].isna(), "fish_id"])

    # Every unresolved fish must genuinely lack a 6 dpf LFP session.
    assert not (unresolved & observed)
    # Endpoint stays three-valued, and the negatives are real observations.
    assert outcomes[TARGET].dropna().isin([0.0, 1.0]).all()
    assert outcomes[TARGET].isna().any(), "expected at least one unobserved fish"
    # Every fish with a 6 dpf session must have a resolved endpoint.
    resolved = set(outcomes.loc[outcomes[TARGET].notna(), "fish_id"])
    assert observed <= resolved
