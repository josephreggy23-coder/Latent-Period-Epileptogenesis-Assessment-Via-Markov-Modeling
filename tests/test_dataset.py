"""Tests for workbook ingestion and the schema contract.

Contract tests run on small in-memory fixtures so they work without the source
workbooks. Tests that need the real workbooks skip automatically when absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_tables
from tbi_markov.common import FEATURES, TARGET, validate_dataset
from tbi_markov.dataset import (
    DEFAULT_BEHAVIOR_WORKBOOK,
    DEFAULT_LFP_WORKBOOK,
    build_tables,
    validate_tables,
)


# ---------------------------------------------------------------- schema
def test_valid_tables_pass_validation():
    lfp, outcomes, behavior = make_tables(n_per_arm=4)
    validate_dataset(lfp, outcomes, behavior)


def test_qc_flag_must_match_the_documented_rule():
    """qc_pass must reproduce the electrode-shift/noise thresholds.

    If the recorded flag and the published rule disagree, the two disagree about
    which sessions are usable, and that must surface loudly rather than silently
    changing which data are modeled.
    """
    lfp, outcomes, behavior = make_tables(n_per_arm=4)
    lfp.loc[0, "rms_noise_mv"] = 0.5  # would fail the rule; flag still True
    with pytest.raises(ValueError, match="qc_pass"):
        validate_dataset(lfp, outcomes, behavior)


def test_sessions_outside_the_window_are_rejected():
    lfp, outcomes, behavior = make_tables(n_per_arm=4)
    lfp.loc[0, "dpf"] = 7
    with pytest.raises(ValueError, match="4-6 dpf"):
        validate_dataset(lfp, outcomes, behavior)


def test_duplicate_sessions_are_rejected():
    lfp, outcomes, behavior = make_tables(n_per_arm=4)
    lfp = pd.concat([lfp, lfp.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_dataset(lfp, outcomes, behavior)


def test_nonfinite_features_are_rejected():
    lfp, outcomes, behavior = make_tables(n_per_arm=4)
    lfp.loc[0, FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_dataset(lfp, outcomes, behavior)


def test_only_lfp_features_are_model_inputs():
    """No protocol, dose, group, QC, or outcome column may be a model input."""
    lfp, _, _ = make_tables(n_per_arm=4)
    forbidden = {
        "group",
        "batch",
        "dpf",
        "days_post_tbi",
        "qc_pass",
        "measured_peak_pressure_kpa",
        "cumulative_pressure_burden_kpa_hits",
        TARGET,
    }
    assert not forbidden & set(FEATURES)
    assert set(FEATURES) <= set(lfp.columns)


# ------------------------------------------------------- source workbooks
_WORKBOOKS_PRESENT = DEFAULT_LFP_WORKBOOK.exists() and DEFAULT_BEHAVIOR_WORKBOOK.exists()
_skip_without_workbooks = pytest.mark.skipif(
    not _WORKBOOKS_PRESENT, reason="source workbooks not present"
)


@_skip_without_workbooks
def test_workbooks_normalize_and_validate():
    lfp, outcomes, behavior = build_tables()
    validate_tables(lfp, outcomes, behavior)
    validate_dataset(lfp, outcomes, behavior)

    # One behavior row per LFP session, including zero-event sessions.
    assert len(behavior) == len(lfp)
    assert behavior["dlc_behavior_abnormality_index"].notna().all()
    assert (behavior["n_scored_events"] == 0).any()
    # No LFP feature may leak into the behavior table.
    assert not set(FEATURES) & set(behavior.columns)


@_skip_without_workbooks
def test_unobserved_fish_get_na_endpoint_not_zero():
    """A fish never observed at 6 dpf has an unknown endpoint, not a negative one.

    Coding absence as 0 would pad the negative class with animals that were
    never actually checked, inflating both the negative rate and the apparent
    discrimination.
    """
    lfp, outcomes, _ = build_tables()
    observed = set(lfp.loc[lfp["dpf"] == 6, "fish_id"])
    unresolved = set(outcomes.loc[outcomes[TARGET].isna(), "fish_id"])

    assert not (unresolved & observed)
    assert outcomes[TARGET].dropna().isin([0.0, 1.0]).all()
    assert outcomes[TARGET].isna().any(), "expected at least one unobserved fish"
    resolved = set(outcomes.loc[outcomes[TARGET].notna(), "fish_id"])
    assert observed <= resolved


@_skip_without_workbooks
def test_endpoint_is_independent_of_lfp_features():
    """The forecast target must not be derivable from the model's own inputs."""
    lfp, outcomes, _ = build_tables()
    merged = lfp.merge(outcomes[["fish_id", TARGET]], on="fish_id")
    # The endpoint comes from the behavioral log, so it is constant within a
    # fish while the LFP features vary across that fish's sessions. Fish with an
    # unresolved endpoint contribute no distinct value at all (nunique == 0).
    per_fish = merged.groupby("fish_id")[TARGET].nunique()
    assert per_fish.isin([0, 1]).all()
    assert np.isfinite(merged[list(FEATURES)].to_numpy(float)).all()
