from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from tbi_markov.common import (
    FEATURES,
    GROUPS,
    INJURY_DPF,
    NONNEGATIVE_LFP_FEATURES,
    OBSERVATION_DPF,
)
from tbi_markov.synthetic import generate_dataset, write_dataset


def _frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def test_generation_is_deterministic():
    first = generate_dataset(seed=42, n_per_arm=20)
    second = generate_dataset(seed=42, n_per_arm=20)
    for left, right in zip(first[:3], second[:3]):
        assert _frame_hash(left) == _frame_hash(right)


def test_dataset_scope_and_identifiers():
    lfp, outcomes, behavior, manifest = generate_dataset(seed=42, n_per_arm=20)
    assert set(lfp["dpf"]) == set(OBSERVATION_DPF)
    assert (lfp["tbi_dpf"] == INJURY_DPF).all()
    assert set(lfp["group"]) == set(GROUPS)
    assert not lfp.duplicated(["fish_id", "dpf"]).any()
    assert outcomes["fish_id"].is_unique
    assert set(behavior["dpf"]) == set(OBSERVATION_DPF)
    assert manifest["is_synthetic"] is True


def test_physical_bounds_and_qc_rule():
    lfp, _, behavior, _ = generate_dataset(seed=91, n_per_arm=20)
    assert np.isfinite(lfp[list(FEATURES)].to_numpy(float)).all()
    assert (lfp[list(NONNEGATIVE_LFP_FEATURES)] >= 0).all().all()
    assert (lfp["measured_peak_pressure_kpa"] >= 0).all()
    assert (lfp["recording_duration_min"] > 0).all()
    expected = (
        (lfp["electrode_shift_pct"] <= 50.0)
        & (lfp["rms_noise_mv"] < 0.2)
    )
    assert np.array_equal(expected.to_numpy(), lfp["qc_pass"].to_numpy())
    assert behavior["dlc_mean_keypoint_likelihood"].between(0, 1).all()
    assert behavior["dlc_rest_fraction"].between(0, 1).all()


def test_only_lfp_features_are_model_inputs():
    forbidden_tokens = ("TRUTH", "group", "pressure", "dlc_", "weight", "drop")
    assert all(
        not any(token.lower() in feature.lower() for token in forbidden_tokens)
        for feature in FEATURES
    )


def test_manifest_hashes_written_files(tmp_path):
    manifest = write_dataset(tmp_path, seed=42, n_per_arm=20)
    for item in manifest["files"].values():
        path = tmp_path / item["path"]
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
