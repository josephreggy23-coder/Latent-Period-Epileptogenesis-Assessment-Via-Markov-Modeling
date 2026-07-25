from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tbi_markov.common import (
    FEATURES,
    GROUPS,
    INJURY_DPF,
    NONNEGATIVE_LFP_FEATURES,
    OBSERVATION_DPF,
    PLACEHOLDER_STATUS,
    RECORD_STATUS,
    TARGET,
    validate_dataset,
)
from tbi_markov.template_data import generate_dataset, write_dataset


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
    assert manifest["data_status"] == PLACEHOLDER_STATUS
    assert set(lfp[RECORD_STATUS]) == {PLACEHOLDER_STATUS}
    assert set(outcomes[RECORD_STATUS]) == {PLACEHOLDER_STATUS}
    assert set(behavior[RECORD_STATUS]) == {PLACEHOLDER_STATUS}


def test_physical_bounds_and_qc_rule():
    lfp, _, behavior, _ = generate_dataset(seed=91, n_per_arm=20)
    assert np.isfinite(lfp[list(FEATURES)].to_numpy(float)).all()
    assert (lfp[list(NONNEGATIVE_LFP_FEATURES)] >= 0).all().all()
    assert (lfp["measured_peak_pressure_kpa"] >= 0).all()
    assert (lfp["recording_duration_min"] > 0).all()
    expected = (
        (lfp["electrode_resistance_change_pct"] <= 50.0)
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


def test_initializer_refuses_to_overwrite_existing_template(tmp_path):
    write_dataset(tmp_path, seed=42, n_per_arm=20)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_dataset(tmp_path, seed=42, n_per_arm=20)
    manifest = write_dataset(tmp_path, seed=43, n_per_arm=20, force=True)
    assert manifest["template_seed"] == 43


def test_committed_manifest_matches_normalized_tables():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "template"
    manifest = json.loads(
        (data_dir / "tbi_4_6dpf_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for item in manifest["files"].values():
        path = data_dir / item["path"]
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_validation_rejects_corrupt_domains_and_keys():
    lfp, outcomes, behavior, _ = generate_dataset(seed=42, n_per_arm=20)

    duplicate_outcomes = pd.concat([outcomes, outcomes.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="must be unique"):
        validate_dataset(lfp, duplicate_outcomes, behavior)

    invalid_target = outcomes.copy()
    invalid_target.loc[invalid_target.index[0], TARGET] = 0.5
    with pytest.raises(ValueError, match="must be binary"):
        validate_dataset(lfp, invalid_target, behavior)

    invalid_group = outcomes.copy()
    invalid_group.loc[invalid_group.index[0], "group"] = "not_an_arm"
    with pytest.raises(ValueError, match="unexpected experimental arm"):
        validate_dataset(lfp, invalid_group, behavior)

    invalid_boolean = lfp.copy()
    invalid_boolean["qc_pass"] = invalid_boolean["qc_pass"].astype(str)
    with pytest.raises(ValueError, match="true boolean"):
        validate_dataset(invalid_boolean, outcomes, behavior)

    inconsistent_group = behavior.copy()
    inconsistent_group.loc[inconsistent_group.index[0], "group"] = "tbi_high"
    with pytest.raises(ValueError, match="group must match"):
        validate_dataset(lfp, outcomes, inconsistent_group)
