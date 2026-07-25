from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from tbi_markov.common import (
    FEATURES,
    TARGET,
    build_sequences,
    fit_robust_scaler,
    qc_sessions,
)
from tbi_markov.modeling import (
    fish_level_split,
    macrostate_mapping,
    propagate_ordered_probabilities,
    run_analysis,
)
from tbi_markov.template_data import generate_dataset


def test_analysis_blocks_placeholder_records(tmp_path):
    lfp, outcomes, dlc, _ = generate_dataset(seed=50, n_per_arm=20)
    with pytest.raises(ValueError, match="placeholder records remain"):
        run_analysis(
            lfp,
            outcomes,
            dlc,
            output_dir=tmp_path,
            candidates=(3,),
            restarts=1,
            cv_folds=2,
            bootstrap_iterations=10,
        )


def test_fish_split_has_no_overlap_and_preserves_arms():
    lfp, outcomes, _, _ = generate_dataset(seed=42, n_per_arm=20)
    train_ids, test_ids, assignments = fish_level_split(outcomes, seed=42)
    assert not train_ids & test_ids
    assert train_ids | test_ids == set(outcomes["fish_id"])
    assert set(assignments.loc[assignments["split"] == "test", "group"]) == set(
        outcomes["group"]
    )
    assert set(lfp["fish_id"]) == train_ids | test_ids


def test_prefix_builder_excludes_dpf6():
    lfp, outcomes, _, _ = generate_dataset(seed=44, n_per_arm=20)
    lfp = qc_sessions(lfp)
    train_ids, test_ids, _ = fish_level_split(outcomes, seed=44)
    center, scale = fit_robust_scaler(lfp, train_ids)
    sequences, order, frames = build_sequences(
        lfp, test_ids, center, scale, cutoff_dpf=5
    )
    assert sequences
    assert order
    assert all(frame["dpf"].max() <= 5 for frame in frames.values())
    assert all(frame["dpf"].min() == 4 for frame in frames.values())
    assert all(
        np.array_equal(np.diff(frame["dpf"].to_numpy(int)), np.ones(len(frame) - 1))
        for frame in frames.values()
    )
    assert all(len(sequence) <= 2 for sequence in sequences)


def test_qc_gap_truncates_prefix_and_missing_baseline_excludes_fish():
    lfp, outcomes, _, _ = generate_dataset(seed=45, n_per_arm=20)
    lfp = qc_sessions(lfp)
    complete_ids = [
        fish_id
        for fish_id, frame in lfp.groupby("fish_id")
        if frame["dpf"].astype(int).tolist() == [4, 5, 6]
    ]
    assert len(complete_ids) >= 2
    gap_id, no_baseline_id = complete_ids[:2]
    modified = lfp.loc[
        ~(
            ((lfp["fish_id"] == gap_id) & (lfp["dpf"] == 5))
            | ((lfp["fish_id"] == no_baseline_id) & (lfp["dpf"] == 4))
        )
    ].copy()
    center, scale = fit_robust_scaler(modified, set(outcomes["fish_id"]))
    _, order, frames = build_sequences(
        modified,
        {gap_id, no_baseline_id},
        center,
        scale,
    )
    assert gap_id in order
    assert frames[gap_id]["dpf"].astype(int).tolist() == [4]
    assert no_baseline_id not in order


def test_markov_forecast_propagates_to_target_day():
    probability = np.array([0.7, 0.3])
    transition = np.array([[0.8, 0.2], [0.1, 0.9]])
    expected = probability @ transition @ transition
    actual = propagate_ordered_probabilities(probability, transition, steps=2)
    np.testing.assert_allclose(actual, expected)


def test_macrostate_collapse_uses_ordered_score_gaps():
    severity_to_raw = np.array([0, 3, 2, 1])
    score_by_raw = np.array([-0.60, 2.90, 1.05, -0.48])
    mapping = macrostate_mapping(score_by_raw, severity_to_raw)
    np.testing.assert_array_equal(mapping, [0, 0, 1, 2])


def test_end_to_end_smoke(tmp_path):
    lfp, outcomes, dlc, _ = generate_dataset(seed=51, n_per_arm=20)
    metrics = run_analysis(
        lfp,
        outcomes,
        dlc,
        output_dir=tmp_path,
        seed=51,
        candidates=(3,),
        restarts=1,
        cv_folds=2,
        bootstrap_iterations=50,
        allow_placeholder_data=True,
    )
    assert metrics["split"]["fish_overlap"] == 0
    assert metrics["early_prediction"]["prediction_cutoff_dpf"] == 5
    assert metrics["early_prediction"]["target_dpf"] == 6
    assert "propagated to 6 dpf" in metrics["early_prediction"]["forecast_method"]
    assert metrics["features"] == list(FEATURES)
    assert 0 <= metrics["early_prediction"]["roc_auc"] <= 1
    assert (tmp_path / "tbi_model_metrics.json").exists()
    assert (tmp_path / "TBI_MODEL_RESULTS.md").exists()
    assert (tmp_path / "figures" / "tbi_early_prediction_roc.png").exists()
    predictions = pd.read_csv(
        tmp_path / "tables" / "tbi_early_predictions.csv"
    )
    assert np.array_equal(
        predictions["forecast_steps_to_dpf6"].to_numpy(int),
        6 - predictions["last_lfp_dpf_used"].to_numpy(int),
    )
    assert predictions["forecast_risk_dpf6"].between(0, 1).all()
    np.testing.assert_array_equal(
        predictions["forecast_risk_dpf6"].to_numpy(float),
        predictions["forecast_risk_dpf6"].round(6).to_numpy(float),
    )
    assert not np.allclose(
        predictions["forecast_risk_dpf6"],
        predictions["filtered_high_state_probability_last_observation"],
    )
    written = json.loads((tmp_path / "tbi_model_metrics.json").read_text())
    assert written["replacement_notice"].startswith("Current committed")
    assert TARGET not in written["features"]
    y = predictions[TARGET].to_numpy(int)
    score = predictions["forecast_risk_dpf6"].to_numpy(float)
    assert written["early_prediction"]["roc_auc"] == pytest.approx(
        roc_auc_score(y, score)
    )
    assert written["early_prediction"]["average_precision"] == pytest.approx(
        average_precision_score(y, score)
    )
    assert written["early_prediction"]["n_unique_forecast_scores"] == int(
        predictions["forecast_risk_dpf6"].nunique()
    )
    assert "NaN" not in (tmp_path / "tbi_model_metrics.json").read_text()


def test_committed_prediction_artifacts_reproduce_reported_metrics():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    data_dir = root / "data" / "template"
    metrics = json.loads((results_dir / "tbi_model_metrics.json").read_text())
    predictions = pd.read_csv(
        results_dir / "tables" / "tbi_early_predictions.csv"
    )
    y = predictions[TARGET].to_numpy(int)
    score = predictions["forecast_risk_dpf6"].to_numpy(float)

    assert metrics["early_prediction"]["roc_auc"] == pytest.approx(
        roc_auc_score(y, score)
    )
    assert metrics["early_prediction"]["average_precision"] == pytest.approx(
        average_precision_score(y, score)
    )
    assert metrics["early_prediction"]["brier_score"] == pytest.approx(
        brier_score_loss(y, score)
    )

    pooled_rho = spearmanr(
        predictions["cumulative_pressure_burden_kpa_hits"],
        score,
    ).statistic
    assert metrics["dynamics"][
        "dose_index_vs_dpf6_forecast_risk_spearman_rho"
    ] == pytest.approx(pooled_rho)

    behavior = pd.read_csv(data_dir / "tbi_4_6dpf_dlc_behavior.csv")
    behavior = behavior.loc[
        (behavior["dpf"] == 6) & behavior["dlc_tracking_qc_pass"]
    ]
    merged = predictions.merge(
        behavior[["fish_id", "dlc_behavior_abnormality_index"]],
        on="fish_id",
        how="inner",
        validate="one_to_one",
    )
    behavior_rho = spearmanr(
        merged["forecast_risk_dpf6"],
        merged["dlc_behavior_abnormality_index"],
    ).statistic
    assert metrics["behavior_validation"][
        "dpf6_forecast_risk_vs_dpf6_dlc_abnormality_spearman_rho"
    ] == pytest.approx(behavior_rho)


def test_module_help_does_not_reset_template(tmp_path):
    env = os.environ.copy()
    source_dir = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(source_dir) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "tbi_markov", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--force-reset-demo" in completed.stdout
    assert not (tmp_path / "data").exists()
