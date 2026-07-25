"""Fit and evaluate the synthetic TBI hidden Markov benchmark.

The primary temporal test uses forward-only filtering: LFP observations through
5 dpf predict a planted endpoint at 6 dpf. No held-out target-fish 6 dpf LFP,
behavior, injury dose, group, or truth field enters that early-risk
calculation. Training-fish 4–6 dpf sessions estimate the HMM dynamics. This is
not causal-effect inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from .common import (
    DATA_DIR,
    DLC_CSV,
    DOSE_INDEX,
    FEATURES,
    GROUP_ORDER,
    GROUPS,
    LFP_CSV,
    OBSERVATION_DPF,
    OUTCOMES_CSV,
    PLACEHOLDER_STATUS,
    RECORD_STATUS,
    RESULTS_DIR,
    PREDICTION_CUTOFF_DPF,
    SEED,
    TARGET,
    TARGET_DPF,
    TRUTH_STATE,
    assert_analysis_ready,
    build_sequences,
    fit_robust_scaler,
    load_dataset,
    qc_sessions,
    validate_dataset,
)
from .hmm import DiagonalGaussianHMM

RISING_FEATURES = (
    "lfp_mean_uv",
    "lfp_variance_uv2",
    "lfp_skewness",
    "lfp_kurtosis",
    "lfp_fourth_power_mean_uv4",
    "lfp_seizure_event_rate_per_h",
)
FALLING_FEATURES = ("lfp_ica_complexity",)
FORECAST_PROBABILITY_DECIMALS = 6
GROUP_LABELS = {
    "sham": "Sham",
    "tbi_low": "3 hits",
    "tbi_moderate": "5 hits",
    "tbi_high": "7 hits",
}
GROUP_COLORS = {
    "sham": "#64748B",
    "tbi_low": "#0EA5E9",
    "tbi_moderate": "#F59E0B",
    "tbi_high": "#DC2626",
}


def fish_level_split(
    outcomes: pd.DataFrame,
    test_fraction: float = 0.30,
    seed: int = SEED,
) -> tuple[set[str], set[str], pd.DataFrame]:
    """Create a deterministic arm/outcome-balanced fish-level partition."""
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5.")
    frame = outcomes.copy()
    target_label = frame[TARGET].map(
        lambda value: "missing" if pd.isna(value) else str(int(value))
    )
    frame["_stratum"] = frame["group"].astype(str) + "__" + target_label
    rng = np.random.default_rng(seed)
    test_ids: set[str] = set()

    for _, stratum in frame.groupby("_stratum", sort=True):
        ids = stratum["fish_id"].astype(str).to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n_test = int(round(test_fraction * len(ids)))
        if len(ids) >= 3:
            n_test = max(1, min(len(ids) - 1, n_test))
        else:
            n_test = 0
        test_ids.update(ids[:n_test])

    # Keep the arm sizes close even when a rare "missing" stratum has one fish.
    for group, group_frame in frame.groupby("group", sort=True):
        desired = int(round(test_fraction * len(group_frame)))
        current = sum(fish_id in test_ids for fish_id in group_frame["fish_id"])
        if current < desired:
            candidates = [
                str(fish_id)
                for fish_id in group_frame["fish_id"]
                if str(fish_id) not in test_ids
            ]
            rng.shuffle(candidates)
            test_ids.update(candidates[: desired - current])
        elif current > desired:
            candidates = [
                str(fish_id)
                for fish_id in group_frame["fish_id"]
                if str(fish_id) in test_ids
            ]
            rng.shuffle(candidates)
            for fish_id in candidates[: current - desired]:
                test_ids.remove(fish_id)

    all_ids = set(frame["fish_id"].astype(str))
    train_ids = all_ids - test_ids
    if train_ids & test_ids or train_ids | test_ids != all_ids:
        raise RuntimeError("Invalid fish-level split.")
    if not train_ids or not test_ids:
        raise RuntimeError("Both train and test partitions must be non-empty.")

    assignments = frame[["fish_id", "group", TARGET]].copy()
    assignments["split"] = assignments["fish_id"].map(
        lambda fish_id: "test" if str(fish_id) in test_ids else "train"
    )
    assignments = assignments.sort_values("fish_id").reset_index(drop=True)
    return train_ids, test_ids, assignments


def severity_mapping(means: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order arbitrary HMM labels with a prespecified LFP severity direction."""
    rising = [FEATURES.index(feature) for feature in RISING_FEATURES]
    falling = [FEATURES.index(feature) for feature in FALLING_FEATURES]
    score = means[:, rising].mean(axis=1)
    if falling:
        score = score - means[:, falling].mean(axis=1)
    severity_to_raw = np.argsort(score)
    raw_to_severity = np.empty_like(severity_to_raw)
    raw_to_severity[severity_to_raw] = np.arange(len(severity_to_raw))
    return raw_to_severity, severity_to_raw, score


def macrostate_mapping(
    severity_score: np.ndarray,
    severity_to_raw: np.ndarray,
    n_macrostates: int = 3,
) -> np.ndarray:
    """Collapse adjacent statistical components into interpretable severity states.

    Model-order selection is free to choose more Gaussian components than the
    template's three planted severity categories.  The collapse is based only
    on the largest gaps in the prespecified LFP severity score; truth labels are
    not consulted.
    """
    ordered_score = np.asarray(severity_score)[severity_to_raw]
    n_microstates = len(ordered_score)
    if n_microstates == n_macrostates:
        return np.arange(n_macrostates, dtype=int)
    if n_microstates < n_macrostates:
        return np.rint(
            np.linspace(0, n_macrostates - 1, n_microstates)
        ).astype(int)
    gaps = np.diff(ordered_score)
    cut_after = set(
        np.argsort(gaps)[-(n_macrostates - 1) :].astype(int).tolist()
    )
    mapping = np.zeros(n_microstates, dtype=int)
    macrostate = 0
    for microstate in range(n_microstates):
        mapping[microstate] = macrostate
        if microstate in cut_after:
            macrostate += 1
    return mapping


def _make_hmm(
    n_states: int,
    seed: int,
    restarts: int,
    n_iter: int = 160,
) -> DiagonalGaussianHMM:
    return DiagonalGaussianHMM(
        n_components=n_states,
        random_state=seed,
        n_restarts=restarts,
        n_iter=n_iter,
        tol=1e-5,
        min_covar=1e-3,
        variance_regularization=0.05,
        start_pseudocount=0.10,
        transition_pseudocount=0.25,
    )


def select_state_count(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    train_ids: set[str],
    candidates: Iterable[int],
    seed: int,
    restarts: int,
    cv_folds: int,
) -> tuple[int, dict[int, dict], dict[int, DiagonalGaussianHMM], np.ndarray, np.ndarray]:
    """Compare K on train-only BIC and train-only fish-level CV likelihood."""
    candidates = tuple(sorted(set(int(value) for value in candidates)))
    train_outcomes = outcomes.loc[outcomes["fish_id"].isin(train_ids)].reset_index(drop=True)
    train_fish = train_outcomes["fish_id"].astype(str).to_numpy()
    strata = train_outcomes["group"].astype(str).to_numpy()
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least two.")
    splitter = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)

    center, scale = fit_robust_scaler(lfp, train_ids)
    full_sequences, _, _ = build_sequences(lfp, train_ids, center, scale)
    n_observations = sum(len(sequence) for sequence in full_sequences)
    n_features = len(FEATURES)
    results: dict[int, dict] = {}
    full_models: dict[int, DiagonalGaussianHMM] = {}

    for n_states in candidates:
        full_model = _make_hmm(n_states, seed + n_states * 101, restarts).fit(
            full_sequences
        )
        full_models[n_states] = full_model
        train_loglik = full_model.score(full_sequences)
        n_parameters = (
            (n_states - 1)
            + n_states * (n_states - 1)
            + 2 * n_states * n_features
        )
        bic = -2.0 * train_loglik + n_parameters * math.log(n_observations)
        fold_loglik: list[float] = []

        for fold, (fit_index, validation_index) in enumerate(
            splitter.split(train_fish, strata)
        ):
            fit_ids = set(train_fish[fit_index])
            validation_ids = set(train_fish[validation_index])
            fold_center, fold_scale = fit_robust_scaler(lfp, fit_ids)
            fit_sequences, _, _ = build_sequences(
                lfp, fit_ids, fold_center, fold_scale
            )
            validation_sequences, _, _ = build_sequences(
                lfp, validation_ids, fold_center, fold_scale
            )
            model = _make_hmm(
                n_states,
                seed + n_states * 1_000 + fold,
                max(1, restarts - 1),
                n_iter=130,
            ).fit(fit_sequences)
            n_validation = sum(len(sequence) for sequence in validation_sequences)
            fold_loglik.append(model.score(validation_sequences) / n_validation)

        cv_sd = float(np.std(fold_loglik, ddof=1)) if len(fold_loglik) > 1 else 0.0
        results[n_states] = {
            "bic": float(bic),
            "train_log_likelihood": float(train_loglik),
            "cv_log_likelihood_per_session": float(np.mean(fold_loglik)),
            "cv_standard_deviation": cv_sd,
            "cv_standard_error": float(cv_sd / math.sqrt(len(fold_loglik))),
            "fold_log_likelihoods": [float(value) for value in fold_loglik],
            "n_parameters": int(n_parameters),
            "converged": bool(full_model.converged_),
            "iterations": int(full_model.n_iter_),
        }

    selected = min(results, key=lambda value: results[value]["bic"])
    return selected, results, full_models, center, scale


def _ordered_filter(
    model: DiagonalGaussianHMM,
    sequence: np.ndarray,
    severity_to_raw: np.ndarray,
) -> np.ndarray:
    return model.filter(sequence)[:, severity_to_raw]


def score_test_sessions(
    model: DiagonalGaussianHMM,
    lfp: pd.DataFrame,
    test_ids: set[str],
    center: np.ndarray,
    scale: np.ndarray,
    raw_to_severity: np.ndarray,
    severity_to_raw: np.ndarray,
    micro_to_macro: np.ndarray,
) -> pd.DataFrame:
    sequences, order, frames = build_sequences(lfp, test_ids, center, scale)
    rows: list[dict] = []
    for sequence, fish_id in zip(sequences, order):
        frame = frames[fish_id]
        microstates = raw_to_severity[model.predict(sequence)]
        micro_probabilities = _ordered_filter(model, sequence, severity_to_raw)
        macro_probabilities = np.column_stack(
            [
                micro_probabilities[:, micro_to_macro == macrostate].sum(axis=1)
                for macrostate in range(3)
            ]
        )
        states = micro_to_macro[microstates]
        for row_index, (_, source) in enumerate(frame.iterrows()):
            record = {
                "fish_id": fish_id,
                "group": source["group"],
                "dpf": int(source["dpf"]),
                "measured_peak_pressure_kpa": float(
                    source["measured_peak_pressure_kpa"]
                ),
                DOSE_INDEX: float(source[DOSE_INDEX]),
                "hidden_state_TRUTH": int(source[TRUTH_STATE]),
                "predicted_microstate": int(microstates[row_index]),
                "predicted_state": int(states[row_index]),
                "expected_state": float(
                    macro_probabilities[row_index] @ np.arange(3)
                ),
            }
            for state in range(model.n_components):
                record[f"p_microstate_{state}"] = float(
                    micro_probabilities[row_index, state]
                )
            for state in range(3):
                record[f"p_state_{state}"] = float(
                    macro_probabilities[row_index, state]
                )
            rows.append(record)
    return pd.DataFrame(rows).sort_values(["fish_id", "dpf"]).reset_index(drop=True)


def state_recovery_metrics(scored_sessions: pd.DataFrame) -> dict:
    truth = scored_sessions["hidden_state_TRUTH"].to_numpy(int)
    predicted = scored_sessions["predicted_state"].to_numpy(int)
    labels = sorted(set(truth) | set(predicted))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    recall = {}
    for index, label in enumerate(labels):
        denominator = matrix[index].sum()
        recall[str(label)] = float(matrix[index, index] / denominator) if denominator else None
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "adjusted_rand_index": float(adjusted_rand_score(truth, predicted)),
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
        "per_state_recall": recall,
        "n_test_sessions": int(len(scored_sessions)),
    }


def propagate_ordered_probabilities(
    probability: np.ndarray,
    ordered_transition_matrix: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Propagate a filtered ordered-state distribution to a future DPF."""
    if steps < 0:
        raise ValueError("Forecast steps cannot be negative.")
    probability = np.asarray(probability, dtype=float)
    transition = np.asarray(ordered_transition_matrix, dtype=float)
    return probability @ np.linalg.matrix_power(transition, steps)


def early_prediction(
    model: DiagonalGaussianHMM,
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    test_ids: set[str],
    center: np.ndarray,
    scale: np.ndarray,
    severity_to_raw: np.ndarray,
    micro_to_macro: np.ndarray,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[dict, pd.DataFrame]:
    eligible = outcomes.loc[
        outcomes["fish_id"].isin(test_ids)
        & outcomes[TARGET].notna()
    ].copy()
    sequences, order, frames = build_sequences(
        lfp,
        set(eligible["fish_id"].astype(str)),
        center,
        scale,
        cutoff_dpf=PREDICTION_CUTOFF_DPF,
    )
    outcome_lookup = eligible.set_index("fish_id")
    ordered_transition = model.transmat_[
        np.ix_(severity_to_raw, severity_to_raw)
    ]
    rows: list[dict] = []
    for sequence, fish_id in zip(sequences, order):
        probabilities = _ordered_filter(model, sequence, severity_to_raw)
        source = frames[fish_id].iloc[-1]
        last_dpf = int(source["dpf"])
        forecast_steps = TARGET_DPF - last_dpf
        forecast_probability = propagate_ordered_probabilities(
            probabilities[-1],
            ordered_transition,
            forecast_steps,
        )
        current_high_probability = float(
            probabilities[-1, micro_to_macro == 2].sum()
        )
        forecast_high_probability = float(
            forecast_probability[micro_to_macro == 2].sum()
        )
        rows.append(
            {
                "fish_id": fish_id,
                "group": source["group"],
                "last_lfp_dpf_used": last_dpf,
                "forecast_steps_to_dpf6": forecast_steps,
                "filtered_high_state_probability_last_observation": (
                    current_high_probability
                ),
                "forecast_risk_dpf6": forecast_high_probability,
                TARGET: int(outcome_lookup.loc[fish_id, TARGET]),
                "batch": int(outcome_lookup.loc[fish_id, "batch"]),
                DOSE_INDEX: float(outcome_lookup.loc[fish_id, DOSE_INDEX]),
            }
        )
    predictions = pd.DataFrame(rows).sort_values("fish_id").reset_index(drop=True)
    if (predictions["last_lfp_dpf_used"] > PREDICTION_CUTOFF_DPF).any():
        raise RuntimeError("Early prediction used a post-cutoff LFP observation.")
    if (predictions["forecast_steps_to_dpf6"] <= 0).any():
        raise RuntimeError("Early prediction did not preserve a future forecast horizon.")

    # Synthetic emissions yield almost one-hot posteriors. Microscopic
    # floating-point tails should not create an artificial ordering among
    # operationally identical risks, and all reported metrics must reproduce
    # exactly from the serialized CSV. Six decimals is therefore the
    # prespecified evaluation precision.
    raw_score = predictions["forecast_risk_dpf6"].to_numpy(float, copy=True)
    predictions["filtered_high_state_probability_last_observation"] = (
        predictions["filtered_high_state_probability_last_observation"].round(
            FORECAST_PROBABILITY_DECIMALS
        )
    )
    predictions["forecast_risk_dpf6"] = predictions["forecast_risk_dpf6"].round(
        FORECAST_PROBABILITY_DECIMALS
    )
    y = predictions[TARGET].to_numpy(int)
    score = predictions["forecast_risk_dpf6"].to_numpy(float)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Held-out early-prediction set needs both endpoint classes.")

    threshold = 0.5
    label = score >= threshold
    true_positive = int(np.sum((label == 1) & (y == 1)))
    true_negative = int(np.sum((label == 0) & (y == 0)))
    false_positive = int(np.sum((label == 1) & (y == 0)))
    false_negative = int(np.sum((label == 0) & (y == 1)))
    sensitivity = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)

    rng = np.random.default_rng(seed)
    auc_boot: list[float] = []
    ap_boot: list[float] = []
    for _ in range(bootstrap_iterations):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        auc_boot.append(float(roc_auc_score(y[index], score[index])))
        ap_boot.append(float(average_precision_score(y[index], score[index])))

    prevalence = float(y.mean())
    brier_baseline = float(prevalence * (1.0 - prevalence))
    brier = float(brier_score_loss(y, score))
    metrics = {
        "prediction_cutoff_dpf": PREDICTION_CUTOFF_DPF,
        "target_dpf": TARGET_DPF,
        "forecast_method": (
            "Causal filtered probability at the last contiguous QC-passing "
            "4-5 dpf observation, propagated to 6 dpf with the learned "
            "ordered microstate transition matrix."
        ),
        "n_test_fish": int(len(predictions)),
        "n_positive": int(y.sum()),
        "endpoint_prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "average_precision_baseline": prevalence,
        "brier_score": brier,
        "brier_constant_prevalence_baseline": brier_baseline,
        "brier_skill_score_vs_constant_prevalence": float(
            1.0 - brier / brier_baseline
        ),
        "threshold": threshold,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "roc_auc_95ci": [
            float(np.percentile(auc_boot, 2.5)),
            float(np.percentile(auc_boot, 97.5)),
        ],
        "average_precision_95ci": [
            float(np.percentile(ap_boot, 2.5)),
            float(np.percentile(ap_boot, 97.5)),
        ],
        "bootstrap_requested_replicates": int(bootstrap_iterations),
        "bootstrap_valid_replicates": int(len(auc_boot)),
        "forecast_score_precision_decimals": FORECAST_PROBABILITY_DECIMALS,
        "n_unique_forecast_scores": int(predictions["forecast_risk_dpf6"].nunique()),
        "numerical_precision_sensitivity": {
            "unrounded_roc_auc": float(roc_auc_score(y, raw_score)),
            "unrounded_average_precision": float(
                average_precision_score(y, raw_score)
            ),
            "max_absolute_rounding_delta": float(
                np.max(np.abs(raw_score - score))
            ),
            "note": (
                "Primary metrics use forecast probabilities rounded to six "
                "decimals so ties are operationally meaningful and metrics "
                "reproduce from the committed CSV."
            ),
        },
    }
    return metrics, predictions


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def dynamics_metrics(
    scored_sessions: pd.DataFrame,
    early_predictions: pd.DataFrame,
    n_states: int = 3,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    occupancy_rows: list[dict] = []
    for (group, dpf), frame in scored_sessions.groupby(["group", "dpf"], sort=True):
        row = {
            "group": group,
            "dpf": int(dpf),
            "n_sessions": int(len(frame)),
            "mean_expected_state": float(frame["expected_state"].mean()),
            "sem_expected_state": float(
                frame["expected_state"].std(ddof=1) / math.sqrt(len(frame))
            )
            if len(frame) > 1
            else 0.0,
        }
        for state in range(n_states):
            row[f"fraction_state_{state}"] = float(
                (frame["predicted_state"] == state).mean()
            )
        occupancy_rows.append(row)
    occupancy = pd.DataFrame(occupancy_rows).sort_values(
        ["group", "dpf"], key=lambda series: series.map(GROUP_ORDER) if series.name == "group" else series
    )

    transition_rows: list[dict] = []
    for group in GROUPS:
        worsening = recovery = stable = total = 0
        for _, fish in scored_sessions.loc[
            scored_sessions["group"] == group
        ].groupby("fish_id"):
            states = fish.sort_values("dpf")["predicted_state"].to_numpy(int)
            delta = np.diff(states)
            worsening += int(np.sum(delta > 0))
            recovery += int(np.sum(delta < 0))
            stable += int(np.sum(delta == 0))
            total += len(delta)
        transition_rows.append(
            {
                "group": group,
                "n_transitions": total,
                "worsening_fraction": worsening / total if total else np.nan,
                "recovery_fraction": recovery / total if total else np.nan,
                "stable_fraction": stable / total if total else np.nan,
            }
        )
    transitions = pd.DataFrame(transition_rows)
    pressure_rho, pressure_p = spearmanr(
        early_predictions[DOSE_INDEX],
        early_predictions["forecast_risk_dpf6"],
    )
    injured = early_predictions.loc[early_predictions["group"] != "sham"]
    injured_rho, injured_p = spearmanr(
        injured[DOSE_INDEX],
        injured["forecast_risk_dpf6"],
    )
    within_group = {}
    for group, frame in injured.groupby("group", sort=True):
        rho, p_value = spearmanr(
            frame[DOSE_INDEX],
            frame["forecast_risk_dpf6"],
        )
        within_group[group] = {
            "n_fish": int(len(frame)),
            "rho": _finite_or_none(rho),
            "p": _finite_or_none(p_value),
        }
    metrics = {
        "dose_index_vs_dpf6_forecast_risk_spearman_rho": _finite_or_none(
            pressure_rho
        ),
        "dose_index_vs_dpf6_forecast_risk_p": _finite_or_none(pressure_p),
        "injured_only_dose_risk_spearman_rho": _finite_or_none(injured_rho),
        "injured_only_dose_risk_spearman_p": _finite_or_none(injured_p),
        "within_injury_arm_dose_risk": within_group,
        "dose_association_note": (
            "The pooled value is a synthetic arm-gradient check, not an "
            "individual or within-arm dose-response estimate."
        ),
        "group_transition_summary": transitions.to_dict("records"),
    }
    return metrics, occupancy, transitions


def behavior_validation(
    early_predictions: pd.DataFrame,
    dlc: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    behavior = dlc.loc[
        (dlc["dpf"] == TARGET_DPF)
        & dlc["dlc_tracking_qc_pass"].astype(bool)
    ].copy()
    merged = early_predictions.merge(
        behavior[
            [
                "fish_id",
                "dlc_behavior_abnormality_index",
                "dlc_mean_speed_mm_s",
                "dlc_rest_fraction",
                "dlc_whirlpool_rate_per_min",
                "manual_pts_stage_TRUTH",
            ]
        ],
        on="fish_id",
        how="inner",
        validate="one_to_one",
    )
    rho, p_value = spearmanr(
        merged["forecast_risk_dpf6"],
        merged["dlc_behavior_abnormality_index"],
    )

    # Partial rank correlation after removing the planted dose and batch terms.
    x = rankdata(merged["forecast_risk_dpf6"])
    y = rankdata(merged["dlc_behavior_abnormality_index"])
    group_dummies = pd.get_dummies(merged["group"], drop_first=True).to_numpy(float)
    batch_dummies = pd.get_dummies(merged["batch"], drop_first=True).to_numpy(float)
    covariates = np.column_stack(
        [
            np.ones(len(merged)),
            np.log1p(merged[DOSE_INDEX].to_numpy(float)),
            group_dummies,
            batch_dummies,
        ]
    )
    x_residual = x - covariates @ np.linalg.lstsq(covariates, x, rcond=None)[0]
    y_residual = y - covariates @ np.linalg.lstsq(covariates, y, rcond=None)[0]
    partial_rho, partial_p = spearmanr(x_residual, y_residual)
    metrics = {
        "n_fish": int(len(merged)),
        "dpf6_forecast_risk_vs_dpf6_dlc_abnormality_spearman_rho": (
            _finite_or_none(rho)
        ),
        "dpf6_forecast_risk_vs_dpf6_dlc_abnormality_p": _finite_or_none(
            p_value
        ),
        "dose_batch_adjusted_partial_spearman_rho": _finite_or_none(partial_rho),
        "dose_batch_adjusted_partial_spearman_p": _finite_or_none(partial_p),
        "note": (
            "Generated pose-style values share the planted latent-state "
            "generator with LFP features. They are withheld from HMM inputs "
            "but are not an independent validation set; severe-arm locomotor "
            "speed is intentionally non-monotonic."
        ),
    }
    return metrics, merged


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _normalized_frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _software_versions() -> dict[str, str]:
    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "matplotlib": "matplotlib",
    }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        **{
            label: importlib.metadata.version(distribution)
            for label, distribution in distributions.items()
        },
    }


def _style_axis(axis: plt.Axes, *, grid_axis: str | None = "y") -> None:
    axis.set_facecolor("#FFFFFF")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#CBD5E1")
    axis.spines["bottom"].set_color("#CBD5E1")
    axis.tick_params(colors="#334155")
    if grid_axis:
        axis.grid(
            axis=grid_axis,
            color="#E2E8F0",
            linewidth=0.8,
            alpha=0.85,
        )
        axis.set_axisbelow(True)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def make_figures(
    output_dir: Path,
    selection: dict[int, dict],
    recovery: dict,
    early: dict,
    early_predictions: pd.DataFrame,
    occupancy: pd.DataFrame,
    behavior: pd.DataFrame,
    transition_matrix: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    states = sorted(selection)

    fig, axis_left = plt.subplots(figsize=(7.2, 4.6))
    _style_axis(axis_left)
    axis_left.plot(
        states,
        [selection[k]["bic"] for k in states],
        "o-",
        color="#0F766E",
        linewidth=2.2,
        markersize=7,
    )
    axis_left.set_xlabel("Hidden states (K)")
    axis_left.set_ylabel("Train-only BIC", color="#0F766E")
    axis_left.set_xticks(states)
    axis_right = axis_left.twinx()
    axis_right.errorbar(
        states,
        [selection[k]["cv_log_likelihood_per_session"] for k in states],
        yerr=[selection[k]["cv_standard_error"] for k in states],
        fmt="s-",
        color="#C2410C",
        capsize=4,
        linewidth=2.0,
        markersize=6,
    )
    axis_right.set_ylabel("Train-only CV log likelihood/session", color="#C2410C")
    axis_right.spines["top"].set_visible(False)
    axis_right.spines["left"].set_visible(False)
    axis_right.spines["right"].set_color("#CBD5E1")
    axis_right.tick_params(axis="y", colors="#C2410C")
    axis_left.set_title(
        "HMM model-order comparison",
        loc="left",
        fontweight="bold",
        color="#0F172A",
        pad=30,
    )
    axis_left.text(
        0,
        1.01,
        "Training fish only • K=4 is the tested upper boundary",
        transform=axis_left.transAxes,
        color="#64748B",
        fontsize=9,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir / "tbi_model_selection.png")

    matrix = np.asarray(recovery["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(5.2, 4.6))
    _style_axis(axis, grid_axis=None)
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="#FFFFFF" if matrix[row, column] > matrix.max() / 2 else "#0F172A",
                fontweight="bold",
            )
    axis.set_xlabel("Predicted validation macrostate")
    axis.set_ylabel("Planted validation macrostate")
    axis.set_xticks(range(len(recovery["labels"])), recovery["labels"])
    axis.set_yticks(range(len(recovery["labels"])), recovery["labels"])
    axis.set_title(
        "Held-out planted-state self-check",
        loc="left",
        fontweight="bold",
        color="#0F172A",
    )
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    _save_figure(fig, output_dir / "tbi_state_confusion.png")

    y = early_predictions[TARGET].to_numpy(int)
    risk = early_predictions["forecast_risk_dpf6"].to_numpy(float)
    fpr, tpr, _ = roc_curve(y, risk)
    fig, axis = plt.subplots(figsize=(5.8, 4.8))
    _style_axis(axis, grid_axis="both")
    axis.plot(fpr, tpr, lw=2.2, color="#0F766E", label=f"AUC = {early['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="#64748B")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title(
        "Forward-only 4–5 dpf LFP forecast",
        loc="left",
        fontweight="bold",
        color="#0F172A",
        pad=30,
    )
    axis.text(
        0,
        1.01,
        "Synthetic 6 dpf endpoint • score precision: 6 decimals",
        transform=axis.transAxes,
        color="#64748B",
        fontsize=9,
    )
    axis.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, output_dir / "tbi_early_prediction_roc.png")

    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    _style_axis(axis)
    for group in GROUPS:
        frame = occupancy.loc[occupancy["group"] == group].sort_values("dpf")
        axis.errorbar(
            frame["dpf"],
            frame["mean_expected_state"],
            yerr=frame["sem_expected_state"],
            fmt="o-",
            capsize=3,
            linewidth=2.0,
            markersize=6,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )
    axis.set_xticks(OBSERVATION_DPF)
    axis.set_xlabel("dpf")
    axis.set_ylabel("Mean filtered validation macrostate ± SEM")
    axis.set_ylim(-0.05, 2.05)
    axis.set_yticks([0, 1, 2])
    axis.set_title(
        "Synthetic held-out state trajectories",
        loc="left",
        fontweight="bold",
        color="#0F172A",
    )
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(fig, output_dir / "tbi_state_trajectories.png")

    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    _style_axis(axis, grid_axis=None)
    image = axis.imshow(transition_matrix, cmap="YlGnBu", vmin=0, vmax=1)
    for row in range(transition_matrix.shape[0]):
        for column in range(transition_matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{transition_matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color=(
                    "#FFFFFF"
                    if transition_matrix[row, column] > 0.48
                    else "#0F172A"
                ),
                fontweight="bold",
            )
    microstate_ticks = range(transition_matrix.shape[0])
    axis.set_xticks(microstate_ticks, microstate_ticks)
    axis.set_yticks(microstate_ticks, microstate_ticks)
    axis.set_xlabel("To ordered microstate")
    axis.set_ylabel("From ordered microstate")
    axis.set_title(
        "Ordered HMM microstate transitions",
        loc="left",
        fontweight="bold",
        color="#0F172A",
    )
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    _save_figure(fig, output_dir / "tbi_transition_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for axis in axes:
        _style_axis(axis)
    for group in GROUPS:
        frame = behavior.loc[behavior["group"] == group]
        axes[0].scatter(
            frame["forecast_risk_dpf6"],
            frame["dlc_behavior_abnormality_index"],
            s=28,
            alpha=0.72,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )
        axes[1].scatter(
            frame["forecast_risk_dpf6"],
            frame["dlc_mean_speed_mm_s"],
            s=28,
            alpha=0.72,
            color=GROUP_COLORS[group],
        )
    axes[0].set_xlabel("Markov-forecast DPF6 high-state probability")
    axes[0].set_ylabel("Generated behavior abnormality index at 6 dpf")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Markov-forecast DPF6 high-state probability")
    axes[1].set_ylabel("Generated mean speed at 6 dpf (mm/s)")
    axes[1].set_title("Speed is intentionally non-monotonic")
    fig.suptitle(
        "Generated pose-style concordance check",
        fontweight="bold",
        color="#0F172A",
    )
    fig.tight_layout()
    _save_figure(fig, output_dir / "tbi_dlc_validation.png")


def write_report(metrics: dict, output_path: Path) -> None:
    selection = metrics["model_selection"]
    selected = metrics["selected_states"]
    recovery = metrics["state_recovery"]
    early = metrics["early_prediction"]
    dynamics = metrics["dynamics"]
    behavior = metrics["behavior_validation"]
    dataset = metrics["dataset_qc"]
    split = metrics["split"]
    selected_row = selection[str(selected)]
    candidates = metrics["run_config"]["candidate_states"]

    def format_metric(value: float | None) -> str:
        return "not estimable" if value is None else f"{value:.3f}"

    text = f"""# Synthetic larval-zebrafish TBI Markov benchmark

> **Synthetic demonstration only.** No committed row represents an experimental
> animal. Every current observation, state, endpoint, and pose-style behavior
> value is generated. These results are not evidence of post-traumatic epilepsy,
> treatment efficacy, or feasibility of repeated invasive recordings.

## Run scope

- {dataset['n_fish']} generated fish across sham and 3/5/7-hit arms
- {dataset['n_lfp_sessions']} generated sessions; {dataset['n_qc_pass_sessions']}
  passed QC ({dataset['qc_pass_rate']:.1%})
- {dataset['n_contiguous_model_sessions']} contiguous model sessions from
  {dataset['n_model_fish_with_4dpf_baseline']} fish with a usable 4 dpf baseline
- {split['n_train_fish']} train / {split['n_test_fish']} test fish, with
  {split['fish_overlap']} overlapping fish
- resistance-change and noise QC failures remain auditable but are excluded;
  a later gap terminates the usable prefix
- positive heavy-tailed features receive `log1p`; robust preprocessing is
  fitted on training fish only
- selected **K={selected}** by lowest train-only BIC ({selected_row['bic']:.1f});
  train-only CV log likelihood/session {selected_row['cv_log_likelihood_per_session']:.3f}

K={selected} is the upper boundary of the tested candidate set
{candidates}; it does not establish {selected} biological states. Adjacent
severity-ordered microstates are collapsed to three planted validation
macrostates without consulting truth labels.

## Held-out planted-state self-check

- balanced accuracy: **{recovery['balanced_accuracy']:.3f}**
- macro F1: **{recovery['macro_f1']:.3f}**
- adjusted Rand index: **{recovery['adjusted_rand_index']:.3f}**
- scored sessions: **{recovery['n_test_sessions']}**

Perfect recovery is an expected self-check for deliberately separated synthetic
emissions, not evidence that biological states have been identified.

## Forward-only early forecast

Each held-out fish's forecast used only its uninterrupted, QC-passing 4-5 dpf
LFP prefix. The final filtered state distribution was propagated through the
learned transition matrix to predict the separate planted 6 dpf high-burden
endpoint. No target-fish 6 dpf LFP or behavior entered its forecast;
training-fish 4-6 dpf sessions were used to estimate the HMM emissions and
transition dynamics.

- held-out fish: **{early['n_test_fish']}** ({early['n_positive']} positive)
- ROC-AUC: **{early['roc_auc']:.3f}** (bootstrap 95% CI
  {early['roc_auc_95ci'][0]:.3f}-{early['roc_auc_95ci'][1]:.3f})
- average precision: **{early['average_precision']:.3f}** versus prevalence
  baseline {early['average_precision_baseline']:.3f}
- Brier score: **{early['brier_score']:.3f}** versus constant-prevalence
  baseline {early['brier_constant_prevalence_baseline']:.3f}
- sensitivity/specificity at probability 0.5:
  **{early['sensitivity']:.3f}/{early['specificity']:.3f}**
- confusion counts: {early['confusion']['true_positive']} TP,
  {early['confusion']['true_negative']} TN,
  {early['confusion']['false_positive']} FP,
  {early['confusion']['false_negative']} FN
- operational score levels: **{early['n_unique_forecast_scores']}** at
  {early['forecast_score_precision_decimals']}-decimal precision

Primary rank metrics use the same rounded probabilities committed to CSV so
ties are meaningful and results reproduce after serialization. For numerical
sensitivity, the unrounded AUC was
{early['numerical_precision_sensitivity']['unrounded_roc_auc']:.3f}.
"Causal" in the implementation means forward-only temporal filtering, not
causal-effect inference; the split is retrospective and endpoint-stratified.

## Dose/dynamics and behavior checks

- pooled synthetic arm-gradient check: Spearman
  **rho={format_metric(dynamics['dose_index_vs_dpf6_forecast_risk_spearman_rho'])}**
- injured-fish-only dose/risk check: Spearman
  **rho={format_metric(dynamics['injured_only_dose_risk_spearman_rho'])}**
- forecast risk vs generated 6 dpf behavior abnormality: Spearman
  **rho={format_metric(behavior['dpf6_forecast_risk_vs_dpf6_dlc_abnormality_spearman_rho'])}**
- dose/batch-adjusted generated-behavior check: partial Spearman
  **rho={format_metric(behavior['dose_batch_adjusted_partial_spearman_rho'])}**

Generated behavior is withheld from HMM inputs but shares the planted latent
generator, so this is concordance rather than independent validation. The pooled
dose value is an arm-gradient check, not a within-arm dose-response estimate.

## Method boundaries

- [Locskai et al.](https://doi.org/10.1242/bio.060601) motivates the
  blast-pressure syringe insult and repeated-hit dose axis.
- [Eimon et al.](https://doi.org/10.1038/s41467-017-02404-4) motivates LFP
  acquisition, resistance-change/noise QC, overlapping-window higher moments,
  and ICA complexity.
- [Mathis et al.](https://doi.org/10.1038/s41593-018-0209-y) and
  [Nath et al.](https://doi.org/10.1038/s41596-019-0176-0) motivate the
  pose-summary interface.

Neither source paper reports this exact 4-6 dpf repeated same-fish LFP design.
Raw LFP feature extraction, pose tracking, injury-event clustering, and model
uncertainty from refitting are outside the current benchmark.
"""
    output_path.write_text(text, encoding="utf-8")


def run_analysis(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    dlc: pd.DataFrame,
    output_dir: Path | str = RESULTS_DIR,
    seed: int = SEED,
    test_fraction: float = 0.30,
    candidates: Iterable[int] = (2, 3, 4),
    restarts: int = 3,
    cv_folds: int = 3,
    bootstrap_iterations: int = 1_000,
    allow_placeholder_data: bool = False,
) -> dict:
    validate_dataset(lfp, outcomes, dlc)
    candidates = tuple(sorted(set(int(value) for value in candidates)))
    if not candidates or any(value < 1 for value in candidates):
        raise ValueError("candidates must contain at least one positive state count.")
    if restarts < 1:
        raise ValueError("restarts must be at least one.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least two.")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least one.")
    if not allow_placeholder_data:
        assert_analysis_ready(lfp, outcomes, dlc)
    status_counts = {
        "lfp": lfp[RECORD_STATUS].value_counts().sort_index().to_dict(),
        "outcomes": outcomes[RECORD_STATUS].value_counts().sort_index().to_dict(),
        "behavior": dlc[RECORD_STATUS].value_counts().sort_index().to_dict(),
    }
    has_placeholders = any(
        PLACEHOLDER_STATUS in table_counts
        for table_counts in status_counts.values()
    )
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_lfp = qc_sessions(lfp)
    train_ids, test_ids, assignments = fish_level_split(
        outcomes, test_fraction=test_fraction, seed=seed
    )
    selected, selection, models, center, scale = select_state_count(
        model_lfp,
        outcomes,
        train_ids,
        candidates,
        seed,
        restarts,
        cv_folds,
    )
    model = models[selected]
    train_model_sequences, train_model_fish_order, _ = build_sequences(
        model_lfp,
        train_ids,
        center,
        scale,
    )
    model_sequences, model_fish_order, _ = build_sequences(
        model_lfp,
        set(outcomes["fish_id"].astype(str)),
        center,
        scale,
    )
    raw_to_severity, severity_to_raw, severity_score = severity_mapping(model.means_)
    micro_to_macro = macrostate_mapping(severity_score, severity_to_raw)
    transition_matrix = model.transmat_[np.ix_(severity_to_raw, severity_to_raw)]
    scored_sessions = score_test_sessions(
        model,
        model_lfp,
        test_ids,
        center,
        scale,
        raw_to_severity,
        severity_to_raw,
        micro_to_macro,
    )
    recovery = state_recovery_metrics(scored_sessions)
    early, early_predictions = early_prediction(
        model,
        model_lfp,
        outcomes,
        test_ids,
        center,
        scale,
        severity_to_raw,
        micro_to_macro,
        seed + 10,
        bootstrap_iterations,
    )
    dynamics, occupancy, group_transitions = dynamics_metrics(
        scored_sessions, early_predictions
    )
    behavior, behavior_frame = behavior_validation(early_predictions, dlc)
    test_outcomes = outcomes.loc[outcomes["fish_id"].isin(test_ids)]
    test_with_endpoint = int(test_outcomes[TARGET].notna().sum())

    metrics = {
        "benchmark_type": (
            "synthetic_placeholder_demo"
            if has_placeholders
            else "labeled_dataset_benchmark"
        ),
        "seed": seed,
        "run_config": {
            "test_fraction": float(test_fraction),
            "candidate_states": list(candidates),
            "restarts": int(restarts),
            "cv_folds": int(cv_folds),
            "bootstrap_iterations": int(bootstrap_iterations),
            "allow_placeholder_data": bool(allow_placeholder_data),
        },
        "software_versions": _software_versions(),
        "input_data": {
            "normalized_table_sha256": {
                "lfp": _normalized_frame_sha256(lfp),
                "outcomes": _normalized_frame_sha256(outcomes),
                "behavior": _normalized_frame_sha256(dlc),
            },
            "record_status_counts": status_counts,
            "note": (
                "Hashes cover normalized in-memory CSV serialization. The "
                "committed manifest describes the initialized seed-42 template."
            ),
        },
        "split": {
            "n_train_fish": len(train_ids),
            "n_test_fish": len(test_ids),
            "fish_overlap": len(train_ids & test_ids),
            "n_train_model_fish_with_4dpf_baseline": len(train_model_fish_order),
            "n_train_model_sessions": int(
                sum(len(sequence) for sequence in train_model_sequences)
            ),
        },
        "dataset_qc": {
            "n_fish": int(len(outcomes)),
            "n_lfp_sessions": int(len(lfp)),
            "n_qc_pass_sessions": int(model_lfp.shape[0]),
            "qc_pass_rate": float(model_lfp.shape[0] / len(lfp)),
            "n_contiguous_model_sessions": int(
                sum(len(sequence) for sequence in model_sequences)
            ),
            "n_model_fish_with_4dpf_baseline": int(len(model_fish_order)),
            "sequence_rule": (
                "Use the uninterrupted QC-passing prefix beginning at 4 dpf; "
                "truncate at the first gap and exclude fish missing 4 dpf."
            ),
            "survival_by_group": outcomes.groupby("group")[
                "survived_to_6dpf"
            ].mean().to_dict(),
        },
        "selected_states": int(selected),
        "model_selection": {str(key): value for key, value in selection.items()},
        "severity_alignment": {
            "raw_to_severity": raw_to_severity.tolist(),
            "severity_to_raw": severity_to_raw.tolist(),
            "prespecified_score_by_raw_state": severity_score.tolist(),
            "microstate_to_macrostate": micro_to_macro.tolist(),
            "macrostate_collapse_rule": (
                "Adjacent severity-ordered components are split at the two "
                "largest prespecified-score gaps; truth labels are not used."
            ),
        },
        "state_recovery": recovery,
        "early_prediction": {
            **early,
            "cohort_flow": {
                "n_held_out_fish": int(len(test_ids)),
                "n_with_observed_dpf6_endpoint": test_with_endpoint,
                "n_excluded_missing_dpf6_endpoint": int(
                    len(test_ids) - test_with_endpoint
                ),
                "n_excluded_without_usable_4dpf_prefix": int(
                    test_with_endpoint - early["n_test_fish"]
                ),
                "n_forecast_eligible": int(early["n_test_fish"]),
                "n_pose_style_check_eligible": int(behavior["n_fish"]),
            },
            "causal_definition": (
                "Causal means forward-only temporal filtering: each state "
                "probability uses only its current and prior LFP observations. "
                "It does not mean causal-effect inference; the split is "
                "retrospective and endpoint-stratified."
            ),
        },
        "dynamics": dynamics,
        "behavior_validation": behavior,
        "selected_model_fit": {
            "converged": bool(model.converged_),
            "iterations": int(model.n_iter_),
            "log_likelihood": float(model.log_likelihood_),
            "log_likelihood_history": list(model.history_),
            "n_restarts": int(model.n_restarts),
            "max_iterations": int(model.n_iter),
            "tolerance": float(model.tol),
            "minimum_covariance": float(model.min_covar),
            "variance_regularization": float(model.variance_regularization),
            "start_pseudocount": float(model.start_pseudocount),
            "transition_pseudocount": float(model.transition_pseudocount),
        },
        "ordered_start_probabilities": model.startprob_[severity_to_raw].tolist(),
        "ordered_transition_matrix": transition_matrix.tolist(),
        "ordered_emission_means_robust_scaled": model.means_[severity_to_raw].tolist(),
        "ordered_emission_variances_robust_scaled": model.covars_[
            severity_to_raw
        ].tolist(),
        "features": list(FEATURES),
        "preprocessing": {
            "fit_partition": "training fish only",
            "transform": (
                "log1p variance, kurtosis, fourth-power mean, and event rate; "
                "then median/IQR scaling"
            ),
            "center": center.tolist(),
            "scale_iqr": scale.tolist(),
        },
        "replacement_notice": (
            "Current committed data are deterministically generated synthetic "
            "placeholders. Daily repeated 4-6 dpf LFP after 3 dpf TBI is a "
            "demonstration protocol, not an experimental result. Raw LFP and "
            "pose feature extraction are outside this repository's scope."
        ),
        "experimental_unit_notice": (
            "The current synthetic split and bootstrap operate at the fish "
            "level. A measured study with multiple larvae per injury event "
            "must add event/clutch identifiers and use grouped inference."
        ),
    }

    assignments.to_csv(tables_dir / "tbi_split_assignments.csv", index=False)
    scored_sessions.to_csv(tables_dir / "tbi_scored_test_sessions.csv", index=False)
    early_predictions.to_csv(tables_dir / "tbi_early_predictions.csv", index=False)
    occupancy.to_csv(tables_dir / "tbi_state_occupancy.csv", index=False)
    group_transitions.to_csv(
        tables_dir / "tbi_group_transition_summary.csv",
        index=False,
    )
    pd.DataFrame(
        transition_matrix,
        index=[f"from_state_{state}" for state in range(selected)],
        columns=[f"to_state_{state}" for state in range(selected)],
    ).to_csv(tables_dir / "tbi_transition_matrix.csv")
    (output_dir / "tbi_model_metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(metrics, output_dir / "TBI_MODEL_RESULTS.md")
    make_figures(
        figures_dir,
        selection,
        recovery,
        early,
        early_predictions,
        occupancy,
        behavior_frame,
        transition_matrix,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--states", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument(
        "--allow-placeholder-data",
        action="store_true",
        help="Run a demonstration despite placeholder_pending_replacement rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lfp, outcomes, dlc = load_dataset(
        args.data_dir / LFP_CSV.name,
        args.data_dir / OUTCOMES_CSV.name,
        args.data_dir / DLC_CSV.name,
    )
    metrics = run_analysis(
        lfp,
        outcomes,
        dlc,
        output_dir=args.output_dir,
        seed=args.seed,
        test_fraction=args.test_fraction,
        candidates=args.states,
        restarts=args.restarts,
        cv_folds=args.cv_folds,
        bootstrap_iterations=args.bootstrap_iterations,
        allow_placeholder_data=args.allow_placeholder_data,
    )
    print(
        f"Selected K={metrics['selected_states']}; held-out state balanced accuracy "
        f"{metrics['state_recovery']['balanced_accuracy']:.3f}; forward-only AUC "
        f"{metrics['early_prediction']['roc_auc']:.3f}"
    )
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
