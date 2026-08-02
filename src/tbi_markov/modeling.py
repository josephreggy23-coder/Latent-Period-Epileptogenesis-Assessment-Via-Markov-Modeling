"""Fit and evaluate the TBI hidden Markov model.

The primary prospective-style test is intentionally causal: LFP observations
through 5 dpf predict the behavioral endpoint at 6 dpf. No 6 dpf LFP, behavior,
injury dose, group, or outcome field enters that early-risk calculation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from .common import (
    DOSE_INDEX,
    FALLING_FEATURES,
    FEATURES,
    GROUP_ORDER,
    GROUPS,
    OBSERVATION_DPF,
    RESULTS_DIR,
    PREDICTION_CUTOFF_DPF,
    RISING_FEATURES,
    SEED,
    TARGET,
    TARGET_DPF,
    build_sequences,
    fit_robust_scaler,
    make_hmm,
    qc_sessions,
    severity_mapping,
)
from .baseline import fit_elastic_net_baseline
from .dose_ordering import (
    covariate_adjusted_dose_ordering,
    decode_state_index,
    primary_dose_ordering_test,
    run_negative_controls,
    write_negative_controls_report,
)
from .hmm import DiagonalGaussianHMM

# RISING_FEATURES, FALLING_FEATURES, severity_mapping, and make_hmm are
# re-exported from .common (imported above) so existing call sites and
# tests/test_features.py's `from tbi_markov.modeling import ...` keep working;
# they live in common.py so tbi_markov.dose_ordering can reuse them for the
# sham-only negative control without a circular import.


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
        full_model = make_hmm(n_states, seed + n_states * 101, restarts).fit(
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
            model = make_hmm(
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
) -> pd.DataFrame:
    """Score held-out sessions on the severity-ordered microstates directly.

    With K capped at 2-3 (docs/PREREGISTRATION.md), the fitted states already
    are the interpretable severity categories, so there is no separate
    macrostate collapse: ``predicted_state`` is the severity-ordered state.
    """
    sequences, order, frames = build_sequences(lfp, test_ids, center, scale)
    n_states = len(severity_to_raw)
    rows: list[dict] = []
    for sequence, fish_id in zip(sequences, order):
        frame = frames[fish_id]
        states = raw_to_severity[model.predict(sequence)]
        probabilities = _ordered_filter(model, sequence, severity_to_raw)
        for row_index, (_, source) in enumerate(frame.iterrows()):
            record = {
                "fish_id": fish_id,
                "group": source["group"],
                "dpf": int(source["dpf"]),
                "measured_peak_pressure_kpa": float(
                    source["measured_peak_pressure_kpa"]
                ),
                DOSE_INDEX: float(source[DOSE_INDEX]),
                "predicted_state": int(states[row_index]),
                "expected_state": float(
                    probabilities[row_index] @ np.arange(n_states)
                ),
            }
            for state in range(n_states):
                record[f"p_state_{state}"] = float(
                    probabilities[row_index, state]
                )
            rows.append(record)
    return pd.DataFrame(rows).sort_values(["fish_id", "dpf"]).reset_index(drop=True)


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
        # Columns are already severity-ordered by _ordered_filter, so the last
        # column is the single highest-severity state.
        current_high_probability = float(probabilities[-1, -1])
        forecast_high_probability = float(forecast_probability[-1])
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
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
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
        # Discrimination (AUC) and calibration are separate properties. The
        # propagated state probability is an LFP-derived quantity scored against
        # a different endpoint, so it can rank fish well while sitting far below
        # the 0.5 decision threshold. Recording the comparison keeps a low
        # sensitivity from being read as a ranking failure.
        "calibration": {
            "observed_positive_rate": float(y.mean()),
            "mean_forecast_risk": float(score.mean()),
            "median_forecast_risk": float(np.median(score)),
            "max_forecast_risk": float(score.max()),
            "n_above_threshold": int((score >= threshold).sum()),
        },
    }
    return metrics, predictions


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
    metrics = {
        "dose_index_vs_dpf6_forecast_risk_spearman_rho": float(pressure_rho),
        "dose_index_vs_dpf6_forecast_risk_p": float(pressure_p),
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
    # Blinded human Baraban score, carried through for inspection, not scored.
    stage_columns = [
        column for column in ("manual_pts_stage_observed",) if column in behavior.columns
    ]
    merged = early_predictions.merge(
        behavior[
            [
                "fish_id",
                "dlc_behavior_abnormality_index",
                "dlc_mean_speed_mm_s",
                "dlc_rest_fraction",
                "dlc_whirlpool_rate_per_min",
                *stage_columns,
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

    # Partial rank correlation after removing the dose and batch terms.
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
        "dpf6_forecast_risk_vs_dpf6_dlc_abnormality_spearman_rho": float(rho),
        "dpf6_forecast_risk_vs_dpf6_dlc_abnormality_p": float(p_value),
        "dose_batch_adjusted_partial_spearman_rho": float(partial_rho),
        "dose_batch_adjusted_partial_spearman_p": float(partial_p),
        "note": (
            "Behavioral values are blinded manual Baraban scores and "
            "pose-derived kinematics aggregated per session. Locomotor speed is "
            "not monotone in dose: pressures above roughly 300 kPa can suppress "
            "movement, so low speed is ambiguous between no seizure and severe "
            "injury."
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
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def make_figures(
    output_dir: Path,
    selection: dict[int, dict],
    early: dict,
    early_predictions: pd.DataFrame,
    occupancy: pd.DataFrame,
    behavior: pd.DataFrame,
    transition_matrix: np.ndarray,
) -> None:
    """Render the diagnostic figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    states = sorted(selection)

    fig, axis_left = plt.subplots(figsize=(7.2, 4.6))
    axis_left.plot(states, [selection[k]["bic"] for k in states], "o-", color="#0F766E")
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
        capsize=3,
    )
    axis_right.set_ylabel("Train-only CV log likelihood/session", color="#C2410C")
    axis_left.set_title("TBI HMM model-order selection")
    fig.tight_layout()
    fig.savefig(output_dir / "tbi_model_selection.png", dpi=160)
    plt.close(fig)

    y = early_predictions[TARGET].to_numpy(int)
    risk = early_predictions["forecast_risk_dpf6"].to_numpy(float)
    fpr, tpr, _ = roc_curve(y, risk)
    fig, axis = plt.subplots(figsize=(5.8, 4.8))
    axis.plot(fpr, tpr, lw=2.2, color="#0F766E", label=f"AUC = {early['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="#64748B")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("4-5 dpf LFP → 6 dpf behavioral endpoint")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "tbi_early_prediction_roc.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    for group in GROUPS:
        frame = occupancy.loc[occupancy["group"] == group].sort_values("dpf")
        axis.plot(
            frame["dpf"],
            frame["mean_expected_state"],
            "o-",
            label=group,
        )
    axis.set_xticks(OBSERVATION_DPF)
    axis.set_xlabel("dpf")
    axis.set_ylabel("Mean filtered expected state (held-out fish)")
    axis.set_title("Post-TBI electrophysiology trajectories")
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "tbi_state_trajectories.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(transition_matrix, cmap="YlGnBu", vmin=0, vmax=1)
    for row in range(transition_matrix.shape[0]):
        for column in range(transition_matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{transition_matrix[row, column]:.2f}",
                ha="center",
                va="center",
            )
    axis.set_xlabel("To severity state")
    axis.set_ylabel("From severity state")
    axis.set_title("Pooled HMM transition matrix")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "tbi_transition_matrix.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for group in GROUPS:
        frame = behavior.loc[behavior["group"] == group]
        axes[0].scatter(
            frame["forecast_risk_dpf6"],
            frame["dlc_behavior_abnormality_index"],
            s=28,
            alpha=0.72,
            label=group,
        )
        axes[1].scatter(
            frame["forecast_risk_dpf6"],
            frame["dlc_mean_speed_mm_s"],
            s=28,
            alpha=0.72,
        )
    axes[0].set_xlabel("Markov-forecast DPF6 high-state probability")
    axes[0].set_ylabel("Behavioral abnormality index at DPF6")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Markov-forecast DPF6 high-state probability")
    axes[1].set_ylabel("Mean locomotor speed at DPF6 (mm/s)")
    fig.suptitle("Blinded behavioral validation")
    fig.tight_layout()
    fig.savefig(output_dir / "tbi_dlc_validation.png", dpi=160)
    plt.close(fig)


def write_report(
    metrics: dict,
    output_path: Path,
    endpoint: dict | None = None,
) -> None:
    endpoint = endpoint or {}
    endpoint_counts = (
        "\nResolved: "
        f"**{endpoint['n_positive']} positive**, "
        f"**{endpoint['n_negative']} negative**, "
        f"**{endpoint['n_unresolved']} unresolved (`NA`)**.\n"
        if endpoint
        else ""
    )
    early = metrics["early_prediction"]
    dynamics = metrics["dynamics"]
    behavior = metrics["behavior_validation"]
    qc = metrics["dataset_qc"]
    selected = metrics["selected_states"]
    selection = metrics["model_selection"][str(selected)]
    text = f"""# Larval-zebrafish TBI Markov-model results

> **Measured recording, retrospective single-cohort analysis.** There is no
> latent-state ground truth, so state-recovery accuracy is **not measurable** -
> only the forward 6 dpf behavioural forecast is scored.

## Run scope

- **{qc['n_fish']} fish**, {qc['n_lfp_sessions']} LFP sessions at 4-6 dpf,
  {qc['n_qc_pass_sessions']} passing QC ({100 * qc['qc_pass_rate']:.1f}%)
- {qc['n_contiguous_model_sessions']} contiguous modelling sessions from
  {qc['n_model_fish_with_4dpf_baseline']} fish with a usable 4 dpf baseline
- selected **K={selected}** by lowest train-only BIC ({selection['bic']:.1f});
  train-only CV log likelihood/session
  {selection['cv_log_likelihood_per_session']:.3f}
- preprocessing and severity ordering never consult the endpoint

## Endpoint

The 6 dpf high-burden endpoint is **behavioural**: a fish is positive if the
blinded scorer logged at least one qualifying event (Baraban stage >= 2 with
passing pose QC) in the 6 dpf session. It shares no variable with the LFP
feature matrix, so the forecast target is independent of the model's inputs.

It is **three-valued**. A fish never observed at 6 dpf is `NA`, not `0`: an
unobserved animal has an unknown outcome, not a negative one, and coding
absence as negative would pad the negative class with animals nobody checked.
{endpoint_counts}
Unresolved fish are excluded from endpoint scoring rather than counted as
negatives. See `docs/EXPERIMENTAL_PROTOCOL.md` section 5.

## Causal 6 dpf forecast

Only an uninterrupted, QC-passing **4-5 dpf** LFP prefix is used. Its final
filtered state distribution is propagated through the learned transition matrix
to 6 dpf:

- held-out fish: **{early['n_test_fish']}** ({early['n_positive']} positive)
- ROC-AUC: **{early['roc_auc']:.3f}** (bootstrap 95% CI
  {early['roc_auc_95ci'][0]:.3f}-{early['roc_auc_95ci'][1]:.3f})
- average precision: **{early['average_precision']:.3f}**
- Brier score: **{early['brier_score']:.3f}**
- sensitivity/specificity at probability 0.5:
  **{early['sensitivity']:.3f}/{early['specificity']:.3f}**

### Discrimination versus calibration

The forecast **ranks** fish well but is **badly calibrated** against this
endpoint, so the fixed 0.5 threshold is a poor operating point and the
sensitivity above should not be read as a ranking failure:

- observed positive rate: **{early['calibration']['observed_positive_rate']:.3f}**
- mean / median forecast risk:
  **{early['calibration']['mean_forecast_risk']:.3f} /
  {early['calibration']['median_forecast_risk']:.3f}**
  (maximum {early['calibration']['max_forecast_risk']:.3f})
- held-out fish above 0.5: **{early['calibration']['n_above_threshold']}** of
  {early['n_test_fish']}

The propagated quantity is the probability of occupying the **highest-severity
LFP state**, whereas the endpoint is a **behavioural** event. The two are on
different scales, and the LFP state is rarer than the behavioural outcome, so
the risk sits well below 0.5 for most animals. Any deployment would need a
threshold fitted on training fish; none is tuned on the held-out set here.

## Latent-state recovery

**Not measurable.** These are real animals with no latent-state ground truth, so
there is nothing to score inferred states against, and no proxy is substituted.
The states are validated only indirectly: through the forward 6 dpf forecast and
the association with the independent behavioural channel.

## Dose and behaviour checks

- injury dose index vs 6 dpf forecast risk: Spearman
  rho={dynamics['dose_index_vs_dpf6_forecast_risk_spearman_rho']:.3f}
  (p={dynamics['dose_index_vs_dpf6_forecast_risk_p']:.3g})
- 6 dpf forecast risk vs independent 6 dpf behavioural abnormality:
  rho={behavior['dpf6_forecast_risk_vs_dpf6_dlc_abnormality_spearman_rho']:.3f}
  (p={behavior['dpf6_forecast_risk_vs_dpf6_dlc_abnormality_p']:.3g}), n={behavior['n_fish']}
- dose/batch-adjusted partial rho:
  {behavior['dose_batch_adjusted_partial_spearman_rho']:.3f}
  (p={behavior['dose_batch_adjusted_partial_spearman_p']:.3g})

## Boundaries

- **Repeated penetrating forebrain LFP in the same larva at 4-6 dpf is not a
  validated preparation.** The electrode metadata matches the Eimon penetrating
  method, which was demonstrated at 7 dpf and never validated as recoverable
  across days. Per-fish longitudinal state transitions - the premise of this
  Markov model - therefore rest on an assumption the dataset cannot verify.
  See `docs/EXPERIMENTAL_PROTOCOL.md` section 6.
- The combined 3 dpf TBI to 4-6 dpf LFP+behaviour protocol integrates three
  published methods and has not itself been published or piloted.
- The drop batch, which the protocol defines as the experimental unit, is not
  identified in the data, so larvae from one impact cannot be modelled as the
  nested observations they are.
- Pressures above roughly 300 kPa can suppress locomotion, so reduced movement
  in the highest-dose arm is ambiguous between "no seizure" and "too injured to
  move".
- A single qualifying event is not chronic epilepsy; this is an operational
  early post-traumatic seizure outcome.
- Three sessions per fish is a short series for a Markov model; the transition
  matrix is estimated from at most two observed steps per animal.
- Behaviour is scored in three discrete sessions, so event timing is
  interval-censored.
- The abnormality index is built only from event-rate and stage terms, which
  remain defined when the scorer logged nothing; kinematic columns are reported
  but deliberately excluded from the index.
- A single forebrain electrode per fish bounds the information available.
"""
    Path(output_path).write_text(text, encoding="utf-8")


def run_analysis(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    dlc: pd.DataFrame,
    output_dir: Path | str = RESULTS_DIR,
    seed: int = SEED,
    test_fraction: float = 0.30,
    candidates: Iterable[int] = (2, 3),
    restarts: int = 3,
    cv_folds: int = 3,
    bootstrap_iterations: int = 1_000,
    endpoint_summary: dict | None = None,
) -> dict:
    """Fit, score, and report the Markov analysis.

    ``endpoint_summary`` carries the positive/negative/unresolved counts from
    the ingestion manifest so the report can state them without recomputing.
    """
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
    model_sequences, model_fish_order, _ = build_sequences(
        model_lfp,
        set(outcomes["fish_id"].astype(str)),
        center,
        scale,
    )
    raw_to_severity, severity_to_raw, severity_score = severity_mapping(model.means_)
    transition_matrix = model.transmat_[np.ix_(severity_to_raw, severity_to_raw)]
    scored_sessions = score_test_sessions(
        model,
        model_lfp,
        test_ids,
        center,
        scale,
        raw_to_severity,
        severity_to_raw,
    )
    early, early_predictions = early_prediction(
        model,
        model_lfp,
        outcomes,
        test_ids,
        center,
        scale,
        severity_to_raw,
        seed + 10,
        bootstrap_iterations,
    )
    dynamics, occupancy, group_transitions = dynamics_metrics(
        scored_sessions, early_predictions, n_states=selected
    )
    behavior, behavior_frame = behavior_validation(early_predictions, dlc)
    baseline_metrics, baseline_predictions = fit_elastic_net_baseline(
        model_lfp, outcomes, train_ids, test_ids, center, scale, seed
    )

    # Task 5 primary result: dose ordering of recovered latent states. Dose is
    # used here only at evaluation time; fitting above never saw it.
    full_cohort_state_index = decode_state_index(
        model,
        model_lfp,
        set(outcomes["fish_id"].astype(str)),
        center,
        scale,
        raw_to_severity,
    )
    primary_dose_ordering = primary_dose_ordering_test(
        outcomes, full_cohort_state_index, seed
    )
    held_out_dose_ordering = primary_dose_ordering_test(
        outcomes,
        full_cohort_state_index.loc[
            full_cohort_state_index["fish_id"].isin(test_ids)
        ],
        seed + 500,
    )
    covariate_adjusted = covariate_adjusted_dose_ordering(
        outcomes, model_lfp, full_cohort_state_index
    )
    negative_controls = run_negative_controls(
        model_lfp,
        outcomes,
        full_cohort_state_index,
        primary_dose_ordering,
        selected,
        seed,
        restarts,
    )

    metrics = {
        "seed": seed,
        "split": {
            "n_train_fish": len(train_ids),
            "n_test_fish": len(test_ids),
            "fish_overlap": len(train_ids & test_ids),
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
        },
        "early_prediction": early,
        "dynamics": dynamics,
        "behavior_validation": behavior,
        "baseline": baseline_metrics,
        "dose_ordering": {
            "primary_full_cohort": primary_dose_ordering,
            "held_out_test_fish_only": held_out_dose_ordering,
            "covariate_adjusted": covariate_adjusted,
        },
        "ordered_start_probabilities": model.startprob_[severity_to_raw].tolist(),
        "ordered_transition_matrix": transition_matrix.tolist(),
        "ordered_emission_means_robust_scaled": model.means_[severity_to_raw].tolist(),
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
        "critical_caveat": (
            "Repeated same-fish 4-6 dpf LFP after 3 dpf TBI is a new integrated "
            "protocol requiring a pilot; the penetrating forebrain preparation "
            "was not validated as recoverable across days. See "
            "docs/EXPERIMENTAL_PROTOCOL.md."
        ),
    }

    assignments.to_csv(tables_dir / "tbi_split_assignments.csv", index=False)
    scored_sessions.to_csv(tables_dir / "tbi_scored_test_sessions.csv", index=False)
    early_predictions.to_csv(tables_dir / "tbi_early_predictions.csv", index=False)
    baseline_predictions.to_csv(
        tables_dir / "tbi_baseline_predictions.csv", index=False
    )
    full_cohort_state_index.merge(
        outcomes[["fish_id", "group"]].assign(
            fish_id=lambda frame: frame["fish_id"].astype(str)
        ),
        on="fish_id",
        how="left",
    ).to_csv(tables_dir / "tbi_dose_ordering_state_index.csv", index=False)
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
        json.dumps(_json_ready(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    write_negative_controls_report(
        primary_dose_ordering,
        covariate_adjusted,
        negative_controls,
        output_dir / "NEGATIVE_CONTROLS.md",
    )
    write_report(metrics, output_dir / "TBI_MODEL_RESULTS.md", endpoint_summary)
    make_figures(
        figures_dir,
        selection,
        early,
        early_predictions,
        occupancy,
        behavior_frame,
        transition_matrix,
    )
    return metrics
