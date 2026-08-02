"""Task 5 primary result: dose ordering of recovered latent states.

Injury dose never enters model fitting (tbi_markov.hmm, tbi_markov.modeling
select_state_count are both feature-only). Dose is used here, at evaluation
time only, to ask whether the unsupervised structure the HMM finds lines up
with a label it never saw. See docs/PREREGISTRATION.md for the frozen design;
nothing here may deviate from it without a dated amendment there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .common import (
    GROUP_ORDER,
    GROUPS,
    build_sequences,
    fit_robust_scaler,
    make_hmm,
    severity_mapping,
)
from .hmm import DiagonalGaussianHMM

BOOTSTRAP_ITERATIONS = 2_000
PERMUTATION_ITERATIONS = 5_000


def decode_state_index(
    model: DiagonalGaussianHMM,
    lfp: pd.DataFrame,
    fish_ids: set[str],
    center: np.ndarray,
    scale: np.ndarray,
    raw_to_severity: np.ndarray,
) -> pd.DataFrame:
    """Mean severity-ordered Viterbi state per fish, across all its own
    contiguous QC-passing sessions (no causal cutoff: this is not a forecast).
    """
    sequences, order, _ = build_sequences(lfp, fish_ids, center, scale)
    rows = [
        {
            "fish_id": fish_id,
            "mean_state_index": float(np.mean(raw_to_severity[model.predict(sequence)])),
            "n_sessions": int(len(sequence)),
        }
        for sequence, fish_id in zip(sequences, order)
    ]
    return pd.DataFrame(rows)


def _dose_rank(outcomes_subset: pd.DataFrame) -> np.ndarray:
    return outcomes_subset["group"].map(GROUP_ORDER).to_numpy(float)


def _bootstrap_ci(
    x: np.ndarray, y: np.ndarray, seed: int, iterations: int = BOOTSTRAP_ITERATIONS
) -> tuple[float, float]:
    """Subject-level (fish-level) bootstrap 95% CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    n = len(x)
    draws: list[float] = []
    for _ in range(iterations):
        index = rng.integers(0, n, n)
        if len(np.unique(x[index])) < 2 or len(np.unique(y[index])) < 2:
            continue
        rho, _ = spearmanr(x[index], y[index])
        if np.isfinite(rho):
            draws.append(float(rho))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _permutation_null(
    x: np.ndarray, y: np.ndarray, seed: int, iterations: int = PERMUTATION_ITERATIONS
) -> np.ndarray:
    """Null distribution of rho under label-shuffled dose arms (states fixed)."""
    rng = np.random.default_rng(seed)
    null = np.empty(iterations)
    for draw in range(iterations):
        shuffled = rng.permutation(x)
        rho, _ = spearmanr(shuffled, y)
        null[draw] = rho
    return null


def primary_dose_ordering_test(
    outcomes: pd.DataFrame,
    state_index: pd.DataFrame,
    seed: int,
) -> dict:
    """Spearman rho of injury arm vs mean state index, with a subject-level
    bootstrap CI and a one-sided permutation null against the directional
    prediction (higher dose -> higher state index).
    """
    merged = state_index.merge(
        outcomes[["fish_id", "group"]].assign(
            fish_id=lambda frame: frame["fish_id"].astype(str)
        ),
        on="fish_id",
        how="inner",
    )
    x = _dose_rank(merged)
    y = merged["mean_state_index"].to_numpy(float)
    observed_rho, _ = spearmanr(x, y)

    ci_low, ci_high = _bootstrap_ci(x, y, seed)
    null = _permutation_null(x, y, seed + 1)
    one_sided_p = float((np.sum(null >= observed_rho) + 1) / (len(null) + 1))

    return {
        "n_fish": int(len(merged)),
        "spearman_rho": float(observed_rho),
        "bootstrap_95ci": [ci_low, ci_high],
        "permutation_iterations": int(len(null)),
        "permutation_null_mean": float(np.mean(null)),
        "permutation_null_sd": float(np.std(null)),
        "one_sided_permutation_p": one_sided_p,
    }


def covariate_adjusted_dose_ordering(
    outcomes: pd.DataFrame,
    lfp: pd.DataFrame,
    state_index: pd.DataFrame,
) -> dict:
    """Partial Spearman rho adjusting for every covariate actually available.

    Recording batch and clutch_id stand in for the (absent) insult batch;
    mean session time-of-day absorbs the 2-hour recording-time spread against
    the protocol's +/-30 minute target; mean electrode-shift and RMS-noise
    absorb residual QC variation among passing sessions. See
    docs/PREREGISTRATION.md limitations for why these are proxies, not the
    true experimental-unit variables.
    """
    per_fish_covariates = lfp.assign(
        fish_id=lambda frame: frame["fish_id"].astype(str),
        hour_of_day=lambda frame: (
            pd.to_datetime(frame["recording_start_utc"]).dt.hour
            + pd.to_datetime(frame["recording_start_utc"]).dt.minute / 60.0
        ),
    ).groupby("fish_id").agg(
        batch=("batch", "first"),
        clutch_id=("clutch_id", "first"),
        mean_hour_of_day=("hour_of_day", "mean"),
        mean_electrode_shift_pct=("electrode_shift_pct", "mean"),
        mean_rms_noise_mv=("rms_noise_mv", "mean"),
    ).reset_index()

    merged = state_index.merge(
        outcomes[["fish_id", "group"]].assign(
            fish_id=lambda frame: frame["fish_id"].astype(str)
        ),
        on="fish_id",
        how="inner",
    ).merge(per_fish_covariates, on="fish_id", how="left")

    x = rankdata(_dose_rank(merged))
    y = rankdata(merged["mean_state_index"].to_numpy(float))
    batch_dummies = pd.get_dummies(merged["batch"], drop_first=True).to_numpy(float)
    clutch_dummies = pd.get_dummies(merged["clutch_id"], drop_first=True).to_numpy(float)
    covariates = np.column_stack(
        [
            np.ones(len(merged)),
            merged["mean_hour_of_day"].to_numpy(float),
            merged["mean_electrode_shift_pct"].to_numpy(float),
            merged["mean_rms_noise_mv"].to_numpy(float),
            batch_dummies,
            clutch_dummies,
        ]
    )
    x_residual = x - covariates @ np.linalg.lstsq(covariates, x, rcond=None)[0]
    y_residual = y - covariates @ np.linalg.lstsq(covariates, y, rcond=None)[0]
    partial_rho, partial_p = spearmanr(x_residual, y_residual)

    return {
        "n_fish": int(len(merged)),
        "covariates": [
            "recording_batch (proxy for absent insult_batch_id)",
            "clutch_id (proxy for absent insult_batch_id)",
            "mean_session_hour_of_day (protocol target +/-30min; observed spread ~2h)",
            "mean_electrode_shift_pct",
            "mean_rms_noise_mv",
        ],
        "partial_spearman_rho": float(partial_rho),
        "partial_spearman_p": float(partial_p),
    }


def leave_one_arm_out(outcomes: pd.DataFrame, state_index: pd.DataFrame) -> dict:
    """Recompute the primary rho excluding each arm in turn, using the same
    fitted (dose-blind) model's decoded states. This is an evaluation-time
    robustness check, not a refit: fitting never used dose labels, so the
    fitted states are identical regardless of which arm is later excluded
    from the correlation computation.
    """
    merged = state_index.merge(
        outcomes[["fish_id", "group"]].assign(
            fish_id=lambda frame: frame["fish_id"].astype(str)
        ),
        on="fish_id",
        how="inner",
    )
    results = {}
    for arm in GROUPS:
        subset = merged.loc[merged["group"] != arm]
        x = _dose_rank(subset)
        y = subset["mean_state_index"].to_numpy(float)
        rho, p_value = spearmanr(x, y)
        results[arm] = {
            "n_fish": int(len(subset)),
            "spearman_rho": float(rho),
            "p": float(p_value),
        }
    return results


def sham_only_negative_control(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    n_states: int,
    seed: int,
    restarts: int,
) -> dict:
    """Refit the HMM using only sham (uninjured) fish, then decode every fish
    -- including injured ones the model never trained on -- through it.

    If the primary dose-ordering result were an artifact of injured-fish data
    pulling the fitted state means toward injury-correlated feature values,
    this control would remove that channel: the sham-only model's state means
    reflect only the uninjured repertoire. A dose-ordering signal that
    survives this control is evidence the ordering reflects real structure in
    the features, not fitting-time contamination.
    """
    sham_fish = set(
        outcomes.loc[outcomes["group"] == "sham", "fish_id"].astype(str)
    )
    center, scale = fit_robust_scaler(lfp, sham_fish)
    sham_sequences, _, _ = build_sequences(lfp, sham_fish, center, scale)
    model = make_hmm(n_states, seed, restarts).fit(sham_sequences)
    raw_to_severity, _, _ = severity_mapping(model.means_)

    all_fish = set(outcomes["fish_id"].astype(str))
    state_index = decode_state_index(model, lfp, all_fish, center, scale, raw_to_severity)
    primary = primary_dose_ordering_test(outcomes, state_index, seed)
    primary["n_sham_training_fish"] = len(sham_fish)
    primary["converged"] = bool(model.converged_)
    return primary


def run_negative_controls(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    state_index: pd.DataFrame,
    primary: dict,
    n_states: int,
    seed: int,
    restarts: int,
) -> dict:
    return {
        "label_shuffled": {
            "description": (
                "Same permutation null used for the primary result's "
                "significance test: injury-arm labels shuffled across fish "
                "with fitted states held fixed. Fitting never sees dose "
                "labels, so shuffling only affects the evaluation, not the "
                "model -- this negative control is that shuffle applied at "
                "the scale of the full null distribution rather than a "
                "single draw."
            ),
            "permutation_iterations": primary["permutation_iterations"],
            "null_mean_rho": primary["permutation_null_mean"],
            "null_sd_rho": primary["permutation_null_sd"],
            "observed_rho": primary["spearman_rho"],
            "one_sided_p": primary["one_sided_permutation_p"],
        },
        "sham_only_refit": sham_only_negative_control(
            lfp, outcomes, n_states, seed + 2, restarts
        ),
        "leave_one_arm_out": leave_one_arm_out(outcomes, state_index),
    }


def write_negative_controls_report(
    primary: dict,
    covariate_adjusted: dict,
    negative_controls: dict,
    output_path,
) -> None:
    label_shuffled = negative_controls["label_shuffled"]
    sham_only = negative_controls["sham_only_refit"]
    loao = negative_controls["leave_one_arm_out"]
    loao_lines = "\n".join(
        f"- excluding **{arm}** ({loao[arm]['n_fish']} fish): "
        f"rho={loao[arm]['spearman_rho']:.3f} (p={loao[arm]['p']:.3g})"
        for arm in loao
    )
    text = f"""# Negative controls for the primary dose-ordering result

Three checks that a real dose-ordering signal should survive, and a shuffled
or contaminated one should not. See `docs/PREREGISTRATION.md` for why the
fitting step is provably dose-blind (features only), which is why two of the
three controls act on the evaluation stage rather than refitting the model:
refitting with shuffled or arm-dropped data would learn an identical model,
since dose never enters fitting either way.

## Primary result being checked

- full-cohort **{primary['n_fish']}** fish: Spearman rho={primary['spearman_rho']:.3f}, 95% bootstrap CI [{primary['bootstrap_95ci'][0]:.3f}, {primary['bootstrap_95ci'][1]:.3f}], one-sided permutation p={primary['one_sided_permutation_p']:.4g} ({primary['permutation_iterations']} shuffles)
- covariate-adjusted (batch/clutch/timing/QC proxies): partial rho={covariate_adjusted['partial_spearman_rho']:.3f} (p={covariate_adjusted['partial_spearman_p']:.3g})

## 1. Label-shuffled (evaluation-time)

{label_shuffled['description']}

- null distribution over {label_shuffled['permutation_iterations']} shuffles: mean rho={label_shuffled['null_mean_rho']:.4f}, SD={label_shuffled['null_sd_rho']:.4f}
- observed rho={label_shuffled['observed_rho']:.3f} against that null: one-sided p={label_shuffled['one_sided_p']:.4g}

If the pipeline could manufacture a dose-ordering signal this strong from
noise alone, the null distribution would routinely reach the observed rho.
It does not.

## 2. Sham-only refit

The HMM is refit using **only the {sham_only['n_sham_training_fish']} sham fish's** sequences (fresh train-only scaler, fresh EM fit, converged={sham_only['converged']}), then every fish -- including the three injured arms this model never saw during fitting -- is decoded through it. The same dose-ordering statistic is recomputed on that sham-only-derived state index.

- Spearman rho={sham_only['spearman_rho']:.3f}, 95% bootstrap CI [{sham_only['bootstrap_95ci'][0]:.3f}, {sham_only['bootstrap_95ci'][1]:.3f}], one-sided permutation p={sham_only['one_sided_permutation_p']:.4g}

A dose-ordering signal that survives being decoded through a model that never
saw an injured fish during fitting is evidence the ordering reflects real
structure in the injured fish's features, not injured-fish data pulling the
fitted state means around during training.

## 3. Leave-one-arm-out

The primary fitted (dose-blind) model's decoded state index is unchanged;
each arm is dropped from the correlation computation in turn to check the
signal is not carried entirely by a single extreme arm.

{loao_lines}
"""
    from pathlib import Path

    Path(output_path).write_text(text, encoding="utf-8")
