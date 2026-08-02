"""Shared schema, loading, and sequence utilities for the TBI LFP model.

The LFP feature matrix is kept strictly separate from injury metadata,
pose-derived behavior, and outcome fields. The latter exist only for quality
control and scoring, and never enter the model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .hmm import DiagonalGaussianHMM

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "data" / "measured"
RESULTS_DIR = REPOSITORY_ROOT / "results"

LFP_CSV = DATA_DIR / "tbi_4_6dpf_lfp_timeseries.csv"
OUTCOMES_CSV = DATA_DIR / "tbi_4_6dpf_fish_outcomes.csv"
BEHAVIOR_CSV = DATA_DIR / "tbi_4_6dpf_behavior.csv"
MANIFEST_JSON = DATA_DIR / "tbi_4_6dpf_manifest.json"

SEED = 42
INJURY_DPF = 3
OBSERVATION_DPF = (4, 5, 6)
TARGET_DPF = 6
PREDICTION_CUTOFF_DPF = 5
GROUPS = ("sham", "tbi_low", "tbi_moderate", "tbi_high")
GROUP_ORDER = {group: index for index, group in enumerate(GROUPS)}

# Three prespecified feature concepts, four columns. Locked in
# docs/PREREGISTRATION.md before refitting; do not add columns back without a
# dated amendment there. No protocol, dose, group, behavior, or outcome field
# is allowed into the HMM feature matrix.
#
# 1. Excitation-inhibition proxy: lfp_variance_uv2 + lfp_kurtosis together.
#    The intended proxy was the aperiodic 1/f spectral exponent
#    (specparam/FOOOF), which rising net excitation is expected to flatten.
#    No PSD or raw trace exists anywhere in this repository or its source
#    workbooks (Task 1 audit), so the exponent is not computable. Variance and
#    kurtosis of the amplitude distribution are the documented fallback: both
#    are expected to rise with the loss of the smooth, low-amplitude baseline
#    rhythm that dominates an inhibition-intact forebrain recording. This is a
#    real substitution, not a validated E/I marker — stated as a limitation
#    everywhere the proxy is used.
# 2. Epileptiform discharge rate: lfp_seizure_event_rate_per_h. The most
#    direct available correlate of epileptogenesis — a higher rate of scored
#    seizure-like LFP events is the closest thing to a face-valid electrographic
#    marker in this feature set.
# 3. Waveform-shape measure: lfp_fourth_power_mean_uv4. The intended measure
#    was line length (a raw-trace complexity measure), unavailable for the
#    same reason as the 1/f exponent — no raw or windowed trace exists. The
#    fourth-power mean is the documented fallback: it is dominated by rare,
#    high-amplitude excursions the way line length is dominated by rapid
#    trace excursions, so it captures a related aspect of waveform shape.
FEATURES = (
    "lfp_variance_uv2",
    "lfp_kurtosis",
    "lfp_seizure_event_rate_per_h",
    "lfp_fourth_power_mean_uv4",
)

# All four reduced features are non-negative and heavy-tailed; every one gets
# log1p before robust scaling.
NONNEGATIVE_LFP_FEATURES = FEATURES
LOG1P_FEATURES = FEATURES

# Behavioral 6 dpf endpoint. Three-valued: 1 qualifying event, 0 observed
# without one, NA never observed. See tbi_markov.dataset.
TARGET = "high_burden_state_dpf6"
DOSE_INDEX = "cumulative_pressure_burden_kpa_hits"

# All four reduced features (see FEATURES above) are prespecified to rise
# with injury severity; none is expected to fall. Shared by tbi_markov.modeling
# (primary model) and tbi_markov.dose_ordering (sham-only negative control),
# so it lives here rather than in either caller to avoid a circular import.
RISING_FEATURES = (
    "lfp_variance_uv2",
    "lfp_kurtosis",
    "lfp_seizure_event_rate_per_h",
    "lfp_fourth_power_mean_uv4",
)
FALLING_FEATURES: tuple[str, ...] = ()


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


def make_hmm(
    n_states: int,
    seed: int,
    restarts: int,
    n_iter: int = 160,
) -> DiagonalGaussianHMM:
    """Fixed hyperparameters shared by every HMM fit in this project,
    including the Task 5 sham-only negative-control refit."""
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


def load_dataset(
    lfp_path: Path | str = LFP_CSV,
    outcomes_path: Path | str = OUTCOMES_CSV,
    behavior_path: Path | str = BEHAVIOR_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three normalized tables and enforce deterministic row order."""
    lfp = pd.read_csv(lfp_path)
    outcomes = pd.read_csv(outcomes_path)
    behavior = pd.read_csv(behavior_path)

    lfp = lfp.sort_values(["fish_id", "dpf"]).reset_index(drop=True)
    outcomes = outcomes.sort_values("fish_id").reset_index(drop=True)
    behavior = behavior.sort_values(["fish_id", "dpf"]).reset_index(drop=True)
    validate_dataset(lfp, outcomes, behavior)
    return lfp, outcomes, behavior


def validate_dataset(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    behavior: pd.DataFrame,
) -> None:
    """Reject schema violations that would invalidate the analysis."""
    required_lfp = {
        "fish_id",
        "group",
        "tbi_dpf",
        "dpf",
        "days_post_tbi",
        "electrode_shift_pct",
        "rms_noise_mv",
        "qc_pass",
        DOSE_INDEX,
        *FEATURES,
    }
    required_outcomes = {
        "fish_id",
        "group",
        "survived_to_6dpf",
        DOSE_INDEX,
        TARGET,
    }
    required_behavior = {
        "fish_id",
        "dpf",
        "dlc_mean_keypoint_likelihood",
        "dlc_tracking_qc_pass",
    }
    missing = {
        "lfp": sorted(required_lfp - set(lfp)),
        "outcomes": sorted(required_outcomes - set(outcomes)),
        "behavior": sorted(required_behavior - set(behavior)),
    }
    if any(missing.values()):
        raise ValueError(f"Missing required columns: {missing}")

    if set(lfp["dpf"].unique()) - set(OBSERVATION_DPF):
        raise ValueError("LFP table contains observations outside 4-6 dpf.")
    if not (lfp["tbi_dpf"] == INJURY_DPF).all():
        raise ValueError("Every row must identify the insult at 3 dpf.")
    if lfp.duplicated(["fish_id", "dpf"]).any():
        raise ValueError("Duplicate fish_id/dpf observations are not allowed.")
    if not set(lfp["group"]).issubset(GROUPS):
        raise ValueError("Unexpected experimental arm.")

    numeric = lfp[list(FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("LFP model inputs must be finite.")
    for column in NONNEGATIVE_LFP_FEATURES:
        if (lfp[column] < 0).any():
            raise ValueError(f"{column} cannot be negative.")
    if (lfp[DOSE_INDEX] < 0).any() or (outcomes[DOSE_INDEX] < 0).any():
        raise ValueError(f"{DOSE_INDEX} cannot be negative.")

    # The published QC rule must reproduce the recorded qc_pass flag; otherwise
    # the two disagree about which sessions are usable for modeling.
    expected_qc = (
        (lfp["electrode_shift_pct"] <= 50.0)
        & (lfp["rms_noise_mv"] < 0.2)
    )
    if not np.array_equal(expected_qc.to_numpy(), lfp["qc_pass"].astype(bool).to_numpy()):
        raise ValueError("qc_pass must reproduce the documented Eimon-style thresholds.")

    lfp_fish = set(lfp["fish_id"])
    outcome_fish = set(outcomes["fish_id"])
    behavior_fish = set(behavior["fish_id"])
    if not lfp_fish <= outcome_fish or not behavior_fish <= outcome_fish:
        raise ValueError("Every session must map to one fish-level outcome row.")


def qc_sessions(lfp: pd.DataFrame) -> pd.DataFrame:
    """Return sessions eligible for model fitting under prespecified QC."""
    return lfp.loc[lfp["qc_pass"].astype(bool)].copy()


def fit_robust_scaler(
    lfp: pd.DataFrame,
    fish_ids: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only log1p + median/IQR scaling using the supplied fish."""
    frames = [
        frame
        for fish_id in set(fish_ids)
        if not (frame := _contiguous_baseline_prefix(lfp, fish_id)).empty
    ]
    if not frames:
        raise ValueError("No training observations available for preprocessing.")
    values = transform_feature_values(
        pd.concat(frames, ignore_index=True).loc[:, list(FEATURES)]
    )
    if not len(values):
        raise ValueError("No training observations available for preprocessing.")
    center = np.median(values, axis=0)
    q1, q3 = np.percentile(values, [25, 75], axis=0)
    scale = q3 - q1
    scale[scale < 1e-9] = 1.0
    return center, scale


def transform_feature_values(frame: pd.DataFrame) -> np.ndarray:
    """Variance-stabilize positive heavy-tailed features before robust scaling."""
    values = frame.loc[:, list(FEATURES)].to_numpy(float, copy=True)
    for feature in LOG1P_FEATURES:
        index = FEATURES.index(feature)
        values[:, index] = np.log1p(values[:, index])
    return values


def _contiguous_baseline_prefix(
    lfp: pd.DataFrame,
    fish_id: str,
    cutoff_dpf: int | None = None,
) -> pd.DataFrame:
    """Return the uninterrupted daily sequence beginning at 4 dpf.

    The caller supplies QC-passing sessions. A failed/missing middle session
    terminates the usable prefix so a 4-to-6 dpf gap is never treated as a
    single one-day Markov transition. Fish without an eligible 4 dpf baseline
    are excluded rather than being assigned the 4 dpf start distribution at a
    later age.
    """
    frame = lfp.loc[lfp["fish_id"] == fish_id].sort_values("dpf")
    if cutoff_dpf is not None:
        frame = frame.loc[frame["dpf"] <= cutoff_dpf]
    if frame.empty or int(frame.iloc[0]["dpf"]) != OBSERVATION_DPF[0]:
        return frame.iloc[0:0].copy()

    keep = 0
    expected_dpf = OBSERVATION_DPF[0]
    for dpf in frame["dpf"].astype(int):
        if dpf != expected_dpf:
            break
        keep += 1
        expected_dpf += 1
    return frame.iloc[:keep].copy()


def build_sequences(
    lfp: pd.DataFrame,
    fish_ids: Iterable[str],
    center: np.ndarray,
    scale: np.ndarray,
    cutoff_dpf: int | None = None,
) -> tuple[list[np.ndarray], list[str], dict[str, pd.DataFrame]]:
    """Build one contiguous 4 dpf-based sequence per fish.

    QC gaps terminate a sequence; they are never compressed into a one-step
    transition. ``cutoff_dpf`` supplies the causal observation boundary.
    """
    requested = set(fish_ids)
    sequences: list[np.ndarray] = []
    order: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for fish_id in pd.unique(lfp["fish_id"]):
        if fish_id not in requested:
            continue
        frame = _contiguous_baseline_prefix(lfp, fish_id, cutoff_dpf)
        if frame.empty:
            continue
        values = (transform_feature_values(frame[list(FEATURES)]) - center) / scale
        sequences.append(values)
        order.append(str(fish_id))
        frames[str(fish_id)] = frame.reset_index(drop=True)
    return sequences, order, frames


def concatenate_sequences(sequences: Iterable[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    sequences = [np.asarray(sequence, dtype=float) for sequence in sequences]
    if not sequences:
        raise ValueError("At least one sequence is required.")
    return np.vstack(sequences), [len(sequence) for sequence in sequences]
