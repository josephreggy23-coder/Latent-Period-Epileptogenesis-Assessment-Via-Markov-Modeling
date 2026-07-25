"""Shared schema, loading, and sequence utilities for the synthetic TBI model.

The LFP feature matrix is intentionally kept separate from injury metadata,
DeepLabCut-derived behavior, and planted simulator truth.  The latter fields
exist only for quality control and benchmark scoring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "data" / "synthetic"
REAL_DATA_DIR = REPOSITORY_ROOT / "data" / "real"
RESULTS_DIR = REPOSITORY_ROOT / "results"
REAL_RESULTS_DIR = REPOSITORY_ROOT / "results_real"

LFP_CSV = DATA_DIR / "tbi_4_6dpf_lfp_timeseries.csv"
OUTCOMES_CSV = DATA_DIR / "tbi_4_6dpf_fish_outcomes.csv"
DLC_CSV = DATA_DIR / "tbi_4_6dpf_dlc_behavior.csv"
WORKBOOK_PATH = DATA_DIR / "TBI_4_6dpf_synthetic_data.xlsx"

REAL_LFP_CSV = REAL_DATA_DIR / "tbi_4_6dpf_real_lfp_timeseries.csv"
REAL_OUTCOMES_CSV = REAL_DATA_DIR / "tbi_4_6dpf_real_fish_outcomes.csv"
REAL_DLC_CSV = REAL_DATA_DIR / "tbi_4_6dpf_real_behavior.csv"

SEED = 42
INJURY_DPF = 3
OBSERVATION_DPF = (4, 5, 6)
TARGET_DPF = 6
PREDICTION_CUTOFF_DPF = 5
GROUPS = ("sham", "tbi_low", "tbi_moderate", "tbi_high")
GROUP_ORDER = {group: index for index, group in enumerate(GROUPS)}

# Eimon-inspired session summaries.  No protocol, dose, group, behavior, or
# planted-truth field is allowed into the HMM feature matrix.
FEATURES = (
    "lfp_mean_uv",
    "lfp_variance_uv2",
    "lfp_skewness",
    "lfp_kurtosis",
    "lfp_fourth_power_mean_uv4",
    "lfp_seizure_event_rate_per_h",
    "lfp_ica_complexity",
)

NONNEGATIVE_LFP_FEATURES = (
    "lfp_variance_uv2",
    "lfp_kurtosis",
    "lfp_fourth_power_mean_uv4",
    "lfp_seizure_event_rate_per_h",
    "lfp_ica_complexity",
)

LOG1P_FEATURES = (
    "lfp_variance_uv2",
    "lfp_kurtosis",
    "lfp_fourth_power_mean_uv4",
    "lfp_seizure_event_rate_per_h",
)

TRUTH_STATE = "hidden_state_TRUTH"
TARGET = "high_burden_state_dpf6_TRUTH"
DOSE_INDEX = "cumulative_pressure_burden_kpa_hits"


def load_dataset(
    lfp_path: Path | str = LFP_CSV,
    outcomes_path: Path | str = OUTCOMES_CSV,
    dlc_path: Path | str = DLC_CSV,
    *,
    expect_synthetic: bool = True,
    require_truth: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three normalized tables and enforce deterministic row order.

    The same reader serves the synthetic benchmark and the real recording; the
    provenance and planted-truth expectations are the only difference. See
    ``tbi_markov.real_data.load_real_dataset`` for the real-data entry point.
    """
    lfp = pd.read_csv(lfp_path)
    outcomes = pd.read_csv(outcomes_path)
    dlc = pd.read_csv(dlc_path)

    lfp = lfp.sort_values(["fish_id", "dpf"]).reset_index(drop=True)
    outcomes = outcomes.sort_values("fish_id").reset_index(drop=True)
    dlc = dlc.sort_values(["fish_id", "dpf"]).reset_index(drop=True)
    validate_dataset(
        lfp,
        outcomes,
        dlc,
        expect_synthetic=expect_synthetic,
        require_truth=require_truth,
    )
    return lfp, outcomes, dlc


def validate_dataset(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    dlc: pd.DataFrame,
    *,
    expect_synthetic: bool = True,
    require_truth: bool = True,
) -> None:
    """Reject schema violations that would invalidate the benchmark.

    ``expect_synthetic`` asserts the direction of the ``is_synthetic`` provenance
    flag: the synthetic benchmark requires it True on every row, while a real
    recording must be marked False on every row. The flag is always required and
    always checked, so synthetic and measured data can never be silently mixed.

    ``require_truth`` demands the planted ``hidden_state_TRUTH`` column. Real
    animals have no planted latent state, so real datasets are loaded with
    ``require_truth=False`` and simply cannot be scored for state recovery.
    """
    required_lfp = {
        "fish_id",
        "group",
        "tbi_dpf",
        "dpf",
        "days_post_tbi",
        "electrode_shift_pct",
        "rms_noise_mv",
        "qc_pass",
        "is_synthetic",
        DOSE_INDEX,
        *FEATURES,
    }
    if require_truth:
        required_lfp.add(TRUTH_STATE)
    required_outcomes = {
        "fish_id",
        "group",
        "survived_to_6dpf",
        DOSE_INDEX,
        TARGET,
        "is_synthetic",
    }
    required_dlc = {
        "fish_id",
        "dpf",
        "dlc_mean_keypoint_likelihood",
        "dlc_tracking_qc_pass",
        "is_synthetic",
    }
    missing = {
        "lfp": sorted(required_lfp - set(lfp)),
        "outcomes": sorted(required_outcomes - set(outcomes)),
        "dlc": sorted(required_dlc - set(dlc)),
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
    provenance = "synthetic" if expect_synthetic else "real (is_synthetic=False)"
    for name, frame in (("LFP", lfp), ("outcome", outcomes), ("behavior", dlc)):
        flags = frame["is_synthetic"].astype(bool)
        if bool(flags.all()) is not expect_synthetic or bool(flags.any()) is not expect_synthetic:
            raise ValueError(
                f"Every {name} row must remain explicitly marked {provenance}."
            )

    numeric = lfp[list(FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("LFP model inputs must be finite.")
    for column in NONNEGATIVE_LFP_FEATURES:
        if (lfp[column] < 0).any():
            raise ValueError(f"{column} cannot be negative.")
    if (lfp[DOSE_INDEX] < 0).any() or (outcomes[DOSE_INDEX] < 0).any():
        raise ValueError(f"{DOSE_INDEX} cannot be negative.")

    expected_qc = (
        (lfp["electrode_shift_pct"] <= 50.0)
        & (lfp["rms_noise_mv"] < 0.2)
    )
    if not np.array_equal(expected_qc.to_numpy(), lfp["qc_pass"].astype(bool).to_numpy()):
        raise ValueError("qc_pass must reproduce the documented Eimon-style thresholds.")

    lfp_fish = set(lfp["fish_id"])
    outcome_fish = set(outcomes["fish_id"])
    dlc_fish = set(dlc["fish_id"])
    if not lfp_fish <= outcome_fish or not dlc_fish <= outcome_fish:
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
