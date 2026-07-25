"""Shared schema, loading, and sequence utilities for the TBI data template.

The LFP feature matrix is intentionally kept separate from injury metadata,
DeepLabCut-derived behavior, and reference labels. Placeholder rows are allowed
for data entry but are blocked from normal analysis until explicitly marked
``analysis_ready``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# Command-line defaults are intentionally relative to the invocation directory.
# Deriving them from ``__file__`` breaks wheel installs by pointing into
# site-packages, where the repository data and results directories do not exist.
REPOSITORY_ROOT = Path.cwd()
DATA_DIR = REPOSITORY_ROOT / "data" / "template"
RESULTS_DIR = REPOSITORY_ROOT / "results"

LFP_CSV = DATA_DIR / "tbi_4_6dpf_lfp_timeseries.csv"
OUTCOMES_CSV = DATA_DIR / "tbi_4_6dpf_fish_outcomes.csv"
DLC_CSV = DATA_DIR / "tbi_4_6dpf_dlc_behavior.csv"
WORKBOOK_PATH = DATA_DIR / "TBI_4_6dpf_data_template.xlsx"

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
RECORD_STATUS = "record_status"
PLACEHOLDER_STATUS = "placeholder_pending_replacement"
ANALYSIS_READY_STATUS = "analysis_ready"
ALLOWED_RECORD_STATUSES = {PLACEHOLDER_STATUS, ANALYSIS_READY_STATUS}


def load_dataset(
    lfp_path: Path | str = LFP_CSV,
    outcomes_path: Path | str = OUTCOMES_CSV,
    dlc_path: Path | str = DLC_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three normalized tables and enforce deterministic row order."""
    lfp = pd.read_csv(lfp_path)
    outcomes = pd.read_csv(outcomes_path)
    dlc = pd.read_csv(dlc_path)

    lfp = lfp.sort_values(["fish_id", "dpf"]).reset_index(drop=True)
    outcomes = outcomes.sort_values("fish_id").reset_index(drop=True)
    dlc = dlc.sort_values(["fish_id", "dpf"]).reset_index(drop=True)
    validate_dataset(lfp, outcomes, dlc)
    return lfp, outcomes, dlc


def validate_dataset(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    dlc: pd.DataFrame,
) -> None:
    """Reject schema violations that would invalidate the benchmark."""
    required_lfp = {
        "fish_id",
        "group",
        "batch",
        "tbi_dpf",
        "dpf",
        "days_post_tbi",
        "electrode_resistance_change_pct",
        "rms_noise_mv",
        "qc_pass",
        "measured_peak_pressure_kpa",
        DOSE_INDEX,
        RECORD_STATUS,
        TRUTH_STATE,
        *FEATURES,
    }
    required_outcomes = {
        "fish_id",
        "group",
        "batch",
        "survived_to_6dpf",
        DOSE_INDEX,
        TARGET,
        RECORD_STATUS,
    }
    required_dlc = {
        "fish_id",
        "group",
        "dpf",
        "dlc_mean_keypoint_likelihood",
        "dlc_pct_frames_below_pcutoff",
        "dlc_tracking_qc_pass",
        "dlc_mean_speed_mm_s",
        "dlc_rest_fraction",
        "dlc_whirlpool_rate_per_min",
        "dlc_behavior_abnormality_index",
        "manual_pts_stage_TRUTH",
        RECORD_STATUS,
    }
    missing = {
        "lfp": sorted(required_lfp - set(lfp)),
        "outcomes": sorted(required_outcomes - set(outcomes)),
        "dlc": sorted(required_dlc - set(dlc)),
    }
    if any(missing.values()):
        raise ValueError(f"Missing required columns: {missing}")

    for table_name, frame in (
        ("LFP", lfp),
        ("outcome", outcomes),
        ("behavior", dlc),
    ):
        if frame.empty:
            raise ValueError(f"{table_name} table cannot be empty.")
        fish_ids = frame["fish_id"].astype("string")
        if fish_ids.isna().any() or fish_ids.str.strip().eq("").any():
            raise ValueError(f"{table_name}.fish_id must contain non-empty identifiers.")

    if outcomes["fish_id"].duplicated().any():
        raise ValueError("Outcome fish_id values must be unique.")
    if lfp.duplicated(["fish_id", "dpf"]).any():
        raise ValueError("Duplicate LFP fish_id/dpf observations are not allowed.")
    if dlc.duplicated(["fish_id", "dpf"]).any():
        raise ValueError("Duplicate behavior fish_id/dpf observations are not allowed.")

    for table_name, frame in (("LFP", lfp), ("behavior", dlc)):
        if not is_numeric_dtype(frame["dpf"]) or frame["dpf"].isna().any():
            raise ValueError(f"{table_name}.dpf must contain integer day values.")
        dpf_values = frame["dpf"].to_numpy(float)
        if not np.isfinite(dpf_values).all() or not np.equal(
            dpf_values, np.floor(dpf_values)
        ).all():
            raise ValueError(f"{table_name}.dpf must contain integer day values.")
        if set(dpf_values.astype(int)) - set(OBSERVATION_DPF):
            raise ValueError(f"{table_name} table contains observations outside 4-6 dpf.")

    if not (lfp["tbi_dpf"] == INJURY_DPF).all():
        raise ValueError("Every row must identify the TBI insult at 3 dpf.")
    if not np.array_equal(
        lfp["days_post_tbi"].to_numpy(float),
        lfp["dpf"].to_numpy(float) - INJURY_DPF,
    ):
        raise ValueError("days_post_tbi must equal dpf - tbi_dpf.")

    for table_name, frame in (
        ("LFP", lfp),
        ("outcome", outcomes),
        ("behavior", dlc),
    ):
        if not set(frame["group"].astype(str)).issubset(GROUPS):
            raise ValueError(f"{table_name} table contains an unexpected experimental arm.")
        statuses = set(frame[RECORD_STATUS].astype(str))
        unexpected = statuses - ALLOWED_RECORD_STATUSES
        if unexpected:
            raise ValueError(
                f"{table_name} table has unsupported record_status values: "
                f"{sorted(unexpected)}"
            )

    for table_name, frame, columns in (
        (
            "LFP",
            lfp,
            [
                *FEATURES,
                "batch",
                "tbi_dpf",
                "dpf",
                "days_post_tbi",
                "electrode_resistance_change_pct",
                "rms_noise_mv",
                "measured_peak_pressure_kpa",
                DOSE_INDEX,
                TRUTH_STATE,
            ],
        ),
        ("outcome", outcomes, ["batch", DOSE_INDEX]),
        (
            "behavior",
            dlc,
            [
                "dpf",
                "dlc_mean_keypoint_likelihood",
                "dlc_pct_frames_below_pcutoff",
                "dlc_mean_speed_mm_s",
                "dlc_rest_fraction",
                "dlc_whirlpool_rate_per_min",
                "dlc_behavior_abnormality_index",
                "manual_pts_stage_TRUTH",
            ],
        ),
    ):
        for column in columns:
            if not is_numeric_dtype(frame[column]):
                raise ValueError(f"{table_name}.{column} must be numeric.")
            values = frame[column].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"{table_name}.{column} must contain finite values.")

    for table_name, frame, column in (
        ("LFP", lfp, "qc_pass"),
        ("outcome", outcomes, "survived_to_6dpf"),
        ("behavior", dlc, "dlc_tracking_qc_pass"),
    ):
        if not is_bool_dtype(frame[column]):
            raise ValueError(
                f"{table_name}.{column} must contain true boolean values, not strings."
            )

    endpoint = pd.to_numeric(outcomes[TARGET], errors="coerce")
    invalid_endpoint = outcomes[TARGET].notna() & ~endpoint.isin([0, 1])
    if invalid_endpoint.any():
        raise ValueError(f"{TARGET} must be binary (0/1) or missing.")
    if not lfp[TRUTH_STATE].isin([0, 1, 2]).all():
        raise ValueError(f"{TRUTH_STATE} must contain only 0, 1, or 2.")
    if not dlc["manual_pts_stage_TRUTH"].isin([0, 1, 2, 3]).all():
        raise ValueError("manual_pts_stage_TRUTH must contain only 0, 1, 2, or 3.")

    for column in NONNEGATIVE_LFP_FEATURES:
        if (lfp[column] < 0).any():
            raise ValueError(f"{column} cannot be negative.")
    if (lfp[DOSE_INDEX] < 0).any() or (outcomes[DOSE_INDEX] < 0).any():
        raise ValueError(f"{DOSE_INDEX} cannot be negative.")
    for column in (
        "measured_peak_pressure_kpa",
        "electrode_resistance_change_pct",
        "rms_noise_mv",
    ):
        if (lfp[column] < 0).any():
            raise ValueError(f"{column} cannot be negative.")
    if not dlc["dlc_mean_keypoint_likelihood"].between(0, 1).all():
        raise ValueError("dlc_mean_keypoint_likelihood must be between 0 and 1.")
    if not dlc["dlc_pct_frames_below_pcutoff"].between(0, 100).all():
        raise ValueError("dlc_pct_frames_below_pcutoff must be between 0 and 100.")
    if not dlc["dlc_rest_fraction"].between(0, 1).all():
        raise ValueError("dlc_rest_fraction must be between 0 and 1.")
    for column in (
        "dlc_mean_speed_mm_s",
        "dlc_whirlpool_rate_per_min",
        "dlc_behavior_abnormality_index",
    ):
        if (dlc[column] < 0).any():
            raise ValueError(f"{column} cannot be negative.")

    expected_qc = (
        (lfp["electrode_resistance_change_pct"] <= 50.0)
        & (lfp["rms_noise_mv"] < 0.2)
    )
    if not np.array_equal(expected_qc.to_numpy(), lfp["qc_pass"].to_numpy()):
        raise ValueError("qc_pass must reproduce the documented Eimon-style thresholds.")
    expected_tracking_qc = (
        (dlc["dlc_mean_keypoint_likelihood"] >= 0.90)
        & (dlc["dlc_pct_frames_below_pcutoff"] <= 10.0)
    )
    if not np.array_equal(
        expected_tracking_qc.to_numpy(),
        dlc["dlc_tracking_qc_pass"].to_numpy(),
    ):
        raise ValueError(
            "dlc_tracking_qc_pass must reproduce the documented likelihood thresholds."
        )

    lfp_fish = set(lfp["fish_id"])
    outcome_fish = set(outcomes["fish_id"])
    dlc_fish = set(dlc["fish_id"])
    if not lfp_fish <= outcome_fish or not dlc_fish <= outcome_fish:
        raise ValueError("Every session must map to one fish-level outcome row.")

    outcome_lookup = outcomes.set_index("fish_id")
    for table_name, frame in (("LFP", lfp), ("behavior", dlc)):
        expected_group = frame["fish_id"].map(outcome_lookup["group"])
        if not frame["group"].astype(str).equals(expected_group.astype(str)):
            raise ValueError(f"{table_name} group must match the fish outcome table.")
    expected_lfp_batch = lfp["fish_id"].map(outcome_lookup["batch"]).to_numpy(float)
    if not np.array_equal(lfp["batch"].to_numpy(float), expected_lfp_batch):
        raise ValueError("LFP batch must match the fish outcome table.")
    expected_lfp_dose = lfp["fish_id"].map(outcome_lookup[DOSE_INDEX]).to_numpy(float)
    if not np.allclose(
        lfp[DOSE_INDEX].to_numpy(float),
        expected_lfp_dose,
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError(f"LFP {DOSE_INDEX} must match the fish outcome table.")


def assert_analysis_ready(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    dlc: pd.DataFrame,
) -> None:
    """Block normal analysis until every placeholder row has been replaced."""
    pending = {
        "lfp": int((lfp[RECORD_STATUS] != ANALYSIS_READY_STATUS).sum()),
        "outcomes": int(
            (outcomes[RECORD_STATUS] != ANALYSIS_READY_STATUS).sum()
        ),
        "behavior": int((dlc[RECORD_STATUS] != ANALYSIS_READY_STATUS).sum()),
    }
    if any(pending.values()):
        raise ValueError(
            "Analysis blocked because placeholder records remain. Replace each "
            f"row, then set record_status='{ANALYSIS_READY_STATUS}'. Pending "
            f"counts: {pending}"
        )


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
