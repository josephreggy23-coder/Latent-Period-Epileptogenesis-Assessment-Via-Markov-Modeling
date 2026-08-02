"""Shared test fixtures.

These builders construct small in-memory tables directly. They exist to
exercise split, prefix, and propagation logic deterministically without
touching the measured workbooks, which are large and slow to parse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tbi_markov.common import FEATURES, GROUPS, TARGET


def make_lfp(
    n_per_arm: int = 6,
    dpf_by_fish: dict[str, tuple[int, ...]] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build an LFP table with a severity gradient across arms.

    Feature means rise with arm index so severity ordering is well defined and
    the HMM has real structure to find.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for arm_index, group in enumerate(GROUPS):
        for member in range(n_per_arm):
            fish_id = f"F{arm_index}{member:03d}"
            days = (dpf_by_fish or {}).get(fish_id, (4, 5, 6))
            for dpf in days:
                row = {
                    "fish_id": fish_id,
                    "group": group,
                    "batch": 1 + (member % 3),
                    "tbi_dpf": 3,
                    "dpf": dpf,
                    "days_post_tbi": dpf - 3,
                    "electrode_shift_pct": 5.0,
                    "rms_noise_mv": 0.05,
                    "qc_pass": True,
                    "measured_peak_pressure_kpa": 100.0 * arm_index,
                    "cumulative_pressure_burden_kpa_hits": 100.0 * arm_index,
                    "clutch_id": f"CL{1 + (member % 3):02d}",
                    "recording_start_utc": pd.Timestamp("2026-07-14 14:00:00")
                    + pd.Timedelta(minutes=15 * (member % 8)),
                }
                level = arm_index + 0.5 * (dpf - 4)
                for index, feature in enumerate(FEATURES):
                    value = level + 0.1 * index + rng.normal(0, 0.05)
                    row[feature] = float(abs(value))
                rows.append(row)
    return pd.DataFrame(rows)


def make_outcomes(lfp: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Fish-level outcomes whose positive rate rises with arm index."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for fish_id, frame in lfp.groupby("fish_id", sort=True):
        group = frame["group"].iloc[0]
        arm_index = GROUPS.index(group)
        rows.append(
            {
                "fish_id": fish_id,
                "group": group,
                "batch": int(frame["batch"].iloc[0]),
                "survived_to_6dpf": bool(6 in set(frame["dpf"])),
                "cumulative_pressure_burden_kpa_hits": float(
                    frame["cumulative_pressure_burden_kpa_hits"].iloc[0]
                ),
                TARGET: float(rng.random() < 0.15 + 0.25 * arm_index),
            }
        )
    return pd.DataFrame(rows)


def make_behavior(lfp: pd.DataFrame) -> pd.DataFrame:
    """One behavior row per LFP session."""
    frame = lfp[["fish_id", "group", "dpf"]].drop_duplicates().reset_index(drop=True)
    frame["dlc_mean_keypoint_likelihood"] = 0.95
    frame["dlc_tracking_qc_pass"] = True
    frame["dlc_mean_speed_mm_s"] = 1.5
    frame["dlc_rest_fraction"] = 0.3
    frame["dlc_whirlpool_rate_per_min"] = 0.1
    frame["dlc_behavior_abnormality_index"] = 0.2
    frame["manual_pts_stage_observed"] = 0
    return frame


def make_tables(
    n_per_arm: int = 6,
    dpf_by_fish: dict[str, tuple[int, ...]] | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lfp = make_lfp(n_per_arm=n_per_arm, dpf_by_fish=dpf_by_fish, seed=seed)
    return lfp, make_outcomes(lfp, seed=seed), make_behavior(lfp)
