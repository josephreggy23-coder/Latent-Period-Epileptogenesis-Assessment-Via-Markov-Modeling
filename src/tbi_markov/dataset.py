"""Ingest and analyze the larval-zebrafish TBI recording.

Reads the source workbooks, normalizes them into three tables, and runs the
Markov analysis: 240 fish, 3 dpf weight-drop TBI, single forebrain electrode,
LFP at 4/5/6 dpf, sham plus three injury doses.

Source workbooks:

``actualdata1(lfp).xlsx``        sheet ``LFP Recordings``   one row per session
``actualdata(behavioral).xlsx``  sheets ``Behavioral Outcomes``, ``Event Log``

Two ingestion decisions carry scientific weight:

1. **The 6 dpf endpoint is three-valued.** Positive if the blinded scorer logged
   at least one qualifying event (Baraban stage >= 2 with passing pose QC) in
   the 6 dpf session, negative if the fish was observed at 6 dpf without one,
   and ``NA`` if there is no evidence it was observed at all. An unobserved
   animal has an unknown outcome, not a negative one. The endpoint is derived
   purely from behaviour and shares no variable with the LFP feature matrix, so
   the forecast target stays independent of the model's inputs.
2. **Behaviour is per-event, not per-session.** The Event Log lists scored
   events; sessions with none are absent entirely. They are materialized as
   zero-event rows rather than dropped, because "no scored behaviour" is an
   observation, and omitting them would restrict the behavioural validation to
   the abnormal subset and bias it.

There is no latent-state ground truth, so state-recovery accuracy is not
measurable here and none is reported.

The recording modality carries a caveat the modelling cannot resolve: the
electrode metadata (forebrain target, 1 M chloride, ~2.5-3.6 MOhm) matches the
Eimon penetrating-electrode preparation, which was demonstrated at 7 dpf and has
not been validated as a recoverable, repeated measurement in the same larva at
4, 5, and 6 dpf. Per-fish longitudinal state transitions therefore rest on an
assumption this dataset cannot verify. See ``docs/EXPERIMENTAL_PROTOCOL.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    BEHAVIOR_CSV,
    DATA_DIR,
    DOSE_INDEX,
    FEATURES,
    GROUPS,
    INJURY_DPF,
    LFP_CSV,
    OBSERVATION_DPF,
    OUTCOMES_CSV,
    RESULTS_DIR,
    SEED,
    TARGET,
    TARGET_DPF,
    load_dataset,
)
from .modeling import run_analysis

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LFP_WORKBOOK = REPOSITORY_ROOT / "actualdata1(lfp).xlsx"
DEFAULT_BEHAVIOR_WORKBOOK = REPOSITORY_ROOT / "actualdata(behavioral).xlsx"

LFP_SHEET = "LFP Recordings"
OUTCOMES_SHEET = "Behavioral Outcomes"
EVENTS_SHEET = "Event Log"

# Baraban stage >= 2 with passing pose QC, as flagged by the blinded scorer.
QUALIFYING_FLAG = "event_qualifies"
STAGE_COLUMN = "manual_baraban_stage"

BEHAVIOR_REFERENCE_QUANTILE = 0.90


# ===========================================================================
# Ingestion
# ===========================================================================
def build_tables(
    lfp_workbook: Path | str = DEFAULT_LFP_WORKBOOK,
    behavior_workbook: Path | str = DEFAULT_BEHAVIOR_WORKBOOK,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the source workbooks and emit the three normalized tables."""
    raw_lfp = pd.read_excel(lfp_workbook, sheet_name=LFP_SHEET)
    raw_outcomes = pd.read_excel(behavior_workbook, sheet_name=OUTCOMES_SHEET)
    raw_events = pd.read_excel(behavior_workbook, sheet_name=EVENTS_SHEET)

    lfp = _normalize_lfp(raw_lfp)
    endpoint = _dpf6_endpoint(raw_events, raw_lfp)
    outcomes = _normalize_outcomes(raw_lfp, raw_outcomes, raw_events, endpoint)
    behavior = _normalize_behavior(raw_lfp, raw_events)
    return lfp, outcomes, behavior


def _normalize_lfp(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    keep = [
        "fish_id", "group", "batch", "clutch_id", "plate_id", "well_id",
        "tbi_dpf", "dpf", "days_post_tbi", "hours_post_tbi",
        "n_weight_drops", "nominal_peak_pressure_kpa",
        "measured_peak_pressure_kpa", DOSE_INDEX,
        "channel", "electrode_target", "electrode_impedance_mohm",
        "rms_noise_mv", "electrode_shift_pct", "qc_pass",
        "sampling_rate_hz", "recording_duration_min", "artifact_fraction",
        *FEATURES,
    ]
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    frame["fish_id"] = frame["fish_id"].astype(str)
    frame["group"] = frame["group"].astype(str)
    return frame.sort_values(["fish_id", "dpf"]).reset_index(drop=True)


def _dpf6_endpoint(raw_events: pd.DataFrame, raw_lfp: pd.DataFrame) -> pd.Series:
    """The 6 dpf high-burden behavioural endpoint.

    Three-valued, deliberately:

    ``1`` at least one qualifying blinded event (Baraban stage >= 2 with
          passing pose QC) in the 6 dpf session;
    ``0`` the fish was observed at 6 dpf and no qualifying event occurred;
    ``NA`` there is no evidence the fish was observed at 6 dpf at all.

    The third case matters. A fish that died, was lost, or lacks 6 dpf coverage
    has an *unknown* endpoint, not a negative one. Coding absence as 0 would
    silently inflate the negative class with unobserved animals and bias every
    downstream rate. Only sessions that actually happened count as evidence of
    observation: a 6 dpf LFP recording, or any 6 dpf behavioural row (including
    a normal one - the Event Log stores normal swim bouts too, so presence in
    the log is proof of observation, and absence alone is not proof of absence).
    """
    observed = set(
        raw_lfp.loc[raw_lfp["dpf"] == TARGET_DPF, "fish_id"].astype(str)
    ) | set(
        raw_events.loc[
            raw_events["observation_dpf"] == TARGET_DPF, "fish_id"
        ].astype(str)
    )
    qualifying = raw_events.loc[raw_events[QUALIFYING_FLAG].astype(bool)]
    positive = set(
        qualifying.loc[
            qualifying["observation_dpf"] == TARGET_DPF, "fish_id"
        ].astype(str)
    )
    return pd.Series(
        {
            fish_id: (1.0 if fish_id in positive else 0.0)
            for fish_id in sorted(observed)
        },
        dtype=float,
    )


def _normalize_outcomes(
    raw_lfp: pd.DataFrame,
    raw_outcomes: pd.DataFrame,
    raw_events: pd.DataFrame,
    endpoint: pd.Series,
) -> pd.DataFrame:
    meta = (
        raw_lfp.assign(fish_id=lambda frame: frame["fish_id"].astype(str))
        .groupby("fish_id")
        .agg(
            group=("group", "first"),
            batch=("batch", "first"),
            clutch_id=("clutch_id", "first"),
            plate_id=("plate_id", "first"),
            n_weight_drops=("n_weight_drops", "first"),
            measured_peak_pressure_kpa=("measured_peak_pressure_kpa", "mean"),
            **{DOSE_INDEX: (DOSE_INDEX, "max")},
            last_observed_dpf=("dpf", "max"),
            n_lfp_sessions=("dpf", "size"),
        )
        .reset_index()
    )
    frame = meta.merge(
        raw_outcomes.assign(fish_id=lambda f: f["fish_id"].astype(str)),
        on="fish_id",
        how="left",
        validate="one_to_one",
    )
    # Left as NaN where the fish was never observed at 6 dpf; see _dpf6_endpoint.
    # Downstream, fish_level_split puts these in a "missing" stratum and
    # early_prediction drops them, so an unknown endpoint is never scored.
    frame[TARGET] = frame["fish_id"].map(endpoint).astype(float)

    # An LFP session at 6 dpf is direct evidence the animal was alive then.
    frame["survived_to_6dpf"] = frame["last_observed_dpf"] >= TARGET_DPF

    # Highest blinded Baraban stage reached at any session, for description only.
    stage = (
        raw_events.assign(fish_id=lambda f: f["fish_id"].astype(str))
        .groupby("fish_id")[STAGE_COLUMN]
        .max()
    )
    frame["max_manual_baraban_stage_observed"] = (
        frame["fish_id"].map(stage).fillna(0).astype(int)
    )
    return frame.sort_values("fish_id").reset_index(drop=True)


def _normalize_behavior(
    raw_lfp: pd.DataFrame,
    raw_events: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the per-event log into one row per fish-session.

    Sessions with no scored event are emitted with zero event rates rather than
    omitted: a session in which the blinded scorer logged nothing is a real
    observation of normal behaviour, and dropping those rows would restrict the
    behavioural validation to the abnormal subset and bias it.
    """
    events = raw_events.assign(fish_id=lambda f: f["fish_id"].astype(str)).copy()
    events["_minutes"] = events["session_duration_s"].astype(float) / 60.0
    events["_abnormal"] = events[STAGE_COLUMN] >= 1
    events["_severe"] = events[STAGE_COLUMN] >= 2
    events["_whirlpool"] = events["behavior_class"] == "whirlpool"
    events["_convulsion"] = events["behavior_class"] == "convulsion_posture_loss"

    grouped = events.groupby(["fish_id", "observation_dpf"])
    aggregated = grouped.agg(
        session_minutes=("_minutes", "first"),
        n_scored_events=(STAGE_COLUMN, "size"),
        n_abnormal_events=("_abnormal", "sum"),
        n_severe_events=("_severe", "sum"),
        n_whirlpool_events=("_whirlpool", "sum"),
        n_convulsion_events=("_convulsion", "sum"),
        event_seconds=("event_duration_s", "sum"),
        manual_pts_stage_observed=(STAGE_COLUMN, "max"),
        dlc_mean_speed_mm_s=("mean_velocity_mm_s", "mean"),
        dlc_max_speed_mm_s=("peak_velocity_mm_s", "max"),
        dlc_mean_tail_bend_deg=("rms_tail_angle_deg", "mean"),
        dlc_max_tail_bend_deg=("max_tail_angle_deg", "max"),
        dlc_mean_keypoint_likelihood=("mean_keypoint_likelihood", "mean"),
        dlc_tracking_qc_pass=("dlc_tracking_qc_pass", "all"),
    ).reset_index().rename(columns={"observation_dpf": "dpf"})

    # Materialize every fish-session that has an LFP recording, so zero-event
    # sessions are represented instead of silently missing.
    scaffold = (
        raw_lfp.assign(fish_id=lambda f: f["fish_id"].astype(str))[
            ["fish_id", "group", "dpf"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    frame = scaffold.merge(aggregated, on=["fish_id", "dpf"], how="left")

    default_minutes = float(
        events["_minutes"].median() if len(events) else 5.0
    )
    frame["session_minutes"] = frame["session_minutes"].fillna(default_minutes)
    count_columns = [
        "n_scored_events", "n_abnormal_events", "n_severe_events",
        "n_whirlpool_events", "n_convulsion_events", "event_seconds",
        "manual_pts_stage_observed",
    ]
    frame[count_columns] = frame[count_columns].fillna(0)
    # No scored event means nothing failed pose QC in that session. Filling an
    # object column then casting trips a pandas downcasting FutureWarning, so
    # the null check is done explicitly.
    frame["dlc_tracking_qc_pass"] = (
        frame["dlc_tracking_qc_pass"]
        .map(lambda flag: True if pd.isna(flag) else bool(flag))
        .astype(bool)
    )

    minutes = frame["session_minutes"].to_numpy(float)
    frame["dlc_abnormal_event_rate_per_min"] = frame["n_abnormal_events"] / minutes
    frame["dlc_severe_event_rate_per_min"] = frame["n_severe_events"] / minutes
    frame["dlc_whirlpool_rate_per_min"] = frame["n_whirlpool_events"] / minutes
    frame["dlc_convulsion_rate_per_min"] = frame["n_convulsion_events"] / minutes
    # Fraction of the session NOT inside any scored event.
    frame["dlc_rest_fraction"] = np.clip(
        1.0 - frame["event_seconds"] / (minutes * 60.0), 0.0, 1.0
    )
    frame["dlc_behavior_abnormality_index"] = _abnormality_index(frame)
    return frame.sort_values(["fish_id", "dpf"]).reset_index(drop=True)


def _abnormality_index(frame: pd.DataFrame) -> pd.Series:
    """Composite behavioural abnormality score, defined for zero-event sessions.

    Built only from event-rate and stage terms, which are well defined when the
    scorer logged nothing (they are simply zero). Kinematic columns are left out
    of the index precisely because they are undefined without an event, and
    imputing them would manufacture signal. Each term is divided by a
    label-free upper quantile of its own distribution so the terms are
    commensurate; the endpoint is never consulted.
    """
    terms = []
    for column in (
        "dlc_abnormal_event_rate_per_min",
        "dlc_severe_event_rate_per_min",
        "dlc_whirlpool_rate_per_min",
        "dlc_convulsion_rate_per_min",
    ):
        values = frame[column].to_numpy(float)
        reference = float(np.quantile(values, BEHAVIOR_REFERENCE_QUANTILE))
        terms.append(values / (reference if reference > 0 else 1.0))
    terms.append(frame["manual_pts_stage_observed"].to_numpy(float) / 3.0)
    return pd.Series(np.mean(terms, axis=0), index=frame.index).round(6)


# ===========================================================================
# Validation
# ===========================================================================
def validate_tables(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    behavior: pd.DataFrame,
) -> None:
    """Check the assumptions the analysis depends on."""
    if set(lfp["group"]) - set(GROUPS):
        raise ValueError(f"Unexpected arm: {set(lfp['group']) - set(GROUPS)}")
    if set(lfp["dpf"].unique()) - set(OBSERVATION_DPF):
        raise ValueError("LFP table contains sessions outside 4-6 dpf.")
    if not (lfp["tbi_dpf"] == INJURY_DPF).all():
        raise ValueError("Every LFP row must record the insult at 3 dpf.")
    if lfp.duplicated(["fish_id", "dpf"]).any():
        raise ValueError("Duplicate fish_id/dpf sessions in the LFP table.")

    # The published QC rule must reproduce the vendor's own qc_pass flag; if it
    # does not, the two disagree about which sessions are usable.
    expected = (lfp["electrode_shift_pct"] <= 50.0) & (lfp["rms_noise_mv"] < 0.2)
    if not np.array_equal(expected.to_numpy(), lfp["qc_pass"].astype(bool).to_numpy()):
        raise ValueError(
            "qc_pass does not reproduce the documented electrode-shift/noise rule."
        )
    values = lfp[list(FEATURES)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("LFP model inputs must be finite.")
    resolved = outcomes[TARGET].notna()
    if not resolved.any():
        raise ValueError("No fish has a resolved 6 dpf endpoint.")
    if resolved.mean() < 0.5:
        raise ValueError(
            "More than half of fish lack a 6 dpf observation; the endpoint "
            "cannot carry the analysis."
        )
    if outcomes[TARGET].nunique() < 2:
        raise ValueError("The endpoint must contain both classes.")
    if set(behavior["fish_id"]) - set(outcomes["fish_id"]):
        raise ValueError("Behaviour rows must map to fish-level outcome rows.")


def write_tables(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    behavior: pd.DataFrame,
    output_dir: Path | str = DATA_DIR,
) -> dict:
    """Persist the normalized tables and a small provenance manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lfp.to_csv(output_dir / LFP_CSV.name, index=False)
    outcomes.to_csv(output_dir / OUTCOMES_CSV.name, index=False)
    behavior.to_csv(output_dir / BEHAVIOR_CSV.name, index=False)
    manifest = {
        "n_fish": int(len(outcomes)),
        "n_lfp_sessions": int(len(lfp)),
        "n_behavior_sessions": int(len(behavior)),
        "observation_dpf": sorted(int(value) for value in lfp["dpf"].unique()),
        "groups": {
            group: int((outcomes["group"] == group).sum()) for group in GROUPS
        },
        "endpoint": {
            "name": TARGET,
            "definition": (
                "At least one qualifying blinded behavioural event (Baraban "
                "stage >= 2 with passing pose QC) in the 6 dpf session."
            ),
            "n_positive": int(outcomes[TARGET].sum()),
            "n_negative": int((outcomes[TARGET] == 0).sum()),
            "n_unresolved": int(outcomes[TARGET].isna().sum()),
            "unresolved_rule": (
                "A fish never observed at 6 dpf is NA, not 0: an unobserved "
                "animal has an unknown outcome, not a negative one."
            ),
            "positive_by_group": {
                group: int(outcomes.loc[outcomes["group"] == group, TARGET].sum())
                for group in GROUPS
            },
            "unresolved_by_group": {
                group: int(
                    outcomes.loc[outcomes["group"] == group, TARGET].isna().sum()
                )
                for group in GROUPS
            },
        },
        "features": list(FEATURES),
        }
    (output_dir / "tbi_4_6dpf_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_normalized_dataset(
    lfp_path: Path | str = LFP_CSV,
    outcomes_path: Path | str = OUTCOMES_CSV,
    behavior_path: Path | str = BEHAVIOR_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the normalized tables."""
    return load_dataset(
        lfp_path,
        outcomes_path,
        behavior_path,
    )


# ===========================================================================
def run_full_analysis(
    output_dir: Path | str = RESULTS_DIR,
    seed: int = SEED,
    **kwargs,
) -> dict:
    """Normalize the source workbooks, then run the analysis on them."""
    lfp, outcomes, behavior = build_tables(
        kwargs.pop("lfp_workbook", DEFAULT_LFP_WORKBOOK),
        kwargs.pop("behavior_workbook", DEFAULT_BEHAVIOR_WORKBOOK),
    )
    validate_tables(lfp, outcomes, behavior)
    manifest = write_tables(lfp, outcomes, behavior)
    lfp, outcomes, behavior = load_normalized_dataset()
    return run_analysis(
        lfp,
        outcomes,
        behavior,
        output_dir=output_dir,
        seed=seed,
        endpoint_summary=manifest["endpoint"],
        **kwargs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfp-workbook", type=Path, default=DEFAULT_LFP_WORKBOOK)
    parser.add_argument(
        "--behavior-workbook", type=Path, default=DEFAULT_BEHAVIOR_WORKBOOK
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--states", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_full_analysis(
        output_dir=args.output_dir,
        seed=args.seed,
        lfp_workbook=args.lfp_workbook,
        behavior_workbook=args.behavior_workbook,
        test_fraction=args.test_fraction,
        candidates=args.states,
        restarts=args.restarts,
        cv_folds=args.cv_folds,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    early = metrics["early_prediction"]
    print(
        f"{metrics['dataset_qc']['n_fish']} fish; K="
        f"{metrics['selected_states']} microstates; held-out 6 dpf forecast "
        f"AUC={early['roc_auc']:.3f} on {early['n_test_fish']} fish "
        f"({early['n_positive']} positive). State recovery: not measurable."
    )


if __name__ == "__main__":
    main()
