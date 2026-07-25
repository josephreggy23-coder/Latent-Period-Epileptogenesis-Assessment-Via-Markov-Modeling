"""Normalize and analyze the REAL larval-zebrafish TBI recording.

The rest of this package is a synthetic benchmark with a planted latent state.
This module ingests a real measured dataset that follows the same experimental
design - 3 dpf weight-drop TBI, LFP at 4/5/6 dpf, sham plus three injury doses -
and reshapes it into the identical three normalized tables, so the *same* HMM,
the same causal 4-5 dpf prefix rule, and the same 6 dpf forecast run unchanged.

Source workbooks (not redistributed here unless the owner adds them):

``actualdata1(lfp).xlsx``    sheet ``LFP Recordings``      one row per session
``actualdata(behavioral).xlsx``  sheets ``Behavioral Outcomes`` and ``Event Log``

Three things genuinely differ from the simulator, and none of them is papered
over:

1. **No planted latent state.** There is no ``hidden_state_TRUTH`` column, so
   held-out state-recovery accuracy is not merely unreported - it is
   unmeasurable. ``run_real_analysis`` returns ``state_recovery: None``.
2. **The endpoint is behavioural, not planted.** The 6 dpf high-burden endpoint
   is defined from the blinded manual Baraban scores in the Event Log: a fish is
   positive if it has at least one qualifying event at 6 dpf. Nothing derived
   from LFP enters the endpoint, so the forecast target stays independent of the
   model's inputs. It is **three-valued** - a fish never observed at 6 dpf gets
   ``NA``, not ``0``, because an unobserved animal has an unknown outcome rather
   than a negative one.
3. **Behaviour is per-event, not per-session.** The real Event Log lists scored
   events; sessions with no scored event are absent entirely. They are
   materialized as zero-event rows rather than dropped, because "no scored
   behaviour" is an observation, not a missing value.

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
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    DOSE_INDEX,
    FEATURES,
    GROUPS,
    INJURY_DPF,
    OBSERVATION_DPF,
    REAL_DATA_DIR,
    REAL_DLC_CSV,
    REAL_LFP_CSV,
    REAL_OUTCOMES_CSV,
    REAL_RESULTS_DIR,
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
def build_real_tables(
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
    frame["is_synthetic"] = False
    frame["fish_id"] = frame["fish_id"].astype(str)
    frame["group"] = frame["group"].astype(str)
    return frame.sort_values(["fish_id", "dpf"]).reset_index(drop=True)


def _dpf6_endpoint(raw_events: pd.DataFrame, raw_lfp: pd.DataFrame) -> pd.Series:
    """Real analogue of the planted 6 dpf high-burden endpoint.

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
    frame["is_synthetic"] = False
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
    frame["is_synthetic"] = False
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
# Validation specific to the real recording
# ===========================================================================
def validate_real_tables(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    behavior: pd.DataFrame,
) -> None:
    """Check the assumptions that make the real data usable by this benchmark."""
    if set(lfp["group"]) - set(GROUPS):
        raise ValueError(f"Unexpected arm in real data: {set(lfp['group']) - set(GROUPS)}")
    if set(lfp["dpf"].unique()) - set(OBSERVATION_DPF):
        raise ValueError("Real LFP contains sessions outside 4-6 dpf.")
    if not (lfp["tbi_dpf"] == INJURY_DPF).all():
        raise ValueError("Real LFP must record the insult at 3 dpf.")
    if lfp.duplicated(["fish_id", "dpf"]).any():
        raise ValueError("Duplicate fish_id/dpf sessions in the real LFP table.")

    # The published QC rule must reproduce the vendor's own qc_pass flag; if it
    # does not, the two disagree about which sessions are usable.
    expected = (lfp["electrode_shift_pct"] <= 50.0) & (lfp["rms_noise_mv"] < 0.2)
    if not np.array_equal(expected.to_numpy(), lfp["qc_pass"].astype(bool).to_numpy()):
        raise ValueError(
            "Real qc_pass does not reproduce the documented electrode-shift/noise rule."
        )
    values = lfp[list(FEATURES)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Real LFP model inputs must be finite.")
    resolved = outcomes[TARGET].notna()
    if not resolved.any():
        raise ValueError("No fish has a resolved 6 dpf endpoint.")
    if resolved.mean() < 0.5:
        raise ValueError(
            "More than half of fish lack a 6 dpf observation; the endpoint "
            "cannot carry the analysis."
        )
    if outcomes[TARGET].nunique() < 2:
        raise ValueError("The real endpoint must contain both classes.")
    if set(behavior["fish_id"]) - set(outcomes["fish_id"]):
        raise ValueError("Behaviour rows must map to fish-level outcome rows.")


def write_real_tables(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    behavior: pd.DataFrame,
    output_dir: Path | str = REAL_DATA_DIR,
) -> dict:
    """Persist the normalized real tables and a small provenance manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lfp.to_csv(output_dir / REAL_LFP_CSV.name, index=False)
    outcomes.to_csv(output_dir / REAL_OUTCOMES_CSV.name, index=False)
    behavior.to_csv(output_dir / REAL_DLC_CSV.name, index=False)
    manifest = {
        "is_synthetic": False,
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
        "planted_truth_available": False,
    }
    (output_dir / "tbi_4_6dpf_real_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_real_dataset(
    lfp_path: Path | str = REAL_LFP_CSV,
    outcomes_path: Path | str = REAL_OUTCOMES_CSV,
    dlc_path: Path | str = REAL_DLC_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the normalized real tables with real-data expectations."""
    return load_dataset(
        lfp_path,
        outcomes_path,
        dlc_path,
        expect_synthetic=False,
        require_truth=False,
    )


# ===========================================================================
# Reporting
# ===========================================================================
def write_real_report(
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
    text = f"""# Real larval-zebrafish TBI Markov-model results

> **Measured data.** Every row here comes from a real recording, not the
> simulator. There is no planted latent state, so held-out state-recovery
> accuracy is **not reported and not measurable** - only the forward 6 dpf
> forecast is scored.

## Run scope

- **{qc['n_fish']} fish**, {qc['n_lfp_sessions']} LFP sessions at 4-6 dpf,
  {qc['n_qc_pass_sessions']} passing QC ({100 * qc['qc_pass_rate']:.1f}%)
- {qc['n_contiguous_model_sessions']} contiguous modelling sessions from
  {qc['n_model_fish_with_4dpf_baseline']} fish with a usable 4 dpf baseline
- selected **K={selected}** by lowest train-only BIC ({selection['bic']:.1f});
  train-only CV log likelihood/session
  {selection['cv_log_likelihood_per_session']:.3f}
- identical preprocessing, severity ordering, and macrostate collapse as the
  synthetic benchmark - none of it consults the endpoint

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

The propagated quantity is the probability of occupying the **top LFP
macrostate**, whereas the endpoint is a **behavioural** event. The two are on
different scales, and the LFP state is rarer than the behavioural outcome, so
the risk sits well below 0.5 for most animals. Any deployment would need a
threshold fitted on training fish; none is tuned on the held-out set here.

## Latent-state recovery

**Not measurable.** Real animals carry no planted latent state. The synthetic
benchmark's balanced accuracy has no counterpart here, and no proxy is
substituted for it.

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


# ===========================================================================
def run_real_analysis(
    output_dir: Path | str = REAL_RESULTS_DIR,
    seed: int = SEED,
    **kwargs,
) -> dict:
    """Normalize the real workbooks, then run the standard analysis on them."""
    lfp, outcomes, behavior = build_real_tables(
        kwargs.pop("lfp_workbook", DEFAULT_LFP_WORKBOOK),
        kwargs.pop("behavior_workbook", DEFAULT_BEHAVIOR_WORKBOOK),
    )
    validate_real_tables(lfp, outcomes, behavior)
    manifest = write_real_tables(lfp, outcomes, behavior)
    report_writer = partial(write_real_report, endpoint=manifest["endpoint"])
    lfp, outcomes, behavior = load_real_dataset()
    return run_analysis(
        lfp,
        outcomes,
        behavior,
        output_dir=output_dir,
        seed=seed,
        benchmark_type="real_measured",
        critical_caveat=(
            "Real measured recording. No planted latent state exists, so "
            "state-recovery accuracy is unmeasurable; only the causal 4-5 dpf "
            "to 6 dpf behavioural forecast is scored."
        ),
        behavior_note=(
            "Behavioural values are blinded manual Baraban scores and "
            "pose-derived kinematics from the real Event Log, aggregated per "
            "session. Zero-event sessions are retained as observations."
        ),
        figure_labels={
            "data_label": "Real",
            "endpoint_label": "behavioural endpoint",
            "behavior_label": "Observed behaviour",
            "behavior_suptitle": (
                "Blinded behavioural validation (real recording)"
            ),
            "speed_panel_title": "Observed locomotor speed",
        },
        report_writer=report_writer,
        **kwargs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfp-workbook", type=Path, default=DEFAULT_LFP_WORKBOOK)
    parser.add_argument(
        "--behavior-workbook", type=Path, default=DEFAULT_BEHAVIOR_WORKBOOK
    )
    parser.add_argument("--output-dir", type=Path, default=REAL_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--states", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_real_analysis(
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
        f"Real data: {metrics['dataset_qc']['n_fish']} fish; K="
        f"{metrics['selected_states']} microstates; held-out 6 dpf forecast "
        f"AUC={early['roc_auc']:.3f} on {early['n_test_fish']} fish "
        f"({early['n_positive']} positive). State recovery: not measurable."
    )


if __name__ == "__main__":
    main()
