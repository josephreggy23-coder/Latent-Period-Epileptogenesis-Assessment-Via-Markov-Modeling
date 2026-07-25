"""Initialize deterministic placeholder rows for the 4-6 dpf TBI template.

The generated values are synthetic editable placeholders, not animal measurements.
Locskai et al. motivate the blast-pressure insult and repeated-hit dose axis;
Eimon et al. motivate the LFP acquisition/QC and statistical summaries; and
DeepLabCut motivates the generated pose-style summary interface. The generated
behavior and LFP features share a planted latent state and are not independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    DATA_DIR,
    DLC_CSV,
    DOSE_INDEX,
    GROUPS,
    INJURY_DPF,
    LFP_CSV,
    OBSERVATION_DPF,
    OUTCOMES_CSV,
    PLACEHOLDER_STATUS,
    RECORD_STATUS,
    SEED,
    TARGET,
    TRUTH_STATE,
    validate_dataset,
)

LOCSKAI_URL = "https://doi.org/10.1242/bio.060601"
EIMON_URL = "https://doi.org/10.1038/s41467-017-02404-4"
DLC_URL = "https://doi.org/10.1038/s41593-018-0209-y"
DLC_PROTOCOL_URL = "https://doi.org/10.1038/s41596-019-0176-0"

GROUP_CONFIG = {
    # The apparatus is held fixed and only the repeated-hit count changes.
    # A nominal 195 kPa per-hit center is a template assumption within the
    # approximately 90-300 kPa seizure-associated range reported by Locskai.
    "sham": {
        "n_weight_drops": 0,
        "pressure_kpa": 0.0,
        "initial": np.array([0.94, 0.055, 0.005]),
        "transition": np.array(
            [[0.93, 0.065, 0.005], [0.60, 0.36, 0.04], [0.25, 0.50, 0.25]]
        ),
        "survival": 0.995,
    },
    "tbi_low": {
        "n_weight_drops": 3,
        "pressure_kpa": 195.0,
        "initial": np.array([0.66, 0.29, 0.05]),
        "transition": np.array(
            [[0.82, 0.16, 0.02], [0.38, 0.52, 0.10], [0.10, 0.42, 0.48]]
        ),
        "survival": 0.98,
    },
    "tbi_moderate": {
        "n_weight_drops": 5,
        "pressure_kpa": 195.0,
        "initial": np.array([0.36, 0.48, 0.16]),
        "transition": np.array(
            [[0.70, 0.25, 0.05], [0.22, 0.55, 0.23], [0.06, 0.30, 0.64]]
        ),
        "survival": 0.94,
    },
    "tbi_high": {
        "n_weight_drops": 7,
        "pressure_kpa": 195.0,
        "initial": np.array([0.18, 0.45, 0.37]),
        "transition": np.array(
            [[0.58, 0.32, 0.10], [0.12, 0.48, 0.40], [0.03, 0.17, 0.80]]
        ),
        "survival": 0.84,
    },
}

# Session-level LFP summary centers by planted state.  The five distributional
# summaries reproduce Eimon's feature families; event rate and single-channel
# ICA complexity are derived session summaries.
STATE_LFP = {
    0: {
        "mean_uv": 0.0,
        "variance_uv2": 600.0,
        "skewness": 0.05,
        "kurtosis": 3.2,
        "event_rate": 0.05,
        "ica_complexity": 0.88,
    },
    1: {
        "mean_uv": 10.0,
        "variance_uv2": 1_650.0,
        "skewness": 0.70,
        "kurtosis": 5.8,
        "event_rate": 0.90,
        "ica_complexity": 0.58,
    },
    2: {
        "mean_uv": 24.0,
        "variance_uv2": 4_800.0,
        "skewness": 1.45,
        "kurtosis": 10.5,
        "event_rate": 3.40,
        "ica_complexity": 0.28,
    },
}


def _adjust_probs_for_vulnerability(
    probabilities: np.ndarray,
    vulnerability_z: float,
    strength: float = 0.32,
) -> np.ndarray:
    severity = np.arange(len(probabilities), dtype=float)
    adjusted = probabilities * np.exp(strength * vulnerability_z * severity)
    return adjusted / adjusted.sum()


def _sample_lfp(state: int, fish_effect: float, rng: np.random.Generator) -> dict:
    base = STATE_LFP[state]
    variance = base["variance_uv2"] * np.exp(0.22 * fish_effect + rng.normal(0, 0.24))
    kurtosis = base["kurtosis"] * np.exp(0.08 * fish_effect + rng.normal(0, 0.14))
    fourth_power = kurtosis * variance**2 * np.exp(rng.normal(0, 0.10))
    event_count = rng.poisson(max(0.01, base["event_rate"] * np.exp(0.2 * fish_effect)) * 4.0)
    return {
        "lfp_mean_uv": round(float(rng.normal(base["mean_uv"] + 2.5 * fish_effect, 7.0)), 4),
        "lfp_variance_uv2": round(float(max(1.0, variance)), 4),
        "lfp_skewness": round(float(rng.normal(base["skewness"] + 0.08 * fish_effect, 0.28)), 4),
        "lfp_kurtosis": round(float(max(1.5, kurtosis)), 4),
        "lfp_fourth_power_mean_uv4": round(float(max(1.0, fourth_power)), 4),
        "lfp_seizure_event_rate_per_h": round(float(event_count / 4.0), 4),
        "lfp_ica_complexity": round(
            float(np.clip(rng.normal(base["ica_complexity"] - 0.025 * fish_effect, 0.065), 0.02, 1.0)),
            4,
        ),
    }


def _sample_behavior(
    state: int,
    cumulative_burden: float,
    dpf: int,
    rng: np.random.Generator,
) -> tuple[dict, int]:
    stunned_probability = 0.0
    if state == 2:
        stunned_probability = float(np.clip((cumulative_burden - 900.0) / 900.0, 0.05, 0.72))
    stunned = bool(rng.random() < stunned_probability)

    if state == 0:
        speed, rest, bursts, tail_bend, tail_change, whirlpool = 1.55, 0.27, 4.0, 14.0, 32.0, 0.02
    elif state == 1:
        speed, rest, bursts, tail_bend, tail_change, whirlpool = 3.05, 0.13, 9.0, 28.0, 72.0, 0.35
    elif stunned:
        speed, rest, bursts, tail_bend, tail_change, whirlpool = 0.48, 0.78, 2.0, 36.0, 41.0, 1.15
    else:
        speed, rest, bursts, tail_bend, tail_change, whirlpool = 4.25, 0.07, 15.0, 44.0, 118.0, 1.75

    speed = max(0.02, rng.normal(speed, 0.28))
    rest = float(np.clip(rng.normal(rest, 0.045), 0.0, 0.98))
    bursts = max(0.0, rng.normal(bursts, 1.3))
    tail_bend = max(0.5, rng.normal(tail_bend, 3.0))
    tail_change = max(0.5, rng.normal(tail_change, 8.0))
    whirlpool = max(0.0, rng.normal(whirlpool, 0.16))

    low_quality = rng.random() < 0.035
    likelihood = np.clip(rng.normal(0.78 if low_quality else 0.965, 0.025), 0.55, 0.995)
    pct_low = np.clip(rng.normal(18.0 if low_quality else 2.8, 2.2), 0.0, 40.0)
    tracking_qc = bool(likelihood >= 0.90 and pct_low <= 10.0)

    abnormality = np.mean(
        [
            abs(speed - 1.55) / 0.8,
            abs(rest - 0.27) / 0.14,
            abs(bursts - 4.0) / 3.0,
            abs(tail_bend - 14.0) / 10.0,
            abs(tail_change - 32.0) / 28.0,
            whirlpool / 0.5,
        ]
    )
    if state == 0:
        manual_stage = int(rng.choice([0, 1], p=[0.92, 0.08]))
    elif state == 1:
        manual_stage = int(rng.choice([0, 1, 2], p=[0.05, 0.72, 0.23]))
    else:
        manual_stage = int(rng.choice([1, 2, 3], p=[0.05, 0.53, 0.42]))

    behavior = {
        "dpf": dpf,
        "video_id": "",
        "dlc_model": "dlc_resnet50_placeholder",
        "dlc_keypoints": "head|midline_1|midline_2|midline_3|midline_4|tail_tip",
        "dlc_pcutoff": 0.90,
        "dlc_mean_keypoint_likelihood": round(float(likelihood), 4),
        "dlc_pct_frames_below_pcutoff": round(float(pct_low), 4),
        "dlc_tracking_qc_pass": tracking_qc,
        "dlc_mean_speed_mm_s": round(float(speed), 4),
        "dlc_max_speed_mm_s": round(float(max(speed, speed * rng.uniform(1.6, 2.8))), 4),
        "dlc_rest_fraction": round(float(rest), 4),
        "dlc_burst_rate_per_min": round(float(bursts), 4),
        "dlc_mean_tail_bend_deg": round(float(tail_bend), 4),
        "dlc_max_tail_bend_deg": round(float(tail_bend * rng.uniform(1.3, 2.0)), 4),
        "dlc_mean_tail_angle_change_deg_s": round(float(tail_change), 4),
        "dlc_max_tail_angle_change_deg_s": round(float(tail_change * rng.uniform(1.4, 2.3)), 4),
        "dlc_whirlpool_rate_per_min": round(float(whirlpool), 4),
        "dlc_behavior_abnormality_index": round(float(abnormality), 4),
        "manual_pts_stage_TRUTH": manual_stage,
        "stunned_phenotype_TRUTH": stunned,
    }
    return behavior, manual_stage


def generate_dataset(
    seed: int = SEED,
    n_per_arm: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if n_per_arm < 20:
        raise ValueError("n_per_arm must be at least 20 for stable grouped evaluation.")
    rng = np.random.default_rng(seed)
    lfp_rows: list[dict] = []
    outcome_rows: list[dict] = []
    behavior_rows: list[dict] = []
    fish_number = 0

    for group in GROUPS:
        config = GROUP_CONFIG[group]
        for _ in range(n_per_arm):
            fish_number += 1
            fish_id = f"TBI{fish_number:04d}"
            batch = int(rng.integers(1, 5))
            vulnerability = float(rng.normal())
            fish_effect = float(rng.normal(0, 0.55))

            survived_to_6 = bool(rng.random() < config["survival"])
            if survived_to_6:
                last_dpf = 6
            else:
                last_dpf = int(rng.choice([4, 5], p=[0.30, 0.70]))

            if group == "sham":
                measured_pressure = 0.0
            else:
                measured_pressure = float(max(60.0, rng.normal(config["pressure_kpa"], 18.0)))
            # Placeholder exposure index (peak kPa x hit count), not a measured
            # pressure integral. Its unit is therefore kPa-hits.
            cumulative_burden = measured_pressure * config["n_weight_drops"]

            initial = _adjust_probs_for_vulnerability(config["initial"], vulnerability)
            state = int(rng.choice(3, p=initial))
            states: dict[int, int] = {}
            first_state2_dpf: int | None = None
            max_manual_stage = 0

            for dpf in OBSERVATION_DPF:
                if dpf > last_dpf:
                    break
                if dpf > OBSERVATION_DPF[0]:
                    transition_row = _adjust_probs_for_vulnerability(
                        config["transition"][state],
                        vulnerability,
                        strength=0.24,
                    )
                    state = int(rng.choice(3, p=transition_row))
                states[dpf] = state
                if state == 2 and first_state2_dpf is None:
                    first_state2_dpf = dpf

                noisy_session = rng.random() < 0.035
                resistance_change = float(
                    rng.uniform(55.0, 90.0) if noisy_session and rng.random() < 0.65
                    else np.clip(rng.gamma(2.2, 5.5), 0.0, 49.5)
                )
                rms_noise = float(
                    rng.uniform(0.205, 0.34) if noisy_session and resistance_change <= 50.0
                    else np.clip(rng.lognormal(np.log(0.075), 0.27), 0.02, 0.195)
                )
                qc_pass = bool(resistance_change <= 50.0 and rms_noise < 0.2)
                lfp = _sample_lfp(state, fish_effect, rng)
                lfp_rows.append(
                    {
                        "fish_id": fish_id,
                        "group": group,
                        "batch": batch,
                        "tbi_dpf": INJURY_DPF,
                        "dpf": dpf,
                        "days_post_tbi": dpf - INJURY_DPF,
                        "syringe_ml": 10,
                        "holder": "three_prong_clamp",
                        "weight_g": 0 if group == "sham" else 200,
                        "drop_height_cm": 0 if group == "sham" else 108,
                        "n_weight_drops": config["n_weight_drops"],
                        "inter_drop_interval_lt_10_min": group != "sham",
                        "nominal_peak_pressure_kpa": config["pressure_kpa"],
                        "measured_peak_pressure_kpa": round(measured_pressure, 4),
                        DOSE_INDEX: round(cumulative_burden, 4),
                        "recording_duration_min": 240,
                        "sampling_rate_hz": 3000,
                        "anti_alias_lowpass_hz": 3000,
                        "bandpass_low_hz": 0.5,
                        "bandpass_high_hz": 1000,
                        "window_length_s": 30,
                        "window_overlap_s": 20,
                        "electrode_impedance_mohm": round(
                            float(np.clip(rng.normal(3.0, 0.28), 1.8, 4.2)), 4
                        ),
                        "rms_noise_mv": round(rms_noise, 4),
                        "electrode_resistance_change_pct": round(
                            resistance_change, 4
                        ),
                        "qc_pass": qc_pass,
                        **lfp,
                        TRUTH_STATE: state,
                        RECORD_STATUS: PLACEHOLDER_STATUS,
                        "template_seed": seed,
                    }
                )

                behavior, manual_stage = _sample_behavior(
                    state, cumulative_burden, dpf, rng
                )
                behavior["fish_id"] = fish_id
                behavior["group"] = group
                behavior["video_id"] = f"VID_{fish_id}_{dpf}dpf"
                behavior[RECORD_STATUS] = PLACEHOLDER_STATUS
                behavior["template_seed"] = seed
                behavior_rows.append(behavior)
                max_manual_stage = max(max_manual_stage, manual_stage)

            state_at_6 = states.get(6)
            high_burden_endpoint = (
                int(state_at_6 == 2) if state_at_6 is not None else np.nan
            )
            outcome_rows.append(
                {
                    "fish_id": fish_id,
                    "group": group,
                    "batch": batch,
                    "n_weight_drops": config["n_weight_drops"],
                    "measured_peak_pressure_kpa": round(measured_pressure, 4),
                    DOSE_INDEX: round(cumulative_burden, 4),
                    "last_observed_dpf": last_dpf,
                    "survived_to_6dpf": survived_to_6,
                    TARGET: high_burden_endpoint,
                    "first_high_state_dpf_TRUTH": first_state2_dpf,
                    "max_hidden_state_TRUTH": max(states.values()),
                    "max_manual_pts_stage_TRUTH": max_manual_stage,
                    "susceptibility_z_TRUTH": round(vulnerability, 4),
                    RECORD_STATUS: PLACEHOLDER_STATUS,
                    "template_seed": seed,
                }
            )

    lfp = pd.DataFrame(lfp_rows).sort_values(["fish_id", "dpf"]).reset_index(drop=True)
    outcomes = pd.DataFrame(outcome_rows).sort_values("fish_id").reset_index(drop=True)
    behavior_columns = [
        "fish_id",
        "group",
        "dpf",
        "video_id",
        "dlc_model",
        "dlc_keypoints",
        "dlc_pcutoff",
        "dlc_mean_keypoint_likelihood",
        "dlc_pct_frames_below_pcutoff",
        "dlc_tracking_qc_pass",
        "dlc_mean_speed_mm_s",
        "dlc_max_speed_mm_s",
        "dlc_rest_fraction",
        "dlc_burst_rate_per_min",
        "dlc_mean_tail_bend_deg",
        "dlc_max_tail_bend_deg",
        "dlc_mean_tail_angle_change_deg_s",
        "dlc_max_tail_angle_change_deg_s",
        "dlc_whirlpool_rate_per_min",
        "dlc_behavior_abnormality_index",
        "manual_pts_stage_TRUTH",
        "stunned_phenotype_TRUTH",
        RECORD_STATUS,
        "template_seed",
    ]
    behavior = (
        pd.DataFrame(behavior_rows)[behavior_columns]
        .sort_values(["fish_id", "dpf"])
        .reset_index(drop=True)
    )
    validate_dataset(lfp, outcomes, behavior)

    manifest = {
        "dataset_id": "larval_zebrafish_tbi_4_6dpf_template_v1",
        "data_status": PLACEHOLDER_STATUS,
        "template_seed": seed,
        "n_per_arm": n_per_arm,
        "n_fish": int(len(outcomes)),
        "n_lfp_sessions": int(len(lfp)),
        "injury_dpf": INJURY_DPF,
        "observation_dpf": list(OBSERVATION_DPF),
        "groups": list(GROUPS),
        "endpoint_definition": (
            f"{TARGET}=1 only when the planted state at 6 dpf is the "
            "high-burden state (state 2); prior high-state membership is not "
            "required."
        ),
        "dose_index_definition": (
            f"{DOSE_INDEX}=placeholder peak kPa multiplied by hit "
            "count; units are kPa-hits and it is not a measured pressure "
            "integral."
        ),
        "source_urls": [LOCSKAI_URL, EIMON_URL, DLC_URL, DLC_PROTOCOL_URL],
        "replacement_notice": (
            "The committed rows are deterministically generated synthetic "
            "placeholders. The cited papers do not report repeated daily LFP from "
            "the same TBI larvae at 4-6 dpf. Do not mark a row analysis_ready until "
            "its values have been replaced with measured observations."
        ),
    }
    return lfp, outcomes, behavior, manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(
    output_dir: Path | str = DATA_DIR,
    seed: int = SEED,
    n_per_arm: int = 60,
    *,
    force: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    paths = {
        "lfp_timeseries": output_dir / LFP_CSV.name,
        "fish_outcomes": output_dir / OUTCOMES_CSV.name,
        "dlc_behavior": output_dir / DLC_CSV.name,
    }
    manifest_path = output_dir / "tbi_4_6dpf_dataset_manifest.json"
    existing = [path for path in [*paths.values(), manifest_path] if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing template files: {names}. "
            "Pass force=True or --force only after backing up measured records."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    lfp, outcomes, behavior, manifest = generate_dataset(seed=seed, n_per_arm=n_per_arm)
    lfp.to_csv(paths["lfp_timeseries"], index=False, lineterminator="\n")
    outcomes.to_csv(paths["fish_outcomes"], index=False, lineterminator="\n")
    behavior.to_csv(paths["dlc_behavior"], index=False, lineterminator="\n")
    manifest["files"] = {
        key: {"path": path.name, "sha256": _sha256(path)}
        for key, path in paths.items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-per-arm", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing template after you have backed it up.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_dataset(
        args.output_dir,
        seed=args.seed,
        n_per_arm=args.n_per_arm,
        force=args.force,
    )
    print(
        f"Wrote {manifest['n_fish']} fish and {manifest['n_lfp_sessions']} "
        f"LFP sessions to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
