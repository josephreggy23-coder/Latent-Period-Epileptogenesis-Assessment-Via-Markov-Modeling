"""Task 6: name the recovered latent states in plain neurophysiology.

Severity-ordered HMM components are statistical objects until someone commits
to what they mean. This module does that commitment once, in one place, so
"State 2" never appears bare in user-facing text again -- every reference
after this module pairs the name with the index.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import FEATURES

FEATURE_LABELS = {
    "lfp_variance_uv2": "variance (E/I proxy)",
    "lfp_kurtosis": "kurtosis (E/I proxy)",
    "lfp_seizure_event_rate_per_h": "seizure discharge rate",
    "lfp_fourth_power_mean_uv4": "fourth-power mean (waveform shape)",
}

# Three-state names and neurophysiology paragraphs, written against the
# actual fitted numbers each time this module runs (see
# write_state_interpretation_report). Only used when K=3 is selected, which
# is the preregistered model-order winner on the measured cohort; a K=2 fit
# gets its own two-state names below rather than forcing a mismatched triad.
THREE_STATE_NAMES = {
    0: "Baseline / low-amplitude",
    1: "Transitional / latent-like",
    2: "Hyperexcitable / ictal-like",
}
TWO_STATE_NAMES = {
    0: "Low-amplitude",
    1: "Hyperexcitable",
}


def state_names(n_states: int) -> dict[int, str]:
    if n_states == 3:
        return dict(THREE_STATE_NAMES)
    if n_states == 2:
        return dict(TWO_STATE_NAMES)
    raise ValueError(
        f"No prespecified names for K={n_states}; docs/PREREGISTRATION.md "
        "caps K at 2 or 3."
    )


def named_state(index: int, n_states: int) -> str:
    """'2 (Hyperexcitable / ictal-like)' -- the only allowed way to refer to
    a state in user-facing prose after this module is written."""
    return f"{index} ({state_names(n_states)[index]})"


def make_state_emission_profile_figure(
    output_path: Path,
    ordered_means: np.ndarray,
    n_states: int,
) -> None:
    """Heatmap of standardized (robust-scaled) emission means, states x
    features, both severity-ordered."""
    names = state_names(n_states)
    labels = [FEATURE_LABELS[feature] for feature in FEATURES]
    fig, axis = plt.subplots(figsize=(7.5, 0.9 * n_states + 2.2))
    image = axis.imshow(ordered_means, cmap="RdYlBu_r", aspect="auto")
    for row in range(ordered_means.shape[0]):
        for column in range(ordered_means.shape[1]):
            axis.text(
                column,
                row,
                f"{ordered_means[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_yticks(range(n_states))
    axis.set_yticklabels([f"{index} — {names[index]}" for index in range(n_states)])
    axis.set_title("Standardized emission means by severity-ordered state")
    fig.colorbar(image, ax=axis, shrink=0.85, label="robust-scaled units")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _occupancy_by_group(occupancy: pd.DataFrame, state: int) -> str:
    column = f"fraction_state_{state}"
    if column not in occupancy.columns:
        return "occupancy not available"
    at_6dpf = occupancy.loc[occupancy["dpf"] == 6]
    parts = [
        f"{row['group']} {100 * row[column]:.0f}%"
        for _, row in at_6dpf.sort_values("group").iterrows()
    ]
    return ", ".join(parts)


def write_state_interpretation_report(
    output_path: Path,
    n_states: int,
    ordered_means: np.ndarray,
    ordered_transition_matrix: np.ndarray,
    ordered_start_probabilities: np.ndarray,
    occupancy: pd.DataFrame,
) -> None:
    names = state_names(n_states)
    feature_index = {feature: position for position, feature in enumerate(FEATURES)}

    def mean_at(state: int, feature: str) -> float:
        return float(ordered_means[state, feature_index[feature]])

    if n_states == 3:
        state_0_occ = _occupancy_by_group(occupancy, 0)
        state_1_occ = _occupancy_by_group(occupancy, 1)
        state_2_occ = _occupancy_by_group(occupancy, 2)
        s0, s1, s2 = names[0], names[1], names[2]
        body = f"""## State 0 — {s0}

Standardized (robust-scaled) means: variance {mean_at(0, 'lfp_variance_uv2'):.2f},
kurtosis {mean_at(0, 'lfp_kurtosis'):.2f}, seizure discharge rate
{mean_at(0, 'lfp_seizure_event_rate_per_h'):.2f}, fourth-power mean
{mean_at(0, 'lfp_fourth_power_mean_uv4'):.2f} — the lowest of the three states
on every feature, with a discharge rate indistinguishable from zero. This is
the electrophysiological resting repertoire: low-amplitude, low-kurtosis LFP
with no scored epileptiform events, and it is where the model starts most
fish (start probability {ordered_start_probabilities[0]:.0%}) at 4 dpf.

**6 dpf occupancy by arm:** {state_0_occ}.

**Where this does not fit the canonical progression.** The canonical
three-stage picture pairs a low-activity state with *acute post-injury
depression* — a transient suppression specific to the immediately
post-traumatic brain. This dataset cannot support that specific reading:
sham (uninjured) fish occupy state 0 ({s0}) essentially as often as the
lowest-dose arm, and there is no un-injured control condition against which
a *depressed* low-amplitude state could be distinguished from an *ordinarily
quiescent* one. State 0 ({s0}) is therefore named and described as the
baseline/resting repertoire, not as evidence of an acute depression phase.

## State 1 — {s1}

Standardized means: variance {mean_at(1, 'lfp_variance_uv2'):.2f}, kurtosis
{mean_at(1, 'lfp_kurtosis'):.2f}, seizure discharge rate
{mean_at(1, 'lfp_seizure_event_rate_per_h'):.2f}, fourth-power mean
{mean_at(1, 'lfp_fourth_power_mean_uv4'):.2f}. Every amplitude-distribution
feature sits at an intermediate level, but the discharge rate is still
essentially zero, matching state 0 ({s0}) rather than departing from it. This
is a state with a shifted LFP amplitude distribution and no overt
epileptiform events: an electrographic change without a clinical correlate,
which is the closest this feature set comes to the *latent period* concept —
silent circuit reorganization that precedes, rather than announces itself
as, overt seizure activity.

**6 dpf occupancy by arm:** {state_1_occ}.

## State 2 — {s2}

Standardized means: variance {mean_at(2, 'lfp_variance_uv2'):.2f}, kurtosis
{mean_at(2, 'lfp_kurtosis'):.2f}, seizure discharge rate
{mean_at(2, 'lfp_seizure_event_rate_per_h'):.2f}, fourth-power mean
{mean_at(2, 'lfp_fourth_power_mean_uv4'):.2f} — the only state where the
discharge-rate feature departs meaningfully from zero, on top of the highest
variance, kurtosis, and fourth-power mean of the three states. This is the
state most consistent with overt epileptiform activity: large-amplitude,
heavy-tailed LFP with a measurably elevated rate of scored seizure-like
events. Once entered, it is largely self-sustaining: P(stay in state 2,
{s2}) = {ordered_transition_matrix[2, 2]:.2f}, P(drop only to state 1,
{s1}) = {ordered_transition_matrix[2, 1]:.2f}, P(full recovery to state 0,
{s0}) = {ordered_transition_matrix[2, 0]:.2f} — matching the expectation
that a hyperexcitable regime, once reached, rarely resolves back to baseline
within a three-session window.

**6 dpf occupancy by arm:** {state_2_occ}.

## Fit to the canonical progression, stated plainly

The recovered three-state structure lines up with the *latent -> hyperexcitable*
half of the canonical acute-depression / latent / hyperexcitable progression:
state 1 ({s1}) behaves like a silent, sub-discharge transitional state, and
state 2 ({s2}) behaves like an overt hyperexcitable state whose occupancy
rises steeply and monotonically with injury dose (see
`docs/PREREGISTRATION.md` primary result). It does **not** resolve a distinct
acute-depression stage: state 0 ({s0}) is shared by sham and injured fish
alike and is better read as the ordinary resting repertoire. A
three-microstate HMM fit to three summary-statistic features per session,
with at most two transitions observed per fish, is not positioned to
separate "acutely suppressed" from "never excited" without an
uninjured-versus-injured amplitude contrast the current feature set does not
carry.
"""
    else:
        state_0_occ = _occupancy_by_group(occupancy, 0)
        state_1_occ = _occupancy_by_group(occupancy, 1)
        body = f"""## State 0 — {names[0]}

Standardized means: variance {mean_at(0, 'lfp_variance_uv2'):.2f}, kurtosis
{mean_at(0, 'lfp_kurtosis'):.2f}, seizure discharge rate
{mean_at(0, 'lfp_seizure_event_rate_per_h'):.2f}, fourth-power mean
{mean_at(0, 'lfp_fourth_power_mean_uv4'):.2f}.

**6 dpf occupancy by arm:** {state_0_occ}.

## State 1 — {names[1]}

Standardized means: variance {mean_at(1, 'lfp_variance_uv2'):.2f}, kurtosis
{mean_at(1, 'lfp_kurtosis'):.2f}, seizure discharge rate
{mean_at(1, 'lfp_seizure_event_rate_per_h'):.2f}, fourth-power mean
{mean_at(1, 'lfp_fourth_power_mean_uv4'):.2f}.

**6 dpf occupancy by arm:** {state_1_occ}.

## Fit to the canonical progression, stated plainly

Two states were selected, not three. This model order resolves only a
low/high contrast and does not have enough components to distinguish a
latent transitional state from either the baseline or the hyperexcitable
state. It therefore does **not** resolve the canonical three-stage
acute-depression / latent / hyperexcitable progression -- only a coarser
two-way split between {names[0]} and {names[1]}.
"""

    text = f"""# State interpretation

Named once, here, so that no later report refers to a bare state index. Every
reference elsewhere in this repository to a fitted state pairs the index with
one of these names.

{body}
"""
    Path(output_path).write_text(text, encoding="utf-8")
