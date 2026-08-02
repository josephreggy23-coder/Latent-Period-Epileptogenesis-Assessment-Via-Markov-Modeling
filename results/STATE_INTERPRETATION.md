# State interpretation

Named once, here, so that no later report refers to a bare state index. Every
reference elsewhere in this repository to a fitted state pairs the index with
one of these names.

## State 0 — Baseline / low-amplitude

Standardized (robust-scaled) means: variance -0.51,
kurtosis -0.44, seizure discharge rate
0.00, fourth-power mean
-0.49 — the lowest of the three states
on every feature, with a discharge rate indistinguishable from zero. This is
the electrophysiological resting repertoire: low-amplitude, low-kurtosis LFP
with no scored epileptiform events, and it is where the model starts most
fish (start probability 57%) at 4 dpf.

**6 dpf occupancy by arm:** sham 83%, tbi_high 0%, tbi_low 29%, tbi_moderate 19%.

**Where this does not fit the canonical progression.** The canonical
three-stage picture pairs a low-activity state with *acute post-injury
depression* — a transient suppression specific to the immediately
post-traumatic brain. This dataset cannot support that specific reading:
sham (uninjured) fish occupy state 0 (Baseline / low-amplitude) essentially as often as the
lowest-dose arm, and there is no un-injured control condition against which
a *depressed* low-amplitude state could be distinguished from an *ordinarily
quiescent* one. State 0 (Baseline / low-amplitude) is therefore named and described as the
baseline/resting repertoire, not as evidence of an acute depression phase.

## State 1 — Transitional / latent-like

Standardized means: variance 0.26, kurtosis
0.35, seizure discharge rate
0.00, fourth-power mean
0.31. Every amplitude-distribution
feature sits at an intermediate level, but the discharge rate is still
essentially zero, matching state 0 (Baseline / low-amplitude) rather than departing from it. This
is a state with a shifted LFP amplitude distribution and no overt
epileptiform events: an electrographic change without a clinical correlate,
which is the closest this feature set comes to the *latent period* concept —
silent circuit reorganization that precedes, rather than announces itself
as, overt seizure activity.

**6 dpf occupancy by arm:** sham 11%, tbi_high 22%, tbi_low 41%, tbi_moderate 56%.

## State 2 — Hyperexcitable / ictal-like

Standardized means: variance 0.83, kurtosis
0.86, seizure discharge rate
1.29, fourth-power mean
0.88 — the only state where the
discharge-rate feature departs meaningfully from zero, on top of the highest
variance, kurtosis, and fourth-power mean of the three states. This is the
state most consistent with overt epileptiform activity: large-amplitude,
heavy-tailed LFP with a measurably elevated rate of scored seizure-like
events. Once entered, it is largely self-sustaining: P(stay in state 2,
Hyperexcitable / ictal-like) = 0.60, P(drop only to state 1,
Transitional / latent-like) = 0.39, P(full recovery to state 0,
Baseline / low-amplitude) = 0.01 — matching the expectation
that a hyperexcitable regime, once reached, rarely resolves back to baseline
within a three-session window.

**6 dpf occupancy by arm:** sham 6%, tbi_high 78%, tbi_low 29%, tbi_moderate 25%.

## Fit to the canonical progression, stated plainly

The recovered three-state structure lines up with the *latent -> hyperexcitable*
half of the canonical acute-depression / latent / hyperexcitable progression:
state 1 (Transitional / latent-like) behaves like a silent, sub-discharge transitional state, and
state 2 (Hyperexcitable / ictal-like) behaves like an overt hyperexcitable state whose occupancy
rises steeply and monotonically with injury dose (see
`docs/PREREGISTRATION.md` primary result). It does **not** resolve a distinct
acute-depression stage: state 0 (Baseline / low-amplitude) is shared by sham and injured fish
alike and is better read as the ordinary resting repertoire. A
three-microstate HMM fit to three summary-statistic features per session,
with at most two transitions observed per fish, is not positioned to
separate "acutely suppressed" from "never excited" without an
uninjured-versus-injured amplitude contrast the current feature set does not
carry.

