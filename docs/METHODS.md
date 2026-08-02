# Methods

## Scope

This project joins three methodological interfaces:

1. a larval-zebrafish weight-drop TBI insult;
2. forebrain LFP acquisition and statistical pattern summaries;
3. markerless pose-derived behavioral validation.

The wet-lab protocol, including the constraints this dataset cannot satisfy, is
in [EXPERIMENTAL_PROTOCOL.md](EXPERIMENTAL_PROTOCOL.md). The source papers do
not report repeated same-fish LFP at 4–6 dpf after a 3 dpf TBI; that
combination is a new integration requiring a pilot.

The feature allowlist, model-order range, primary outcome (dose ordering of
latent states), secondary outcome (6 dpf behavioral forecast), and split rule
below were frozen in [PREREGISTRATION.md](PREREGISTRATION.md) before any
refitting. This document describes how each of those pieces is computed;
PREREGISTRATION.md is the authoritative record of what was committed to in
advance.

## Cohort

240 larvae, 60 per arm: sham, 100 g, 200 g, and 300 g single weight drops from
108 cm through a 20 mL syringe held in a three-prong clamp. Measured peak
pressures are 0, 115, 210, and 319 kPa. TBI occurs at 3 dpf; LFP and behavior
sessions occur at 4, 5, and 6 dpf across 6 clutches and 3 recording batches.

`cumulative_pressure_burden_kpa_hits` is a kPa-hits dose index, not a measured
pressure integral.

## LFP acquisition interface

Following Eimon et al.: penetrating forebrain electrode, 1 M chloride fill,
Ag/AgCl wire, advancement stopped near 3 MΩ with RMS noise below 0.2 mV,
3 kSamples/s with a 3 kHz anti-alias filter, and an offline 0.5–1000 Hz
band-pass. Sessions are summarized into seven statistics.

Prespecified QC excludes a session when `electrode_shift_pct` exceeds 50 or
`rms_noise_mv` reaches 0.2. Validation asserts that the recorded `qc_pass` flag
reproduces this rule exactly; disagreement is a hard error, since the two would
otherwise disagree about which sessions are usable.

## Behavioral validation

Pose-derived behavior is scored per event by a blinded reviewer using
Baraban/Locskai stages, then aggregated per fish-session. It is an independent
validation channel and never an HMM input.

For a reduced-camera pilot, the protocol permits a prespecified central C5:F8
block containing 16 monitored fish—four from each injury condition—with a
single top-down global-shutter camera. Treatment positions are balanced within
the block and rotated between plates/clutches; fish are assigned before any
behavioral outcome is known. Fixed 256 × 256 px well crops are calibrated at
the well plane and analyzed independently with DeepLabCut. Fish outside the
recorded region are `NA` for the video-derived endpoint unless another complete,
prespecified video acquisition exists; they are never counted as behavioral
negatives. Camera, illumination, bandwidth, and validation requirements are in
[EXPERIMENTAL_PROTOCOL.md §3–4](EXPERIMENTAL_PROTOCOL.md#reduced-camera-pilot-layout-proposed).

Sessions with no scored event are materialized with zero event rates rather than
dropped: "no scored behavior" is an observation, and omitting them would
restrict the validation to the abnormal subset and bias it.

The abnormality index combines event-rate and stage terms only — all of which
remain defined at zero events — each divided by a label-free upper quantile of
its own distribution so the terms are commensurate. The endpoint is never
consulted. Kinematic columns (speed, tail angle) are reported but excluded from
the index, because they are undefined without an event and imputing them would
manufacture signal.

## Preprocessing

The HMM uses three prespecified feature concepts, four columns:
`lfp_variance_uv2` and `lfp_kurtosis` together (excitation-inhibition proxy —
the intended proxy, the aperiodic 1/f spectral exponent, is not computable
from this dataset, which contains only pre-summarized session statistics and
no raw or windowed trace); `lfp_seizure_event_rate_per_h` (epileptiform
discharge rate); and `lfp_fourth_power_mean_uv4` (waveform-shape measure —
the intended measure, line length, is unavailable for the same reason). All
four columns receive `log1p`, then median/interquartile-range scaling
estimated from training fish only. See
[PREREGISTRATION.md](PREREGISTRATION.md) for the full substitution rationale.

After QC, each fish contributes the uninterrupted daily prefix beginning at
4 dpf. A missing 4 dpf session excludes that fish from sequence modeling; a
later gap terminates the sequence.

## Hidden Markov model

The implementation is a diagonal-Gaussian HMM fit by log-space Baum–Welch
expectation maximization with:

- seeded restarts;
- start and transition pseudocounts;
- variance regularization;
- full transition matrices permitting worsening and recovery;
- variable-length sequences;
- Viterbi decoding and causal forward filtering.

Two- and three-state candidates are compared on training-only BIC and
three-fold fish-level cross-validated log likelihood (K = 4 was dropped from
the candidate grid when the feature set was reduced to four columns, to keep
the parameter count defensible against a dataset with at most two observed
transitions per fish). Gaussian components are ordered by a prespecified
electrophysiological severity direction, without consulting the endpoint or
dose. With K capped at three, the fitted severity-ordered microstates are
themselves the interpretable states — there is no separate macrostate
collapse step. See [STATE_INTERPRETATION.md](../results/STATE_INTERPRETATION.md)
for what each state means in plain neurophysiology.

## Primary result: dose ordering of recovered states

Injury dose and every other protocol field are excluded from the feature
matrix, so it never influences model fitting. Dose is used at evaluation time
only, to test whether the fitted structure lines up with a label the model
never saw. For every fish, the mean severity-ordered Viterbi state across its
own sessions is correlated (Spearman) against its injury arm, with:

- a subject-level (fish-level) bootstrap 95% confidence interval;
- a one-sided permutation null built from shuffled arm labels, matching the
  prespecified directional prediction (higher dose → higher state index);
- a partial correlation adjusting for every covariate actually available:
  recording batch, clutch, mean session time-of-day, and mean QC
  continuities — proxies for the protocol's true experimental unit
  (`insult_batch_id`, absent from this dataset) and for the ~2-hour
  session-timing spread against the protocol's ±30-minute target.

Three negative controls accompany the primary result:
**label-shuffled** (the permutation null applied as a robustness check on the
whole distribution), **sham-only refit** (the HMM refit using only
uninjured fish, then every fish decoded through it, to check the ordering
is not an artifact of injured-fish data pulling the fitted state means
during training), and **leave-one-arm-out** (the primary correlation
recomputed with each arm excluded in turn, to check no single arm carries
the result). All three are reported in
[NEGATIVE_CONTROLS.md](../results/NEGATIVE_CONTROLS.md).

## Endpoint

The 6 dpf endpoint is behavioral and three-valued: `1` if the blinded scorer
logged at least one qualifying event (Baraban stage ≥ 2 with passing pose QC) in
the 6 dpf session, `0` if the fish was observed at 6 dpf without one, and `NA`
if there is no evidence it was observed at all.

Evidence of observation is a 6 dpf LFP session or any 6 dpf behavioral row —
including a normal one, since the Event Log records normal swim bouts, so
presence in the log proves observation while absence alone proves nothing.

Seven fish fall in the `NA` class. They are excluded from endpoint scoring
rather than counted as negatives, per the protocol rule that dead, lost, or
inadequately covered larvae are `NA` and never `0`. Coding them `0` would pad
the negative class with animals nobody checked and inflate apparent
discrimination.

The endpoint is computed only from the Event Log and shares no variable with the
LFP feature matrix, so the forecast target is independent of the model's inputs.

## Evaluation

The fish-level split assigns 70% to training and 30% to testing, stratified by
injury arm and endpoint. The endpoint is used for split balance and held-out
scoring only. This same split underlies both the primary dose-ordering result
above and the secondary forecast below.

Held-out evaluation of the **secondary** 6 dpf forecast includes:

- arm/dpf occupancy and worsening/recovery fractions;
- causal 6 dpf forecast ROC-AUC, average precision, Brier score, sensitivity,
  specificity, and bootstrap confidence intervals;
- calibration: observed positive rate against the mean, median, and maximum
  forecast risk, and the count above threshold;
- association between forecast risk and 6 dpf behavioral abnormality, including
  a dose/batch-adjusted partial Spearman correlation, and a plain statement of
  whether the unadjusted association survives that adjustment;
- a head-to-head comparison against an elastic-net landmark logistic
  regression fit on the identical causal ≤5 dpf feature vector and the
  identical split, reported honestly regardless of which model wins.

For the forecast, the final filtered distribution from the available 4–5 dpf
prefix is propagated one or two steps through the learned ordered microstate
transition matrix. No 6 dpf LFP or behavior enters the forecast.

### What cannot be measured

These are real animals with no latent-state ground truth, so state-recovery
accuracy has no referent. The analysis reports no such number and substitutes no
proxy; the confusion-matrix figure does not exist. The latent states are
validated only indirectly, through the forward forecast, the dose ordering, and
the independent behavioral channel.

Discrimination and calibration are reported separately. The propagated quantity
is the probability of occupying the top *LFP* macrostate while the endpoint is a
*behavioral* event, so the score can rank well (AUC 0.749) yet sit far below the
fixed 0.5 threshold. The metrics record the observed positive rate alongside the
forecast-risk distribution so a low sensitivity is not mistaken for a ranking
failure.

## References

- Locskai et al. 2025. <https://doi.org/10.1242/bio.060601>
- Eimon et al. 2018. <https://doi.org/10.1038/s41467-017-02404-4>
- Whyte-Fagundes et al. 2025. <https://doi.org/10.1038/s42003-025-08310-6>
- Hong et al. 2016. <https://doi.org/10.1038/srep28248>
- Mathis et al. 2018. <https://doi.org/10.1038/s41593-018-0209-y>
- Nath et al. 2019. <https://doi.org/10.1038/s41596-019-0176-0>
- Baraban et al. 2005. <https://doi.org/10.1016/j.neuroscience.2004.11.031>
