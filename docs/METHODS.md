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

The HMM uses seven LFP features. Variance, kurtosis, fourth-power mean, and event
rate receive `log1p` transforms. Median and interquartile-range scaling
parameters are estimated from training fish only.

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

Two-, three-, and four-state candidates are compared on training-only BIC and
three-fold fish-level cross-validated log likelihood. Gaussian components are
ordered by a prespecified electrophysiological severity direction, without
consulting the endpoint. When four microstates are selected, adjacent ordered
components are collapsed into three interpretable macrostates at the two largest
severity-score gaps.

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
scoring only.

Held-out evaluation includes:

- arm/dpf occupancy and worsening/recovery fractions;
- causal 6 dpf forecast ROC-AUC, average precision, Brier score, sensitivity,
  specificity, and bootstrap confidence intervals;
- calibration: observed positive rate against the mean, median, and maximum
  forecast risk, and the count above threshold;
- association between the injury dose index and forecast risk;
- association between forecast risk and 6 dpf behavioral abnormality, including
  a dose/batch-adjusted partial Spearman correlation.

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
