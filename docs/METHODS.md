# Methods

## Scope

This project is a deterministic synthetic-placeholder demonstration joining
three methodological interfaces:

1. a larval-zebrafish syringe-blast TBI insult;
2. forebrain LFP acquisition and statistical pattern summaries;
3. generated markerless-pose-style session summaries.

The source papers do not contain the row-level data committed here and do not
report repeated same-fish LFP at 4–6 dpf after 3 dpf TBI. Raw-signal feature
extraction is not implemented: seizure rate, ICA complexity, higher moments,
and behavior summaries are generated columns accepted by the analysis
interface, not quantities computed from raw recordings by this repository.

## Placeholder cohort

The default seed-42 cohort contains 240 larvae, with 60 assigned to each of
`sham`, `tbi_low`, `tbi_moderate`, and `tbi_high`.

TBI occurs at 3 dpf. Placeholder LFP and behavior sessions occur at 4, 5, and
6 dpf, subject to planted attrition. The apparatus is held constant at a 10 mL
syringe, three-prong clamp, 200 g weight, and 108 cm height. Injury dose varies
only by 0, 3, 5, or 7 hits.

The 195 kPa per-hit center and all numerical response distributions are
template assumptions. `cumulative_pressure_burden_kpa_hits` equals generated
measured peak kPa multiplied by hit count and is treated only as a dose index.

The current generator assigns pressure and partitions at the fish level. A
measured syringe exposure may cluster several larvae within one injury event.
Such a study must add injury-event, clutch, and plate identifiers and use
grouped splitting and cluster-aware uncertainty; the present fish-level
bootstrap does not model those dependencies.

## LFP acquisition interface

The Eimon-adapted placeholder session uses:

- non-anesthetized, non-paralyzed agar embedding;
- a glass forebrain electrode and Intan-style acquisition;
- 240-minute recording duration;
- 3 kHz sampling and anti-alias low-pass;
- 0.5 Hz–1 kHz bandpass;
- 30-second windows with 20-second overlap;
- amplitude mean, variance, skewness, kurtosis, and fourth-power mean;
- seizure-event rate and single-channel ICA complexity.

Session QC passes only when RMS noise is below 0.2 mV and the absolute
electrode-resistance change is no greater than 50%. The latter is adapted from
Eimon et al.'s resistance-deviation exclusion rule and is a generated QC proxy,
not a directly measured physical displacement. Failed sessions remain in the
table for auditability.

## Generated behavior concordance

The pose-style table declares a head, four-midline, and tail-tip interface and
directly generates session summaries for speed, rest fraction, burst rate, tail
bend, tail-angle change, and whirlpool-like circling. It contains no per-frame
coordinates.

These values are synthetic placeholders. No video was labeled, no network was
trained, and behavior never enters the HMM. However, behavior and LFP emissions
share the same planted latent state, so their association is a concordance
check rather than independent validation. Locomotion is intentionally
non-monotonic because a generated severe state may be hyperactive or
stunned/inactive.

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
- Viterbi decoding and forward-only filtering.

Here, "causal filtering" means that the probability at time *t* uses only
observations at or before *t*. It does not mean causal-effect inference.

Two-, three-, and four-state candidates are compared on training-only BIC and
three-fold fish-level cross-validated log likelihood. Gaussian components are
ordered by a prespecified electrophysiological severity direction without
consulting planted truth. When four microstates are selected, adjacent ordered
components are collapsed into three validation macrostates at the two largest
severity-score gaps.

Selecting four states means only that K=4 had the lowest BIC at the upper edge
of the tested K=2–4 set. It does not establish four biological states.

## Evaluation

The fish-level split assigns 70% to training and 30% to testing, stratified by
injury arm and planted endpoint. The endpoint is used for split balance and
held-out scoring only.

Held-out evaluation includes:

- balanced accuracy, macro F1, adjusted Rand index, and per-state recall;
- arm/dpf occupancy and worsening/recovery fractions;
- forward-only DPF6 forecast ROC-AUC, average precision, Brier score, sensitivity,
  specificity, and bootstrap confidence intervals;
- pooled synthetic arm-gradient check between dose index and forecast risk;
- concordance between forecast risk and generated DPF6 behavior abnormality, including
  dose/batch-adjusted partial Spearman correlation.

For the forecast, the final filtered distribution from the available 4–5 dpf
prefix is propagated one or two steps through the learned ordered microstate
transition matrix. No held-out target-fish 6 dpf LFP or behavior enters that
fish's forecast; training-fish 4–6 dpf sessions are used to estimate the HMM
emissions and transition dynamics. Probabilities are rounded to six decimals
before rank-based metrics and correlations so ties are operationally meaningful
and the metrics reproduce from serialized tables.
Bootstrap intervals resample held-out fish conditional on the already fitted
model; they do not include training or model-selection uncertainty.

## References

- Locskai et al. 2025. <https://doi.org/10.1242/bio.060601>
- Eimon et al. 2018. <https://doi.org/10.1038/s41467-017-02404-4>
- Mathis et al. 2018. <https://doi.org/10.1038/s41593-018-0209-y>
- Nath et al. 2019. <https://doi.org/10.1038/s41596-019-0176-0>
