# Preregistration

Committed before any refitting on the reduced feature set. Everything below is
frozen at the commit recorded in `README.md`. Nothing in Tasks 3 onward may
deviate from this document without an explicit, dated amendment appended to
the [Amendments](#amendments) section at the bottom.

## Hypothesis

Graded mechanical brain injury produces dose-ordered latent electrophysiological
states in larval zebrafish. A hidden Markov model that never sees injury dose
should recover that ordering, and the states should be interpretable as
excitation-inhibition shifts.

**Directional prediction, stated before fitting:** net excitation increases
with injury dose, so the excitation-inhibition proxy should shift monotonically
across the sham / 100 g / 200 g / 300 g arms (measured 0 / 115 / 210 / 319 kPa),
and higher-dose fish should occupy higher-index latent states more often than
lower-dose fish.

## Feature allowlist (locked from Task 3)

Exactly three prespecified feature concepts, four columns:

1. **Excitation-inhibition proxy** — `lfp_variance_uv2` and `lfp_kurtosis`
   together. The originally intended proxy was the aperiodic 1/f spectral
   exponent (specparam/FOOOF), but the [Task 1 audit](../results/) established
   that no power spectral density or raw trace exists anywhere in this
   repository or its source workbooks — only seven pre-summarized scalar
   statistics per session. A 1/f exponent is therefore **not computable** from
   available data. Per the audit's own fallback, the variance/kurtosis pair
   substitutes for it. **This substitution is a real limitation, not a free
   choice**: variance and kurtosis are amplitude-distribution statistics, not
   a validated E/I marker, and this is stated plainly wherever the proxy is
   used.
2. **Epileptiform discharge rate** — `lfp_seizure_event_rate_per_h`.
3. **Waveform-shape measure** — `lfp_fourth_power_mean_uv4`. The originally
   intended measure was line length, which (like the 1/f exponent) requires a
   raw or windowed trace that does not exist in the source data. The existing
   fourth-power mean is the documented fallback.

All four columns already receive `log1p` (heavy-tailed, non-negative) and
train-only median/IQR robust scaling, unchanged from the prior seven-feature
pipeline. `lfp_mean_uv`, `lfp_skewness`, and `lfp_ica_complexity` are dropped
from the allowlist entirely.

## Model

- Diagonal-Gaussian HMM, log-space Baum-Welch EM, as already implemented in
  `tbi_markov.hmm`.
- **K = 2 or K = 3 candidates only** (down from 2-4), selected by **train-only
  BIC**. K = 4 is dropped because the seven-feature model already reached the
  upper edge of its tested range at K = 4 with 71 free parameters against 466
  observed transitions; four features and a lower K ceiling keep the
  parameter count defensible against a dataset with at most two transitions
  per fish.
- Severity ordering of components is prespecified (rising: all four features
  increase with injury severity; there is no falling feature in the reduced
  set) and never consults the endpoint or dose.
- No macrostate collapse. With K capped at 3, the fitted microstates **are**
  the interpretable severity states; collapsing them into a fixed number of
  named categories is no longer necessary and is removed as unneeded
  complexity.

## Primary outcome — dose ordering of state occupancy

For every fish with a usable contiguous LFP sequence, compute the mean
severity-ordered state index (Viterbi-decoded, averaged across that fish's
own sessions). Injury dose never enters model fitting; it is used at
evaluation time only.

Computed across the full cohort (dose was never used for fitting, so this is
an association test of unsupervised structure against a label the model never
saw — not a held-out predictive claim). As a secondary robustness check, the
same statistic is also reported restricted to the held-out test-fish
partition alone.

Reported together:

- Spearman rho between injury arm (ordinal: sham < low < moderate < high) and
  mean state index, with a subject-level (fish-level) bootstrap 95% CI.
- A permutation null built from shuffled arm labels (fish states held fixed),
  reported as a one-sided p-value against the directional prediction.
- The same rho adjusted for every covariate actually available: recording
  `batch`, `clutch_id`, mean session time-of-day, and mean QC continuits
  (`electrode_shift_pct`, `rms_noise_mv`) per fish, via rank-residualized
  partial Spearman correlation (same technique already used for the
  behavioral-validation partial correlation).
  - **Limitation, stated here in advance:** the wet-lab protocol's actual
    experimental unit is the injury/drop batch (`insult_batch_id`), which the
    [Task 1 audit](../docs/EXPERIMENTAL_PROTOCOL.md) established is **absent**
    from the dataset. Recording `batch` (1-3) and `clutch_id` (6 clutches)
    are the only available grouping proxies and are used in its place. They
    adjust for recording-session and breeding-clutch structure, not for the
    physical drop batch the protocol defines as the true nested unit.
  - **Limitation, stated here in advance:** the protocol specifies recording
    at a fixed circadian time within ±30 minutes; the actual data spans
    14:00-16:00 (a 2-hour window). Session time-of-day is included as a
    covariate specifically to absorb this.
- Three negative controls, reported in `results/NEGATIVE_CONTROLS.md`:
  label-shuffled evaluation, sham-only refit, and leave-one-arm-out.

## Secondary outcome — 6 dpf behavioral forecast

Unchanged causal design: the final filtered state distribution from an
uninterrupted, QC-passing 4-5 dpf prefix is propagated through the learned
transition matrix to 6 dpf. No 6 dpf LFP, behavior, dose, or group field
enters the forecast. Fixed decision threshold **0.50**, prespecified and never
tuned on held-out fish.

Reported together in one README subsection: ROC-AUC, average precision,
Brier score, observed positive prevalence, mean forecast risk, and the
dose/batch-adjusted partial Spearman correlation against the independent
behavioral abnormality index. This is explicitly a **secondary** analysis: a
subsection, never the headline.

An elastic-net landmark logistic regression, fit on the same reduced features
and the identical fish-level split, is reported alongside the HMM forecast as
the head-to-head baseline the README has listed as "planned" since the
project began. The comparison is reported honestly regardless of outcome: if
the HMM does not beat the baseline, that is stated in the README, not
omitted.

## Fish-level split

Unchanged: 70% train / 30% test, stratified by injury arm and endpoint,
seed 42, zero fish overlap between partitions. Preprocessing (log1p, robust
scaling) is fit on training fish only, as before.

## State interpretation

Each severity-ordered state receives a name and a one-paragraph plain-language
neurophysiological description (`results/STATE_INTERPRETATION.md`), compared
against the canonical acute-depression / latent / hyperexcitable
three-stage progression. Where the fitted K (2 or 3) does not resolve that
full three-stage structure, the mismatch is stated explicitly rather than
forced. No bare "State N" appears in user-facing prose after this document
is written; every reference pairs the name with the index.

## What would falsify the primary claim

A non-significant or negative Spearman rho between dose and mean state index
after covariate adjustment, or a permutation p-value that does not support
the one-sided directional prediction, falsifies the primary hypothesis as
stated. That outcome would be reported as such, not reframed.

## Amendments

None yet. Any future deviation from this document must be recorded here with
the date and a plain statement of what changed and why.
