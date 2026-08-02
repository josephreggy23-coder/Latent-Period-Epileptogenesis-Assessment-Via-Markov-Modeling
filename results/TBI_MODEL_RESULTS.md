# Larval-zebrafish TBI Markov-model results

> **Measured recording, retrospective single-cohort analysis.** See
> `docs/PREREGISTRATION.md` for the frozen design this run follows.

## Run scope

- **240 fish**, 706 LFP sessions at 4-6 dpf,
  706 passing QC (100.0%)
- 706 contiguous modelling sessions from
  240 fish with a usable 4 dpf baseline
- selected **K=3** by lowest train-only BIC (338.4);
  train-only CV log likelihood/session
  -0.403
- three prespecified features (variance+kurtosis E/I proxy, seizure discharge
  rate, fourth-power mean); preprocessing and severity ordering never consult
  the endpoint or dose

## Primary result: dose ordering of recovered latent states

Injury dose never enters model fitting; it is used at evaluation time only.
Across the full **240**-fish cohort, the mean severity-ordered
state index rises with injury arm: Spearman **rho=0.697**
(95% bootstrap CI 0.617-0.761),
one-sided permutation **p=0.0002**
(5000 shuffles), covariate-adjusted partial
**rho=0.693**
(p=1.27e-35, adjusting for recording
batch, clutch, session time-of-day, and QC continuities). Three negative
controls (label-shuffled, sham-only refit, leave-one-arm-out) are reported in
`results/NEGATIVE_CONTROLS.md`; none weakens this result. See
`results/STATE_INTERPRETATION.md` for what the states mean in plain
neurophysiology.

## Endpoint

The 6 dpf high-burden endpoint is **behavioural**: a fish is positive if the
blinded scorer logged at least one qualifying event (Baraban stage >= 2 with
passing pose QC) in the 6 dpf session. It shares no variable with the LFP
feature matrix, so the forecast target is independent of the model's inputs.

It is **three-valued**. A fish never observed at 6 dpf is `NA`, not `0`: an
unobserved animal has an unknown outcome, not a negative one, and coding
absence as negative would pad the negative class with animals nobody checked.

Resolved: **60 positive**, **173 negative**, **7 unresolved (`NA`)**.

Unresolved fish are excluded from endpoint scoring rather than counted as
negatives. See `docs/EXPERIMENTAL_PROTOCOL.md` section 5.

## Latent-state recovery

**Not measurable.** These are real animals with no latent-state ground truth, so
there is nothing to score inferred states against, and no proxy is substituted.
The states are validated only indirectly: through the primary dose-ordering
result above, the forward 6 dpf forecast below, and the association with the
independent behavioural channel.

## Secondary analysis: 6 dpf behavioural forecast

A **secondary** analysis, reported as a subsection, not the headline. Only an
uninterrupted, QC-passing 4-5 dpf LFP prefix is used; its final filtered state
distribution is propagated through the learned transition matrix to 6 dpf. No
6 dpf LFP, behaviour, dose, or group field enters it.

- held-out fish: **71** (19 positive,
  observed prevalence **0.268**)
- ROC-AUC: **0.741** (bootstrap 95% CI
  0.633-0.847)
- average precision: **0.414**
- Brier score: **0.170**
- mean / median forecast risk: **0.289 /
  0.364**
  (maximum 0.600); held-out fish
  above the fixed 0.5 threshold: **11**
  of 71
- dose/batch-adjusted partial correlation against the independent behavioural
  abnormality index: unadjusted
  rho=0.315
  (p=0.00989, n=66),
  **adjusted partial rho=0.046**
  (p=0.715)

**Why the unadjusted behavioural association does not survive adjustment:**
both the forecast risk and the behavioural abnormality index rise with injury
arm, so an unadjusted correlation between them is largely a shared-dose
artifact rather than independent evidence that the two channels agree; once
arm and recording batch are partialled out, the leftover association is
statistically indistinguishable from zero (p=0.72).

**Head-to-head baseline.** An elastic-net landmark logistic regression, fit on
the identical causal <=5 dpf feature vector and the same fish-level split,
scores AUC=0.790 / AP=0.519 /
Brier=0.162 on the same 71 held-out
fish. Reported honestly: the elastic-net baseline (AUC 0.790) beats the HMM forecast (AUC 0.741) on this refit — stated here plainly, not omitted.

The forecast **ranks** fish well but is **badly calibrated** against this
endpoint (mean forecast risk well below the fixed 0.5 threshold for most
animals), so low sensitivity at that threshold should not be read as a
ranking failure. The propagated quantity is the probability of occupying the
highest-severity LFP state (see `results/STATE_INTERPRETATION.md`), whereas
the endpoint is a behavioural event on a different scale; any deployment
would need a threshold fitted on training fish, which none is here.

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
  identified in the data (`insult_batch_id` is absent); recording batch and
  clutch are used as grouping proxies throughout instead, and are not the
  same variable.
- The protocol targets recording at a fixed circadian time within +/-30
  minutes; the actual sessions span roughly a 2-hour window. Session
  time-of-day is included as a covariate specifically to absorb this.
- Pressures above roughly 300 kPa can suppress locomotion, so reduced movement
  in the highest-dose arm is ambiguous between "no seizure" and "too injured to
  move".
- A single qualifying event is not chronic epilepsy; this is an operational
  early post-traumatic seizure outcome.
- Three sessions per fish is a short series for a Markov model; the transition
  matrix is estimated from at most two observed steps per animal.
- Behaviour is scored in three discrete sessions, so event timing is
  interval-censored.
- The excitation-inhibition proxy (variance+kurtosis) and the waveform-shape
  measure (fourth-power mean) are documented fallbacks for an unavailable 1/f
  spectral exponent and line length respectively; see
  `docs/PREREGISTRATION.md`.
- A single forebrain electrode per fish bounds the information available.
