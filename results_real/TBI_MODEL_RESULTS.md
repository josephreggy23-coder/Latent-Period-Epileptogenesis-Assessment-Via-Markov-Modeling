# Real larval-zebrafish TBI Markov-model results

> **Measured data.** Every row here comes from a real recording, not the
> simulator. There is no planted latent state, so held-out state-recovery
> accuracy is **not reported and not measurable** - only the forward 6 dpf
> forecast is scored.

## Run scope

- **240 fish**, 706 LFP sessions at 4-6 dpf,
  706 passing QC (100.0%)
- 706 contiguous modelling sessions from
  240 fish with a usable 4 dpf baseline
- selected **K=4** by lowest train-only BIC (1802.6);
  train-only CV log likelihood/session
  -1.696
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

Resolved: **60 positive**, **173 negative**, **7 unresolved (`NA`)**.

Unresolved fish are excluded from endpoint scoring rather than counted as
negatives. See `docs/EXPERIMENTAL_PROTOCOL.md` section 5.

## Causal 6 dpf forecast

Only an uninterrupted, QC-passing **4-5 dpf** LFP prefix is used. Its final
filtered state distribution is propagated through the learned transition matrix
to 6 dpf:

- held-out fish: **71** (19 positive)
- ROC-AUC: **0.749** (bootstrap 95% CI
  0.642-0.853)
- average precision: **0.438**
- Brier score: **0.206**
- sensitivity/specificity at probability 0.5:
  **0.105/0.962**

### Discrimination versus calibration

The forecast **ranks** fish well but is **badly calibrated** against this
endpoint, so the fixed 0.5 threshold is a poor operating point and the
sensitivity above should not be read as a ranking failure:

- observed positive rate: **0.268**
- mean / median forecast risk:
  **0.116 /
  0.037**
  (maximum 0.625)
- held-out fish above 0.5: **4** of
  71

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
  rho=0.623
  (p=6.38e-09)
- 6 dpf forecast risk vs independent 6 dpf behavioural abnormality:
  rho=0.320
  (p=0.00874), n=66
- dose/batch-adjusted partial rho:
  0.033
  (p=0.793)

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
