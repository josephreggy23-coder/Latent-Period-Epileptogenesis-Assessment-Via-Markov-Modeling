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
- selected **K=4** by lowest train-only BIC (1782.6);
  train-only CV log likelihood/session
  -1.557
- identical preprocessing, severity ordering, and macrostate collapse as the
  synthetic benchmark - none of it consults the endpoint

## Endpoint

The 6 dpf high-burden endpoint is **behavioural**: a fish is positive if the
blinded scorer logged at least one qualifying event (Baraban stage >= 2 with
passing pose QC) in the 6 dpf session. It shares no variable with the LFP
feature matrix, so the forecast target is independent of the model's inputs.

## Causal 6 dpf forecast

Only an uninterrupted, QC-passing **4-5 dpf** LFP prefix is used. Its final
filtered state distribution is propagated through the learned transition matrix
to 6 dpf:

- held-out fish: **72** (18 positive)
- ROC-AUC: **0.830** (bootstrap 95% CI
  0.732-0.911)
- average precision: **0.554**
- Brier score: **0.182**
- sensitivity/specificity at probability 0.5:
  **0.111/0.981**

### Discrimination is good; calibration is not

The forecast **ranks** fish well but is **badly calibrated** against this
endpoint, so the fixed 0.5 threshold is a poor operating point and the
sensitivity above should not be read as a ranking failure:

- observed positive rate: **0.250**
- mean / median forecast risk:
  **0.102 /
  0.029**
  (maximum 0.707)
- held-out fish above 0.5: **3** of
  72

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
  rho=0.672
  (p=1.03e-10)
- 6 dpf forecast risk vs independent 6 dpf behavioural abnormality:
  rho=0.475
  (p=4.81e-05), n=67
- dose/batch-adjusted partial rho:
  0.226
  (p=0.0656)

## Boundaries

- Three sessions per fish is a short series for a Markov model; the transition
  matrix is estimated from at most two observed steps per animal.
- Behaviour is scored in three discrete sessions, so event timing is
  interval-censored.
- The abnormality index is built only from event-rate and stage terms, which
  remain defined when the scorer logged nothing; kinematic columns are reported
  but deliberately excluded from the index.
- A single forebrain electrode per fish bounds the information available.
