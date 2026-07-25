# Synthetic larval-zebrafish TBI Markov benchmark

> **Synthetic demonstration only.** No committed row represents an experimental
> animal. Every current observation, state, endpoint, and pose-style behavior
> value is generated. These results are not evidence of post-traumatic epilepsy,
> treatment efficacy, or feasibility of repeated invasive recordings.

## Run scope

- 240 generated fish across sham and 3/5/7-hit arms
- 706 generated sessions; 683
  passed QC (96.7%)
- 662 contiguous model sessions from
  231 fish with a usable 4 dpf baseline
- 168 train / 72 test fish, with
  0 overlapping fish
- resistance-change and noise QC failures remain auditable but are excluded;
  a later gap terminates the usable prefix
- positive heavy-tailed features receive `log1p`; robust preprocessing is
  fitted on training fish only
- selected **K=4** by lowest train-only BIC (706.2);
  train-only CV log likelihood/session -0.552

K=4 is the upper boundary of the tested candidate set
[2, 3, 4]; it does not establish 4 biological states. Adjacent
severity-ordered microstates are collapsed to three planted validation
macrostates without consulting truth labels.

## Held-out planted-state self-check

- balanced accuracy: **1.000**
- macro F1: **1.000**
- adjusted Rand index: **1.000**
- scored sessions: **206**

Perfect recovery is an expected self-check for deliberately separated synthetic
emissions, not evidence that biological states have been identified.

## Forward-only early forecast

Each held-out fish's forecast used only its uninterrupted, QC-passing 4-5 dpf
LFP prefix. The final filtered state distribution was propagated through the
learned transition matrix to predict the separate planted 6 dpf high-burden
endpoint. No target-fish 6 dpf LFP or behavior entered its forecast;
training-fish 4-6 dpf sessions were used to estimate the HMM emissions and
transition dynamics.

- held-out fish: **68** (12 positive)
- ROC-AUC: **0.847** (bootstrap 95% CI
  0.717-0.945)
- average precision: **0.479** versus prevalence
  baseline 0.176
- Brier score: **0.104** versus constant-prevalence
  baseline 0.145
- sensitivity/specificity at probability 0.5:
  **0.417/0.946**
- confusion counts: 5 TP,
  53 TN,
  3 FP,
  7 FN
- operational score levels: **5** at
  6-decimal precision

Primary rank metrics use the same rounded probabilities committed to CSV so
ties are meaningful and results reproduce after serialization. For numerical
sensitivity, the unrounded AUC was
0.864.
"Causal" in the implementation means forward-only temporal filtering, not
causal-effect inference; the split is retrospective and endpoint-stratified.

## Dose/dynamics and behavior checks

- pooled synthetic arm-gradient check: Spearman
  **rho=0.329**
- injured-fish-only dose/risk check: Spearman
  **rho=0.181**
- forecast risk vs generated 6 dpf behavior abnormality: Spearman
  **rho=0.605**
- dose/batch-adjusted generated-behavior check: partial Spearman
  **rho=0.480**

Generated behavior is withheld from HMM inputs but shares the planted latent
generator, so this is concordance rather than independent validation. The pooled
dose value is an arm-gradient check, not a within-arm dose-response estimate.

## Method boundaries

- [Locskai et al.](https://doi.org/10.1242/bio.060601) motivates the
  blast-pressure syringe insult and repeated-hit dose axis.
- [Eimon et al.](https://doi.org/10.1038/s41467-017-02404-4) motivates LFP
  acquisition, resistance-change/noise QC, overlapping-window higher moments,
  and ICA complexity.
- [Mathis et al.](https://doi.org/10.1038/s41593-018-0209-y) and
  [Nath et al.](https://doi.org/10.1038/s41596-019-0176-0) motivate the
  pose-summary interface.

Neither source paper reports this exact 4-6 dpf repeated same-fish LFP design.
Raw LFP feature extraction, pose tracking, injury-event clustering, and model
uncertainty from refitting are outside the current benchmark.
