# Synthetic larval-zebrafish TBI Markov-model results

> **Benchmark only.** Every observation, state, endpoint, and DeepLabCut-like
> metric is synthetic. These results are not evidence of post-traumatic
> epilepsy, treatment efficacy, or feasibility of repeated invasive recordings.

## Run scope

- 240 simulated larvae across sham and 3/5/7-drop TBI arms
- TBI at 3 dpf; LFP/behavior sessions at 4-6 dpf
- LFP sessions failing the prespecified electrode-shift/noise QC remain in the
  dataset but are excluded from modeling; a QC gap terminates the usable
  4 dpf-based prefix rather than being compressed into one transition
- positive heavy-tailed features receive `log1p`; robust preprocessing is fit
  on training fish only; the split is at the fish level
- selected **K=4** by lowest train-only BIC (706.2);
  train-only CV log likelihood/session -0.552
- the K statistical components are severity ordered without truth labels, then
  adjacent components are collapsed to the simulator's three validation
  macrostates at the two largest prespecified-score gaps

## Held-out latent-state recovery

- balanced accuracy: **1.000**
- macro F1: **1.000**
- adjusted Rand index: **1.000**

Per-session planted states are used only in this scoring step. The planted 6
dpf endpoint is used to outcome-stratify the fish-level split and to score the
forecast; neither form of truth enters HMM features or fitting.

## Causal early prediction

Only an uninterrupted, QC-passing 4-5 dpf LFP prefix was used. Its final
filtered state distribution was propagated through the learned transition
matrix to predict the separate planted 6 dpf high-burden endpoint:

- held-out fish: **68** (12 positive)
- ROC-AUC: **0.864** (bootstrap 95% CI
  0.758-0.945)
- average precision: **0.489**
- Brier score: **0.104**
- sensitivity/specificity at probability 0.5:
  **0.417/0.946**

## Dose/dynamics and behavior checks

- synthetic dose index vs DPF6 forecast risk: Spearman
  **rho=0.343**
- DPF6 forecast risk vs DPF6 synthetic DLC abnormality: Spearman
  **rho=0.628**
- after synthetic dose/batch adjustment: partial Spearman
  **rho=0.476**

Locomotor speed is not treated as a severity ruler: the simulator permits
hyperactivity at moderate injury and stunned/inactive behavior at high injury.

## Method boundaries

- [Locskai et al.](https://doi.org/10.1242/bio.060601) motivates the
  blast-pressure syringe insult and repeated-hit dose axis.
- [Eimon et al.](https://doi.org/10.1038/s41467-017-02404-4) motivates LFP
  acquisition, QC, overlapping-window higher moments, and ICA complexity.
- [Mathis et al.](https://doi.org/10.1038/s41593-018-0209-y) and
  [Nath et al.](https://doi.org/10.1038/s41596-019-0176-0) motivate the
  DeepLabCut validation workflow.

Neither source paper reports this exact 4-6 dpf repeated same-fish LFP design.
