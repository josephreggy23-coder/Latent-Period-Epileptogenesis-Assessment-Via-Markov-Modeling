# Results

This directory contains the committed seed-42 **synthetic placeholder**
benchmark. No artifact is an experimental or biological result.

## Snapshot

| Measure | Result |
|---|---:|
| Generated fish / LFP sessions | 240 / 706 |
| QC-passing sessions | 683 (96.7%) |
| Contiguous model sessions | 662 from 231 fish |
| Train / held-out fish | 168 / 72 |
| Selected candidate | K = 4 (upper edge of K = 2–4) |
| Forecast-eligible held-out fish | 68 (12 planted positives) |
| Six-decimal tie-aware ROC-AUC | 0.847 (95% bootstrap CI 0.717–0.945) |
| Average precision / prevalence baseline | 0.479 / 0.176 |
| Brier score / prevalence baseline | 0.104 / 0.145 |
| Sensitivity / specificity at 0.50 | 0.417 / 0.946 |

Forecast scores have five operational levels at six-decimal precision. The
unrounded numerical-sensitivity AUC is 0.864; primary metrics use serialized
six-decimal probabilities so ties and correlations reproduce exactly.

## Top-level reports

- [`TBI_MODEL_RESULTS.md`](TBI_MODEL_RESULTS.md): generated human-readable
  interpretation and scientific limits.
- [`tbi_model_metrics.json`](tbi_model_metrics.json): run configuration,
  software versions, normalized input hashes, preprocessing, model parameters,
  convergence, state mappings, cohort flow, held-out metrics, and precision
  sensitivity.

## Figures

- [`tbi_model_selection.png`](figures/tbi_model_selection.png): train-only BIC
  and cross-validated log likelihood.
- [`tbi_state_confusion.png`](figures/tbi_state_confusion.png): held-out planted
  macrostate self-check.
- [`tbi_early_prediction_roc.png`](figures/tbi_early_prediction_roc.png):
  forward-only 6 dpf planted-endpoint ROC curve.
- [`tbi_state_trajectories.png`](figures/tbi_state_trajectories.png): held-out
  synthetic arm trajectories with SEM.
- [`tbi_transition_matrix.png`](figures/tbi_transition_matrix.png): ordered HMM
  microstate transition matrix.
- [`tbi_dlc_validation.png`](figures/tbi_dlc_validation.png): generated
  pose-style concordance plots.

## Tables

- [`tbi_split_assignments.csv`](tables/tbi_split_assignments.csv): fish-level
  train/test assignments.
- [`tbi_scored_test_sessions.csv`](tables/tbi_scored_test_sessions.csv):
  held-out forward-filtered states and planted labels.
- [`tbi_early_predictions.csv`](tables/tbi_early_predictions.csv): serialized
  six-decimal 6 dpf forecast probabilities.
- [`tbi_state_occupancy.csv`](tables/tbi_state_occupancy.csv): arm/day state
  means, SEM, and occupancy fractions.
- [`tbi_group_transition_summary.csv`](tables/tbi_group_transition_summary.csv):
  stable, worsening, and recovery fractions.
- [`tbi_transition_matrix.csv`](tables/tbi_transition_matrix.csv): ordered
  microstate transitions.

## Interpretation boundary

Perfect planted-state recovery is a generator self-check. Generated behavior
shares the planted latent state and is not independent validation. The pooled
dose/risk correlation is an arm-gradient check, not a within-arm dose response.
Bootstrap intervals are conditional on one fitted model and exclude training,
model-selection, injury-event clustering, and generator uncertainty.
