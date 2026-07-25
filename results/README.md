# Results

This directory contains the committed seed-42 synthetic benchmark outputs.

## Top-level reports

- `TBI_MODEL_RESULTS.md`: concise human-readable results and scientific limits.
- `tbi_model_metrics.json`: full machine-readable configuration, preprocessing,
  state mapping, model-selection, held-out, forecast, dynamics, and behavior
  metrics.

## Figures

`figures/` contains:

- model-order selection;
- held-out state-recovery confusion matrix;
- causal DPF6 forecast ROC curve;
- held-out state trajectories;
- ordered transition matrix;
- DeepLabCut-style validation.

## Tables

`tables/` contains:

- fish-level split assignments;
- scored held-out sessions;
- DPF6 forecast probabilities;
- state occupancy by arm and dpf;
- group transition summaries;
- the ordered HMM transition matrix.

All results describe the planted structure of a synthetic simulator. They are
not biological effect estimates.
