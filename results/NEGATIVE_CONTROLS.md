# Negative controls for the primary dose-ordering result

Three checks that a real dose-ordering signal should survive, and a shuffled
or contaminated one should not. See `docs/PREREGISTRATION.md` for why the
fitting step is provably dose-blind (features only), which is why two of the
three controls act on the evaluation stage rather than refitting the model:
refitting with shuffled or arm-dropped data would learn an identical model,
since dose never enters fitting either way.

## Primary result being checked

- full-cohort **240** fish: Spearman rho=0.697, 95% bootstrap CI [0.617, 0.761], one-sided permutation p=0.0002 (5000 shuffles)
- covariate-adjusted (batch/clutch/timing/QC proxies): partial rho=0.693 (p=1.27e-35)

## 1. Label-shuffled (evaluation-time)

Same permutation null used for the primary result's significance test: injury-arm labels shuffled across fish with fitted states held fixed. Fitting never sees dose labels, so shuffling only affects the evaluation, not the model -- this negative control is that shuffle applied at the scale of the full null distribution rather than a single draw.

- null distribution over 5000 shuffles: mean rho=-0.0005, SD=0.0653
- observed rho=0.697 against that null: one-sided p=0.0002

If the pipeline could manufacture a dose-ordering signal this strong from
noise alone, the null distribution would routinely reach the observed rho.
It does not.

## 2. Sham-only refit

The HMM is refit using **only the 60 sham fish's** sequences (fresh train-only scaler, fresh EM fit, converged=True), then every fish -- including the three injured arms this model never saw during fitting -- is decoded through it. The same dose-ordering statistic is recomputed on that sham-only-derived state index.

- Spearman rho=0.705, 95% bootstrap CI [0.632, 0.768], one-sided permutation p=0.0002

A dose-ordering signal that survives being decoded through a model that never
saw an injured fish during fitting is evidence the ordering reflects real
structure in the injured fish's features, not injured-fish data pulling the
fitted state means around during training.

## 3. Leave-one-arm-out

The primary fitted (dose-blind) model's decoded state index is unchanged;
each arm is dropped from the correlation computation in turn to check the
signal is not carried entirely by a single extreme arm.

- excluding **sham** (180 fish): rho=0.551 (p=1.04e-15)
- excluding **tbi_low** (180 fish): rho=0.765 (p=8.28e-36)
- excluding **tbi_moderate** (180 fish): rho=0.756 (p=1.52e-34)
- excluding **tbi_high** (180 fish): rho=0.539 (p=5.88e-15)
