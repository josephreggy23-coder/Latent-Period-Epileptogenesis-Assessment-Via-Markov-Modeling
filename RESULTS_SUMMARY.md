# RELAPSE - epileptogenesis HMM: consolidated results

Synthetic larval-zebrafish LFP data with KNOWN planted ground truth. Goal: confirm a Hidden Markov Model recovers the planted hidden disease states and predicts which fish become epileptic BEFORE their first seizure, using only the 5 LFP features as input.

**Headline: the model recovers the planted truth.** States are recovered at ~99% on held-out fish, every held-out epileptic fish is flagged before its seizure, and the microplastic arm shows credibly faster progression.

## State recovery (did the HMM find the planted hidden states?)

- Tier 1 held-out accuracy: **99.4%** (all-fish 99.1%).
- States severity-aligned to truth by ranking emission means. Confusion matrix in `outputs/tier1_report.md` / `tier1_confusion_matrix.png`.

## Early prediction (flag epilepsy BEFORE first seizure)

| metric | Tier 1 Gaussian HMM | Tier 2 Bayesian HMM |
|---|---|---|
| ROC-AUC (held-out) | 0.879 | 0.879 |
| Accuracy @ optimal threshold | 91.7% | 91.7% |
| Epileptic fish flagged before seizure | 10/10 | 10/10 |
| Mean lead-time before seizure | 3.4 h | 3.4 h |

Risk uses an ONLINE forward filter (observations up to t only) so there is no leakage from the future seizure. Tier 2 additionally integrates each fish's hierarchical random effect, updated by its own early data.

## Microplastic effect on progression

- **Tier 1 (point estimate):** advance-rate 0.163 (control) vs 0.333 (microplastic) = **2.05x** faster.
- **Tier 2 (full posterior):** per-step odds ratio exp(beta_mp) = **2.91** (94% HDI [1.42, 5.15]), **P(effect > 0) = 0.998**.
- Between-fish SD tau (individual variation) median 0.60.
- Both tiers agree: microplastic credibly accelerates progression toward the seizure state. See `tier2_microplastic_posterior.png`.

## Tier 1 vs Tier 2 - what the Bayesian upgrade buys

- Raw predictive accuracy is **equal** here (AUC 0.879 vs 0.879): the synthetic states are highly separable ('optimal case'), so Tier 1 is already at ceiling.
- Tier 2 adds **calibrated uncertainty** (full posterior + credible intervals on every parameter, especially the microplastic effect) and a principled **per-fish random effect** for cross-individual generalization - both of which matter more as real data gets noisier and N shrinks.

## Files

- Tier 1 code: `tier1_gaussian_hmm.py` -> `outputs/tier1_report.md`
- Tier 2 code: `tier2_bayesian_hmm.py` -> `outputs/tier2_report.md`
- Comparison: `outputs/comparison_tier1_vs_tier2.md`
- Plots: `outputs/*.png` (trajectories, risk-over-time, transition matrices, confusion matrix, microplastic posterior, advance hazards, ROC).