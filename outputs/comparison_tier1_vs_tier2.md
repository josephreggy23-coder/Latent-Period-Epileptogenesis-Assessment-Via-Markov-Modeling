# RELAPSE - Tier 1 vs Tier 2 comparison

| metric | Tier 1 (Gaussian HMM) | Tier 2 (Bayesian hierarchical HMM) |
|---|---|---|
| State recovery (held-out) | 99.4% | (emissions shared; ~equal) |
| Early-prediction ROC-AUC | 0.879 | 0.879 |
| Accuracy @ threshold | 91.7% | 91.7% |
| Flagged before seizure | 10/10 | 10/10 |
| Mean lead-time (h) | 3.4 | 3.4 |
| Microplastic effect | point est. 2.05x advance-rate | odds ratio 2.91 (94% HDI [1.42, 5.15]), P>0=0.998 |
| Uncertainty | point estimates | full posterior on every parameter |

**Takeaways**

- Both tiers recover the planted states and flag every held-out epileptic fish before its first seizure.
- Tier 2's advantage is *calibrated uncertainty*: the microplastic effect comes with a full posterior and a credible interval rather than a single number, and per-fish random effects let the model adapt to individuals.
- On this 'optimal-case' synthetic data the states are highly separable, so raw point-prediction accuracy is already near-ceiling in Tier 1; the Bayesian upgrade mainly buys honesty about uncertainty and a principled treatment of cross-individual variation (which matters more as real data gets noisier and sample sizes shrink).