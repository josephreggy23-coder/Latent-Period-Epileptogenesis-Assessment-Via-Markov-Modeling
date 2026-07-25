# RELAPSE Tier 2 - Bayesian hierarchical HMM (numpyro/NUTS)

Hidden states marginalized via the forward algorithm; NUTS samples the continuous posterior. Microplastic enters as a covariate on the per-step progression hazard; each fish has a hierarchical random effect u_i ~ N(0, tau).

## Microplastic effect (full posterior)

- **beta_mp (log-odds of progressing / step): median 1.068, 94% HDI [0.451, 1.699]**
- **Per-step odds ratio exp(beta_mp): median 2.91, 94% HDI [1.42, 5.15]**
- **P(beta_mp > 0) = 0.998** -> microplastic credibly accelerates progression.
- Between-fish SD tau: median 0.60 (individual variation in susceptibility).

## Baseline progression hazards (posterior median P(advance)/step)

| transition | control | microplastic |
|---|---|---|
| 0->1 | 0.13 | 0.30 |
| 1->2 | 0.32 | 0.58 |
| 2->3 | 0.23 | 0.46 |

## Held-out early prediction

- ROC-AUC 0.879, accuracy 91.7%, flagged 10/10 before seizure, mean lead-time 3.4 h.
- Risk integrates each fish's random effect (Gauss-Hermite) updated online by its own early observations -> cross-individual adaptation.

## Figures

- `outputs/tier2_microplastic_posterior.png`
- `outputs/tier2_advance_hazard.png`
- `outputs/tier2_roc_compare.png`