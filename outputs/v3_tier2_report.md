# ECHO V3 - Tier 2 Bayesian treatment-effect analysis

Full posterior uncertainty on the SE and VPA effects. Two models on the full cohort (N=128): a progression-hazard HMM (latent mechanism) and a logistic outcome model (clinical sensitization). Per-fish random effects are omitted (only 2 timepoints/fish -> not identifiable); group effects are fixed with weakly-informative priors.

## (A) Effect on latent progression (HMM, log-odds of advancing vs sham)

| contrast | median log-OR | 94% HDI | P(>0) |
|---|---|---|---|
| se_core vs sham | +0.41 | [-0.58, +1.26] | 0.789 |
| se_vehicle vs sham | +0.69 | [-0.29, +1.58] | 0.914 |
| se_vpa vs sham | -0.35 | [-1.50, +0.82] | 0.282 |
| **se_vpa vs se_vehicle** | **-1.03** | [-2.22, +0.30] | P(<0)=0.948 |

## (B) Effect on the actual outcome (logistic, odds ratios vs sham)

| contrast | OR | 94% HDI |
|---|---|---|
| se_core vs sham | 2.97 | [0.72, 7.20] |
| se_vehicle vs sham | 10.99 | [2.23, 27.30] |
| se_vpa vs sham | 1.75 | [0.38, 4.54] |
| **se_vpa vs se_vehicle (VPA)** | **0.16** | [0.03, 0.39] |

- **VPA is credibly protective on the outcome: OR 0.16 (94% HDI [0.03, 0.39]), P(OR<1) = 1.000.**
- SE is harmful vs sham (se_vehicle OR 11.0), and VPA pulls the odds back down toward sham.

## Figures
- `outputs/v3_outcome_odds_ratios.png` (forest plot)
- `outputs/v3_vpa_posterior.png` (VPA protective-effect posterior)