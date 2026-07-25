# ECHO V3 - Tier 1 HMM results (treated as the real experiment)

SE -> PTZ re-challenge sensitization model. A 3-state Gaussian HMM was trained on the 5 silent-period LFP features ONLY (4 h & 20 h post-SE); all TRUTH/outcome columns were held out for scoring. Below is what the model classified and how well it matches the planted truth.

## Model selection

| K states | BIC | logL |
|---|---|---|
| 2 | 2156.3 | -1018.6 |
| 3 | 2109.7 | -956.4 |

BIC favors K=3; we report K=3 to match the 3 planted latent states (state 2 is rare, 6/256 timepoints).

## (a) Hidden-state recovery

- **Held-out accuracy 97.4%** (all-fish 97.7%).
- Confusion (held-out, rows=true, cols=pred):

| true\pred | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 57 | 0 | 0 |
| 1 | 2 | 18 | 0 |
| 2 | 0 | 0 | 1 |

## (b) Sensitization classification (the clinical target)

Risk = model P(inferred latent state >= 1). NOTE the outcome is a SEPARATE re-challenge event, only probabilistically tied to silent-period state, so perfect prediction is impossible by construction.

| evaluation | AUC | acc | sens | spec | PPV | NPV |
|---|---|---|---|---|---|---|
| all fish | 4h+20h | 0.735 | 0.79 | 0.65 | 0.85 | 0.67 | 0.84 |
| held-out | 4h+20h | 0.750 | 0.72 | 0.83 | 0.67 | 0.53 | 0.90 |
| all fish | 4h ONLY (early) | 0.617 | 0.73 | 0.33 | 0.91 | 0.62 | 0.75 |

All-fish confusion @ threshold: TN=75 FP=13 FN=14 TP=26 (n=128, 40 sensitized).

## (c) Group & VPA treatment effect

| group | n | mean state@20h | inferred progression | actual sensitization |
|---|---|---|---|---|
| sham | 32 | 0.28 | 12.5% | 6.2% |
| se_core | 39 | 0.36 | 23.1% | 33.3% |
| se_vehicle | 27 | 0.67 | 33.3% | 66.7% |
| se_vpa | 30 | 0.20 | 13.3% | 23.3% |

- **VPA is protective:** inferred progression se_vpa 13.3% vs se_vehicle 33.3% = **60% relative reduction**, back down to sham (12.5%).
- Ordering recovered by the model: se_vehicle > se_core > se_vpa ≈ sham — matches the planted SE-harm + VPA-rescue design.

## Emission means (severity-aligned, standardized units)

| state | lfp_discharge_amp_uV | lfp_discharge_freq_hz | lfp_delta_power_norm | lfp_ied_interval_s | lfp_line_length |
|---|---|---|---|---|---|
| 0 | -0.32 | -0.39 | -0.36 | +0.38 | -0.40 |
| 1 | +0.71 | +0.85 | +0.91 | -1.09 | +1.07 |
| 2 | +3.14 | +3.93 | +2.91 | -2.06 | +2.69 |

(values are standardized; all features rise with state except `lfp_ied_interval_s` which falls — shorter inter-discharge interval = worse.)

## Files
- `outputs/v3_per_fish_classification.csv` (every fish's inferred states + risk + predicted vs actual sensitization)
- plots: `v3_bic_selection.png`, `v3_confusion_matrix.png`, `v3_group_effects.png`, `v3_risk_distribution.png`, `v3_transition_matrices.png`, `v3_roc.png`