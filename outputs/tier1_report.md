# RELAPSE Tier 1 - Gaussian HMM results

Standard 4-state Gaussian HMM (`hmmlearn`), diagonal covariance, trained on 70% of fish (fish-level split), validated on the held-out 30%. The 5 LFP features are the only model inputs; all TRUTH columns are used only to score.

## (a) Hidden-state recovery

- **Held-out accuracy: 99.4%**  |  all-fish accuracy: 99.1%
- States were severity-aligned to ground truth by ranking emission means.

Confusion matrix (held-out, rows = true state, cols = predicted):

| true \ pred | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 82 | 0 | 0 | 0 |
| **1** | 0 | 34 | 1 | 0 |
| **2** | 0 | 0 | 20 | 0 |
| **3** | 0 | 0 | 0 | 31 |

## (b) Early prediction (before first seizure)

- **ROC-AUC (held-out): 0.879**
- Accuracy at Youden-optimal threshold (0.90): **91.7%**
- Epileptic held-out fish flagged BEFORE their seizure: **10/10**
- **Mean lead-time: 3.4 h** (median 2.0 h) before first seizure
- Risk = filtered P(state>=2) from an ONLINE forward pass (uses only observations up to t -> no leakage from the future seizure).

## (c) Microplastic effect on progression

- Advance-rate P(state increases per 2 h step): control **0.163** vs microplastic **0.333** = **2.05x** faster progression under microplastic.
- Fraction of fish reaching seizure state 3: control 25% vs microplastic 62%.
- Direction matches the planted sanity check (microplastic -> faster progression, higher incidence).

## Learned emission means (severity-aligned, original units)

| state | lfp_spike_amp_uV | lfp_spike_freq_hz | lfp_hfo_power | lfp_signal_entropy | lfp_line_length |
|---|---|---|---|---|---|
| 0 | 38.81 | 0.39 | 0.10 | 0.85 | 118.10 |
| 1 | 58.62 | 0.88 | 0.18 | 0.77 | 181.02 |
| 2 | 92.66 | 1.92 | 0.32 | 0.65 | 303.49 |
| 3 | 153.49 | 3.60 | 0.57 | 0.48 | 544.91 |

## Figures

- `outputs/tier1_state_trajectories.png`
- `outputs/tier1_risk_over_time.png`
- `outputs/tier1_transition_matrices.png`
- `outputs/tier1_confusion_matrix.png`