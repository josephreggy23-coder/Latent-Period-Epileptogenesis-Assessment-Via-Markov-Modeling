# ECHO V3 — Full Results Report
### SE → PTZ re-challenge sensitization model · Hidden Markov classification

> Analysis run treating `ECHO_V3_synthetic_data.xlsx` as the real experiment. A Hidden Markov Model was trained on the 5 silent-period LFP features ONLY (4 h & 20 h post-status-epilepticus). All ground-truth / outcome columns were withheld and used only to score the model. N = 128 larvae across 4 arms (sham, se_core, se_vehicle, se_vpa).

---
## 0. Executive summary

1. **The model works.** It recovers the planted latent disease state at **97.4%** on held-out fish — so its state calls are trustworthy.
2. **Sensitization is only partly predictable from silent-period LFP** (ROC-AUC **0.73**). This is a real biological ceiling, not a model failure: **all 14 missed sensitizers showed no latent elevation at all** — they looked normal until re-challenge.
3. **VPA is protective — decisively.** On the actual outcome, VPA cuts the odds of sensitization vs vehicle to **OR 0.16** (94% HDI [0.03, 0.39], P(protective) = 1.000).
4. **SE itself is harmful** (se_vehicle vs sham OR **11.0**), and VPA pulls the risk back down toward sham.

---
## 1. Model & methodology

- **Inputs (only these):** `lfp_discharge_amp_uV`, `lfp_discharge_freq_hz`, `lfp_delta_power_norm`, `lfp_ied_interval_s`, `lfp_line_length`.
- **Sequence:** each fish = 2 observations (4 h, 20 h post-SE), the latent/silent window.
- **Tier 1:** Gaussian HMM (`hmmlearn`), diagonal covariance, 15 restarts. Model selection by BIC chose **K = 3 states** (BIC: K2=2156, K3=2110; K≥4 would not fit — the data only support 3 latent states).
- **Split:** 70/30 at the FISH level, stratified by group × outcome; features standardized on train only.
- **Tier 2:** Bayesian models via NUTS (`numpyro`) — a progression-hazard HMM with group covariates (latent mechanism) and a logistic outcome model (clinical effect). Per-fish random effects omitted (2 timepoints → not identifiable); group fixed effects estimated on the full cohort.

---
## 2. Hidden-state recovery (does the model find the planted states?)

- **Held-out accuracy 97.4%**, all-fish 97.7%.
- Per-timepoint: **4 h 99.2%**, 20 h 96.1% (20 h is harder — that is where the rare state 2 appears).
- Confusion (held-out, 78 timepoints; rows = true, cols = predicted):

| true \ pred | 0 | 1 | 2 |
|---|---|---|---|
| **0** | 57 | 0 | 0 |
| **1** | 2 | 18 | 0 |
| **2** | 0 | 0 | 1 |

*(only 2 off-diagonal errors; see `v3_confusion_matrix.png`.)*

**Verdict:** state classification is reliable — downstream results rest on solid footing.

---
## 3. Sensitization classification — what the model called, per fish

Risk score = model-inferred P(latent state ≥ 1). Threshold = Youden-optimal.

### 3.1 Overall confusion (all 128 fish)

| | predicted NOT sensitized | predicted sensitized |
|---|---|---|
| **actually NOT sensitized** | 75 (TN) | 13 (FP) |
| **actually sensitized** | 14 (FN) | 26 (TP) |

- **Accuracy 78.9%**, Sensitivity 65%, Specificity 85%, PPV 67%, NPV 84%.
- **ROC-AUC 0.735** (all fish) / 0.750 (held-out). From the **4 h timepoint alone** AUC drops to **0.617** — most of the predictive signal accrues by 20 h.

### 3.2 Accuracy by arm

| group | n | actual sensitized | model-predicted | accuracy |
|---|---|---|---|---|
| sham | 32 | 2 | 7 | 84% |
| se_core | 39 | 13 | 13 | 85% |
| se_vehicle | 27 | 18 | 15 | 67% |
| se_vpa | 30 | 7 | 4 | 77% |

### 3.3 Where the model fails — and why it is not the model's fault

- **14 false negatives** (sensitized but called normal): **every one was inferred at latent state 0** — i.e. their silent-period LFP looked healthy and they only converted at re-challenge. By arm: se_vehicle 6, se_vpa 5, se_core 3. These are biologically unpredictable from the silent window — the irreducible error.
- **13 false positives** (elevated latent state but did not sensitize): sham 5, se_core 3, se_vehicle 3, se_vpa 2. These fish showed transient LFP elevation that did not convert — the noisy upper tail.

---
## 4. Treatment effect — is VPA protective?  (the headline question)

### 4.1 Tier 1 point estimates (inferred latent progression, 4 h→20 h)

| group | n | mean state @20h | inferred progression | actual sensitization |
|---|---|---|---|---|
| sham | 32 | 0.28 | 12% | 6% |
| se_core | 39 | 0.36 | 23% | 33% |
| se_vehicle | 27 | 0.67 | 33% | 67% |
| se_vpa | 30 | 0.20 | 13% | 23% |

Inferred progression ordering **se_vehicle > se_core > se_vpa ≈ sham**; VPA shows a **60% relative reduction** vs vehicle.

### 4.2 Tier 2 posterior — full uncertainty

**(A) Effect on latent progression (log-odds of advancing vs sham):**

| contrast | median log-OR | 94% HDI |
|---|---|---|
| se_core vs sham | +0.41 | [-0.58, +1.26] |
| se_vehicle vs sham | +0.69 | [-0.29, +1.58] |
| se_vpa vs sham | -0.35 | [-1.50, +0.82] |
| **se_vpa vs se_vehicle** | **-1.03** | [-2.22, +0.30] · P(protective)=0.948 |

**(B) Effect on the actual outcome (odds ratios):**

| contrast | OR | 94% HDI |
|---|---|---|
| se_core vs sham | 2.97 | [0.72, 7.20] |
| se_vehicle vs sham | 10.99 | [2.23, 27.30] |
| se_vpa vs sham | 1.75 | [0.38, 4.54] |
| **se_vpa vs se_vehicle (VPA effect)** | **0.16** | [0.03, 0.39] · P(OR<1)=1.000 |

> **VPA reduces the odds of sensitization by ~84% vs vehicle, and the entire 94% credible interval lies below 1.** The latent-mechanism model agrees in direction (P 0.95) but is less certain — expected, since the 2-timepoint latent signal is noisier than the outcome.

---
## 5. Caveats & limitations

- **Probabilistic outcome ceiling:** sensitization is only partly written into the silent-period LFP, so AUC ≈ 0.74 is near the achievable maximum here, not a tuning failure.
- **Only 2 timepoints/fish:** limits dynamic modeling and rules out per-fish random effects; denser sampling would sharpen both prediction and the latent-progression posterior.
- **State 2 is rare** (6/256 timepoints): its emission/transition estimates are the least certain part of the HMM.
- **Held-out threshold metrics are small-sample-noisy** (39 test fish); AUC (threshold-free) is the more stable read.
- Synthetic data with a planted truth — real larvae will be noisier; treat these numbers as a best case for the pipeline.

---
## 6. Bottom line for the experiment

- The HMM pipeline is **validated**: it recovers latent states almost perfectly and detects the planted treatment structure.
- **VPA's protective effect is robust and credible** on the outcome (OR ≈ 0.16); this is the result to carry into the real study.
- For prediction, expect to **flag the fish that telegraph progression**, while a subset of sensitizers will remain invisible in the silent window — budget for that, and add timepoints if early prediction is the goal.

---
## 7. Files

- **Per-fish classifications:** `outputs/v3_per_fish_classification.csv`
- **Reports:** `outputs/v3_tier1_report.md`, `outputs/v3_tier2_report.md`, this file
- **Figures:** `v3_bic_selection.png`, `v3_confusion_matrix.png`, `v3_group_effects.png`, `v3_risk_distribution.png`, `v3_transition_matrices.png`, `v3_roc.png`, `v3_outcome_odds_ratios.png`, `v3_vpa_posterior.png`

## Appendix A — per-fish classification (all 128 larvae)

Sorted by arm, then model risk (high → low). State 0 = normal, 1 = elevated, 2 = high. `FLAG` = model predicted sensitization. Result: TP/TN correct; FP = false alarm; FN = missed sensitizer (also in `outputs/v3_per_fish_summary.csv`).

| Fish | Arm | St@4h | St@20h | Risk% | Model | Actual | Result | Set |
|---|---|---|---|---|---|---|---|---|
| F002 | Sham | 1 | 1 | 100.0 | FLAG | resistant | FP (false alarm) | train |
| F011 | Sham | 1 | 1 | 100.0 | FLAG | resistant | FP (false alarm) | train |
| F012 | Sham | 1 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | test |
| F017 | Sham | 1 | 2 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F021 | Sham | 0 | 1 | 100.0 | FLAG | resistant | FP (false alarm) | train |
| F029 | Sham | 1 | 1 | 100.0 | FLAG | resistant | FP (false alarm) | train |
| F032 | Sham | 0 | 1 | 99.9 | FLAG | resistant | FP (false alarm) | test |
| F030 | Sham | 0 | 0 | 69.9 | clear | resistant | TN (correctly cleared) | test |
| F010 | Sham | 0 | 1 | 55.6 | clear | resistant | TN (correctly cleared) | train |
| F008 | Sham | 0 | 0 | 22.8 | clear | resistant | TN (correctly cleared) | train |
| F005 | Sham | 0 | 0 | 0.6 | clear | resistant | TN (correctly cleared) | train |
| F007 | Sham | 0 | 0 | 0.5 | clear | resistant | TN (correctly cleared) | train |
| F016 | Sham | 0 | 0 | 0.3 | clear | resistant | TN (correctly cleared) | test |
| F020 | Sham | 0 | 0 | 0.3 | clear | resistant | TN (correctly cleared) | test |
| F026 | Sham | 0 | 0 | 0.3 | clear | resistant | TN (correctly cleared) | test |
| F024 | Sham | 0 | 0 | 0.2 | clear | resistant | TN (correctly cleared) | train |
| F003 | Sham | 0 | 0 | 0.1 | clear | resistant | TN (correctly cleared) | train |
| F004 | Sham | 0 | 0 | 0.1 | clear | resistant | TN (correctly cleared) | train |
| F001 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F006 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F009 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F013 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F014 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F015 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F018 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F019 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F022 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F023 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F025 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F027 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F028 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F031 | Sham | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F033 | SE only | 1 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F039 | SE only | 0 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F043 | SE only | 0 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F044 | SE only | 1 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | test |
| F048 | SE only | 1 | 2 | 100.0 | FLAG | resistant | FP (false alarm) | train |
| F052 | SE only | 1 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F059 | SE only | 1 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | test |
| F062 | SE only | 0 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | test |
| F055 | SE only | 0 | 1 | 99.9 | FLAG | sensitized | TP (correctly flagged) | train |
| F067 | SE only | 0 | 1 | 99.8 | FLAG | sensitized | TP (correctly flagged) | test |
| F060 | SE only | 0 | 1 | 99.6 | FLAG | resistant | FP (false alarm) | test |
| F069 | SE only | 0 | 1 | 99.5 | FLAG | sensitized | TP (correctly flagged) | train |
| F046 | SE only | 0 | 1 | 98.8 | FLAG | resistant | FP (false alarm) | test |
| F051 | SE only | 0 | 0 | 46.8 | clear | resistant | TN (correctly cleared) | train |
| F053 | SE only | 0 | 0 | 16.2 | clear | resistant | TN (correctly cleared) | train |
| F054 | SE only | 0 | 0 | 6.6 | clear | resistant | TN (correctly cleared) | test |
| F057 | SE only | 0 | 0 | 1.8 | clear | resistant | TN (correctly cleared) | train |
| F034 | SE only | 0 | 0 | 0.4 | clear | resistant | TN (correctly cleared) | train |
| F036 | SE only | 0 | 0 | 0.3 | clear | resistant | TN (correctly cleared) | train |
| F068 | SE only | 0 | 0 | 0.3 | clear | resistant | TN (correctly cleared) | train |
| F035 | SE only | 0 | 0 | 0.2 | clear | resistant | TN (correctly cleared) | train |
| F038 | SE only | 0 | 0 | 0.2 | clear | resistant | TN (correctly cleared) | train |
| F063 | SE only | 0 | 0 | 0.2 | clear | sensitized | FN (missed) | train |
| F070 | SE only | 0 | 0 | 0.2 | clear | resistant | TN (correctly cleared) | train |
| F047 | SE only | 0 | 0 | 0.1 | clear | resistant | TN (correctly cleared) | train |
| F064 | SE only | 0 | 0 | 0.1 | clear | resistant | TN (correctly cleared) | test |
| F037 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F040 | SE only | 0 | 0 | 0.0 | clear | sensitized | FN (missed) | train |
| F041 | SE only | 0 | 0 | 0.0 | clear | sensitized | FN (missed) | train |
| F042 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F045 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F049 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F050 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F056 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F058 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F061 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F065 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F066 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F071 | SE only | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F102 | SE+Vehicle | 0 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F105 | SE+Vehicle | 0 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F108 | SE+Vehicle | 1 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F116 | SE+Vehicle | 1 | 2 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F117 | SE+Vehicle | 1 | 1 | 100.0 | FLAG | resistant | FP (false alarm) | train |
| F118 | SE+Vehicle | 0 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F119 | SE+Vehicle | 1 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F123 | SE+Vehicle | 1 | 1 | 100.0 | FLAG | resistant | FP (false alarm) | test |
| F124 | SE+Vehicle | 1 | 1 | 100.0 | FLAG | resistant | FP (false alarm) | train |
| F126 | SE+Vehicle | 1 | 2 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F127 | SE+Vehicle | 1 | 2 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F128 | SE+Vehicle | 1 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | test |
| F120 | SE+Vehicle | 0 | 1 | 99.9 | FLAG | sensitized | TP (correctly flagged) | train |
| F113 | SE+Vehicle | 0 | 1 | 97.5 | FLAG | sensitized | TP (correctly flagged) | test |
| F106 | SE+Vehicle | 0 | 1 | 94.2 | FLAG | sensitized | TP (correctly flagged) | test |
| F112 | SE+Vehicle | 0 | 0 | 40.3 | clear | resistant | TN (correctly cleared) | test |
| F104 | SE+Vehicle | 0 | 0 | 7.2 | clear | sensitized | FN (missed) | test |
| F111 | SE+Vehicle | 0 | 0 | 0.3 | clear | sensitized | FN (missed) | train |
| F103 | SE+Vehicle | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F107 | SE+Vehicle | 0 | 0 | 0.0 | clear | sensitized | FN (missed) | train |
| F109 | SE+Vehicle | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F110 | SE+Vehicle | 0 | 0 | 0.0 | clear | sensitized | FN (missed) | test |
| F114 | SE+Vehicle | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F115 | SE+Vehicle | 0 | 0 | 0.0 | clear | sensitized | FN (missed) | train |
| F121 | SE+Vehicle | 0 | 0 | 0.0 | clear | sensitized | FN (missed) | train |
| F122 | SE+Vehicle | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F125 | SE+Vehicle | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F080 | SE+VPA | 1 | 1 | 100.0 | FLAG | sensitized | TP (correctly flagged) | train |
| F088 | SE+VPA | 1 | 2 | 100.0 | FLAG | resistant | FP (false alarm) | test |
| F093 | SE+VPA | 0 | 1 | 98.5 | FLAG | resistant | FP (false alarm) | train |
| F084 | SE+VPA | 0 | 1 | 96.1 | FLAG | sensitized | TP (correctly flagged) | train |
| F098 | SE+VPA | 0 | 1 | 68.2 | clear | resistant | TN (correctly cleared) | train |
| F076 | SE+VPA | 0 | 0 | 61.8 | clear | resistant | TN (correctly cleared) | test |
| F087 | SE+VPA | 0 | 0 | 6.8 | clear | resistant | TN (correctly cleared) | train |
| F099 | SE+VPA | 0 | 0 | 2.7 | clear | resistant | TN (correctly cleared) | train |
| F089 | SE+VPA | 0 | 0 | 1.5 | clear | resistant | TN (correctly cleared) | train |
| F083 | SE+VPA | 0 | 0 | 1.0 | clear | sensitized | FN (missed) | test |
| F082 | SE+VPA | 0 | 0 | 0.9 | clear | resistant | TN (correctly cleared) | test |
| F077 | SE+VPA | 0 | 0 | 0.5 | clear | resistant | TN (correctly cleared) | train |
| F092 | SE+VPA | 0 | 0 | 0.3 | clear | resistant | TN (correctly cleared) | test |
| F090 | SE+VPA | 0 | 0 | 0.2 | clear | resistant | TN (correctly cleared) | test |
| F079 | SE+VPA | 0 | 0 | 0.1 | clear | resistant | TN (correctly cleared) | train |
| F101 | SE+VPA | 0 | 0 | 0.1 | clear | resistant | TN (correctly cleared) | train |
| F072 | SE+VPA | 0 | 0 | 0.0 | clear | sensitized | FN (missed) | train |
| F073 | SE+VPA | 0 | 0 | 0.0 | clear | sensitized | FN (missed) | train |
| F074 | SE+VPA | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F075 | SE+VPA | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F078 | SE+VPA | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F081 | SE+VPA | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F085 | SE+VPA | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F086 | SE+VPA | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F091 | SE+VPA | 0 | 0 | 0.0 | clear | sensitized | FN (missed) | test |
| F094 | SE+VPA | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | test |
| F095 | SE+VPA | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F096 | SE+VPA | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |
| F097 | SE+VPA | 0 | 0 | 0.0 | clear | sensitized | FN (missed) | train |
| F100 | SE+VPA | 0 | 0 | 0.0 | clear | resistant | TN (correctly cleared) | train |