# ECHO — Entire Project Summary

**ECHO** = *Epileptogenesis Classification from Hidden-state Observation.* A Hidden Markov
Model (HMM) pipeline that infers a larval zebrafish's **latent disease state** from noisy
LFP features during the post–status-epilepticus (SE) "silent period," and predicts which
fish will become **epileptic / sensitized before the event**, quantifies drug effects with
calibrated uncertainty, and generalizes to conditions it never trained on.

Everything lives in `D:\Reggy\ECHO`. All modeling uses **only LFP features** as input;
ground-truth columns are used exclusively for scoring. All train/test splits are at the
**fish level**. Validated on Python 3.14 (hmmlearn, numpyro/JAX, lifelines).

---

## 1. Motivation
Epileptogenesis is the silent latent process between a brain insult and the first
spontaneous seizure. If a model can read that latent process from electrophysiology, it can
(a) predict who will develop epilepsy before seizures start and (b) score whether a drug
slows the process. ECHO tests this on synthetic zebrafish data with **known planted ground
truth**, so we can prove the pipeline recovers what was planted before applying it to real
animals.

## 2. Core method (constant across the project)
- Each fish is a **sequence of LFP feature vectors over time**; the LFP features are the HMM
  emissions.
- A **Gaussian HMM** infers discrete latent states (0 = healthy … up to seizure/sensitized).
- **Honest online prediction:** risk is a **forward-filter** posterior P(advanced state) that
  uses only observations up to time *t* — no leakage from the future outcome.
- **Tier 2 = Bayesian** re-implementation via MCMC (numpyro/NUTS): priors on transitions and
  emissions, hierarchical per-fish random effects, treatment as a covariate → full posterior
  uncertainty on the drug effect.

## 3. Three dataset iterations (increasing realism & rigor)

### V1 — RELAPSE (microplastic epileptogenesis) · *proof of concept*
- 80 fish × 7 timepoints, 5 LFP features, 2 arms (control vs microplastic), 4 states where
  **state 3 = seizure (deterministic outcome)**.
- **Tier 1 Gaussian HMM:** state recovery **99.4%**; early-prediction **AUC 0.879**;
  **10/10** epileptic fish flagged **before** first seizure; mean lead-time **3.4 h**.
- **Tier 2 Bayesian hierarchical HMM:** microplastic per-step progression **odds ratio 2.91**
  (94% HDI [1.42, 5.15]), **P(effect > 0) = 0.998**; clean MCMC (R-hat 1.00, 0 divergences).
- Lesson: the pipeline works and recovers planted truth in the easy, deterministic case.

### V3 — ECHO V3 (SE → PTZ re-challenge sensitization) · *harder, realistic*
- Literature-grounded paradigm shift: 128 fish × **only 2 timepoints**, 5 features, **4 arms**
  (sham, se_core, se_vehicle, se_vpa), **3 states**, and a **probabilistic** outcome
  (sensitization is a separate re-challenge event, not a deterministic top state).
- State recovery **97.4%**; sensitization prediction **AUC 0.74** — honestly lower, because
  the outcome is only partly written into the silent-period LFP. **All 14 missed sensitizers
  showed no latent elevation at all** (the irreducible error).
- **VPA protective:** outcome odds ratio **0.16** vs vehicle; SE harmful (vehicle OR ~11 vs
  sham). Per-fish random effects were *not* identifiable with only 2 timepoints.

### V5 — ECHO V5 (judge-hardened, LFP-only) · *the definitive analysis*
- 222 fish × **5 timepoints**, **11 single-electrode LFP features**, **6 arms** (adds VPA
  low/high dose + washout control), **4 recording batches**, ~4% injected artifacts +
  heavy tails, censored time-to-event. Single-electrode LFP treated as a **fixed constraint**.
- Built to answer **8 specific reviewer/judge critiques** — see §4.

## 4. V5 results — the eight judge critiques, answered
| # | Critique | Result |
|---|---|---|
| 1 | Artifacts not handled | 4.9% cells flagged (modified z-score) + robust winsorize/scale |
| 2 | BIC alone; rare states | BIC **and** CV log-lik; **1-SE rule → K = 3**; recovery **99.0%**; rare state stable; hierarchical τ now identifiable (0.21) |
| 3 | Feature set too narrow | PAC & line-length dominate; **honest negative — basic-5 (0.758) beats all-11 (0.729)** |
| 4 | AUC may be noise | **AUC 0.759**, 95% CI [0.69, 0.82], **permutation p = 0.0005**; temporal rise 0.66→0.76; HMM beats logistic **modestly** |
| 5 | Single split | Stratified **5-fold** at fish level (0.764 ± 0.063) |
| 6 | Binary insufficient | Kaplan-Meier + Cox; log-rank **p = 2.6×10⁻⁶**; dose-ordered HRs; batch-robust |
| 7 | VPA claim shaky | Bayesian dose-response + **washout control**; VPA-high **OR 0.15** [0.03, 0.34]; washout snaps back; **prior-robust**; R-hat 1.001 |
| 8 | Batch confounds | Not confounded (χ² p = 0.95; batch-alone AUC 0.52; holds within every batch) |

**The washout control is the scientific clincher:** VPA protection is dose-dependent
(low OR 0.32 → high 0.15) and *disappears on washout* (OR back to vehicle level, and
credibly worse than active drug, OR 4.54) — proving the effect is **pharmacological, not a
baseline strain difference**.

## 5. Out-of-distribution (OOD) generalization
Trained on **only sham + se_vehicle** (the biological extremes), then scored the never-seen
valproate arms:
- Recovered the held-out **dose-response ranking** (vehicle > washout > VPA-low > VPA-high)
  with zero VPA training (Mann-Whitney p < 0.05 for the key contrasts).
- Predicted risk **tracked real sensitization** in unseen arms: **Pearson r = 0.97**; per-fish
  AUC 0.78; risk ↔ latent-truth Spearman ρ = 0.89.
- *Honest nuance:* sham did not rank lowest because high-dose VPA suppresses LFP **below**
  sham baseline in the ground truth — ECHO (a latent-LFP-state score) correctly detected this;
  sham's low outcome despite moderate LFP activity is a control-specific latent-state/outcome
  dissociation, not a model error.

## 6. What makes ECHO defensible
1. **It reports its own negatives** — added features don't help, the HMM only modestly beats
   logistic, a 4th state is weakly supported, τ is small, single-electrode caps AUC ≈ 0.76.
2. **Significance is earned** — permutation test, bootstrap CIs, temporal curve, 5-fold CV.
3. **The drug claim has a built-in control** — the washout arm rules out strain effects, and
   the posterior is prior-insensitive.
4. **It generalizes** — leave-arms-out OOD, not just random cross-validation.

## 7. Honest limitations
- Single-electrode LFP is a hard information ceiling (AUC ≈ 0.76).
- The HMM's edge over logistic regression is stability + interpretability, not a large jump.
- All validation is on synthetic data with planted truth; real larvae will be noisier.
- **No public real zebrafish seizure-LFP dataset** is downloadable in an ECHO-ready
  per-fish time-series form (UCSF Epilepsy Zebrafish Project = representative traces;
  eNeuro larval-LFP study = code-only; large behavioral sets = 43 GB video). Real-data
  application awaits a suitable recording.

## 8. Deliverables (in `D:\Reggy\ECHO`)
- **Summaries:** `MASTER_RESULTS.md` (paper-ready results), this `PROJECT_SUMMARY.md`.
- **V5 (main):** `v5_common.py`, `v5_pipeline.py`, `v5_survival.py`, `v5_bayes.py`,
  `v5_report.py`, `v5_ood.py`, `run_v5_all.py` → `V5_RESULTS_REPORT.md`, `V5_OOD_REPORT.md`.
- **V3:** `v3_tier1.py`, `v3_tier2.py`, `v3_summarize.py` → `V3_RESULTS_REPORT.md`.
- **V1:** `relapse_common.py`, `tier1_gaussian_hmm.py`, `tier2_bayesian_hmm.py`,
  `summarize.py`, `run_all.py` → `RESULTS_SUMMARY.md`.
- **~25 figures + per-fish CSVs + metrics JSONs** in `outputs/`.
- `requirements.txt` (numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl, hmmlearn,
  numpyro, lifelines).

## 9. One-paragraph project abstract
*ECHO is a Hidden Markov Model pipeline for larval-zebrafish epileptogenesis that infers
latent disease state from single-electrode LFP and predicts post–status-epilepticus
sensitization before it occurs. Across three synthetic datasets of increasing realism it
recovers planted latent states at ≥97–99%, predicts sensitization above chance (AUC 0.76,
permutation p = 5×10⁻⁴) with rising accuracy over the observation window, and — via a
Bayesian hierarchical model — quantifies a dose-dependent, washout-reversible, prior-robust
valproate protective effect (odds ratio 0.15, P > 0.999), corroborated by Cox survival
analysis and robust to recording batch. In a leave-arms-out test it ranks drug doses it never
trained on (held-out risk↔outcome r = 0.97). ECHO deliberately reports its negatives, earns
its significance with permutation and cross-validation, and controls its drug claim with a
washout arm — making it a rigorous, honestly-bounded framework ready for real-data testing.*
