# RELAPSE — Hidden Markov Model pipeline for zebrafish epileptogenesis

Infer each fish's hidden disease state from noisy LFP features and predict which
fish become **epileptic before their first seizure**, on synthetic data with a
**known planted ground truth**. Two tiers: a standard Gaussian HMM (Tier 1) and a
Bayesian hierarchical HMM fit with MCMC (Tier 2).

## Install

```
pip install -r requirements.txt
```

Validated on **Python 3.14.4 (Windows)**. `hmmlearn` compiles from source if no
wheel is available (needs a C compiler); `numpyro` pulls `jax`/`jaxlib`
(cp314 Windows wheels exist).

## Run

```
python run_all.py            # Tier 1 -> Tier 2 -> consolidated summary
```

or individually:

```
python tier1_gaussian_hmm.py     # foundation; writes artifacts Tier 2 reuses
python tier2_bayesian_hmm.py     # Bayesian upgrade (needs Tier 1 artifacts)
python summarize.py              # consolidated RESULTS_SUMMARY.md
```

## What it does

**Inputs (the ONLY model inputs):** the 5 LFP features
`lfp_spike_amp_uV, lfp_spike_freq_hz, lfp_hfo_power, lfp_signal_entropy,
lfp_line_length`. Each fish is a sequence of 7 observations over hours 0–12.

**Never used as input** (validation only): `hidden_state_TRUTH`,
`destined_path_TRUTH`, `became_epileptic`, `first_seizure_hours`,
`max_state_reached`.

**Split:** 70/30 at the **fish level** (a fish's timepoints never straddle the
split), stratified by group × outcome. Features standardized on TRAIN only.

### Tier 1 — Gaussian HMM (`hmmlearn`)
4-state diagonal-covariance HMM, 12 random restarts. Validates:
- **(a) State recovery** — Viterbi states severity-aligned to truth; accuracy + confusion matrix.
- **(b) Early prediction** — online **forward-filter** risk `P(state≥2 | obs up to t)` (no future leakage); ROC-AUC, accuracy, lead-time before first seizure.
- **(c) Microplastic effect** — control vs microplastic transition matrices & per-step advance rate.

### Tier 2 — Bayesian hierarchical HMM (`numpyro`/NUTS)
States marginalized via the forward algorithm; NUTS samples the continuous posterior.
- Single-step progression "birth chain" (matches the truth: only Δ∈{0,+1}), state 3 absorbing.
- Per-step hazard `p_{i,s} = sigmoid(α_s + β_mp·mp_i + u_i)` with hierarchical `u_i ~ N(0, τ)`.
- Priors on emissions (centered at Tier-1 estimates) and on the transition dynamics.
- Reports the **full posterior of the microplastic effect** (β_mp, odds ratio exp(β_mp), P>0), τ, and per-step hazards with credible intervals.
- Held-out early prediction integrates each fish's random effect (Gauss-Hermite), updated online by its own early data.

## Outputs (`outputs/`)

| file | what |
|---|---|
| `tier1_report.md`, `tier2_report.md` | per-tier results |
| `comparison_tier1_vs_tier2.md` | head-to-head |
| `../RESULTS_SUMMARY.md` | consolidated summary (top level) |
| `tier1_state_trajectories.png` | inferred vs truth states, example fish |
| `tier1_risk_over_time.png` | predicted risk, epileptic vs resilient |
| `tier1_transition_matrices.png` | control vs microplastic transitions |
| `tier1_confusion_matrix.png` | state-recovery confusion matrix |
| `tier2_microplastic_posterior.png` | posterior of the microplastic effect |
| `tier2_advance_hazard.png` | posterior per-step hazards, control vs mp |
| `tier2_roc_compare.png` | Tier 1 vs Tier 2 ROC |
| `tier1_artifacts.npz`, `*_metrics.json` | machine-readable artifacts/metrics |

## Headline result

States recovered at **99.4%** (held-out); **10/10** held-out epileptic fish
flagged before seizure (AUC **0.879**, mean lead-time **3.4 h**); microplastic
shows **~2–3× faster** progression — Tier 2 posterior odds ratio **2.91**
(94% HDI [1.42, 5.15]), **P(effect>0)=0.998**. The model recovers what was
planted.
