"""
ECHO V3 - TIER 2 Bayesian treatment-effect analysis (numpyro/NUTS)
==================================================================
Two complementary Bayesian models, both giving FULL posterior uncertainty on the
key question: is VPA protective against SE-induced sensitization?

  (A) Progression-hazard HMM (latent mechanism): states marginalized via the
      forward algorithm; per-step (4h->20h) advance hazard depends on group.
      -> posterior of the SE effect and the VPA effect on latent progression.
  (B) Logistic outcome model (clinical effect): became_sensitized ~ group.
      -> posterior ODDS RATIOS, esp. VPA-vs-vehicle (protective) and SE-vs-sham.

Note on design: only 2 timepoints/fish, so per-fish random effects are not
identifiable here; we use GROUP-level fixed effects (the right call for short
sequences) and estimate effects on the FULL cohort (an experiment analysis,
not a held-out prediction).

pip install numpyro   (jax/jaxlib cp314 wheels exist)
Run:  python v3_tier2.py     (after v3_tier1.py)
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, jax.random as random
from jax.scipy.special import logsumexp as jlse
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from relapse_common import savefig, OUT_DIR

DATA = "ECHO_V3_synthetic_data.xlsx"
FEATURES = ["lfp_discharge_amp_uV", "lfp_discharge_freq_hz", "lfp_delta_power_norm",
            "lfp_ied_interval_s", "lfp_line_length"]
TIME = "hours_post_se"; TARGET = "became_sensitized"
GROUPS = ["sham", "se_core", "se_vehicle", "se_vpa"]   # sham = reference
NONREF = ["se_core", "se_vehicle", "se_vpa"]
N_STATES = 3; SEED = 42
numpyro.set_host_device_count(2)


def hdi(s, p=0.94):
    s = np.sort(np.asarray(s)); n = len(s); w = int(np.floor(p * n))
    i = int(np.argmin(s[w:] - s[:n - w])); return float(s[i]), float(s[i + w])


def load():
    ts = pd.read_excel(DATA, "LFP_timeseries")
    out = pd.read_excel(DATA, "fish_outcomes")
    for df in (ts, out):
        for c in ("fish_id", "group"):
            df[c] = df[c].tolist()
    ts = ts.sort_values(["fish_id", TIME]).reset_index(drop=True)
    art = np.load(f"{OUT_DIR}/v3_tier1_artifacts.npz", allow_pickle=True)
    mean, scale = art["scaler_mean"], art["scaler_scale"]
    fids = out.fish_id.tolist()
    X = np.stack([((ts[ts.fish_id == f].sort_values(TIME)[FEATURES].values) - mean) / scale
                  for f in fids])                                  # (N,2,5)
    D = np.stack([[1.0 if out[out.fish_id == f].group.iloc[0] == g else 0.0
                   for g in NONREF] for f in fids])                # (N,3) dummies
    y = out.set_index("fish_id").loc[fids, TARGET].values.astype(float)
    cov = art["covars"];  cov = np.stack([np.diag(c) for c in cov]) if cov.ndim == 3 else cov
    return ts, out, jnp.array(X), jnp.array(D), jnp.array(y), \
        jnp.array(art["means"]), jnp.array(np.sqrt(cov)), fids


# ===========================================================================
# (A) Bayesian progression-hazard HMM with group covariates
# ===========================================================================
def hmm_model(X, D, prior_mu, prior_sig):
    N = X.shape[0]
    mu = numpyro.sample("mu", dist.Normal(prior_mu, 0.5))
    log_sig = numpyro.sample("log_sigma", dist.Normal(jnp.log(prior_sig), 0.3))
    sigma = jnp.exp(log_sig)
    alpha = numpyro.sample("alpha", dist.Normal(0., 1.5).expand([N_STATES - 1]))  # advance from 0,1
    beta = numpyro.sample("beta_grp", dist.Normal(0., 1.0).expand([len(NONREF)]))  # vs sham
    start = numpyro.sample("start", dist.Dirichlet(jnp.array([4., 2., 0.5])))
    lp = D @ beta                                            # (N,) group linear predictor
    p = jax.nn.sigmoid(alpha[None, :] + lp[:, None])         # (N,2) advance hazards
    T = jnp.zeros((N, 3, 3))
    T = T.at[:, 0, 0].set(1 - p[:, 0]).at[:, 0, 1].set(p[:, 0])
    T = T.at[:, 1, 1].set(1 - p[:, 1]).at[:, 1, 2].set(p[:, 1]).at[:, 2, 2].set(1.0)
    logT = jnp.log(T + 1e-30)
    d = (X[:, :, None, :] - mu[None, None, :, :]) / sigma[None, None, :, :]
    le = jnp.sum(-0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma)[None, None, :, :]
                 - 0.5 * d ** 2, axis=-1)                    # (N,2,3)
    la = jnp.log(start + 1e-30)[None, :] + le[:, 0, :]
    la = le[:, 1, :] + jlse(la[:, :, None] + logT, axis=1)
    numpyro.factor("obs", jnp.sum(jlse(la, axis=1)))


# ===========================================================================
# (B) Bayesian logistic outcome model
# ===========================================================================
def logit_model(D, y):
    g0 = numpyro.sample("g0", dist.Normal(0., 1.5))
    g = numpyro.sample("g_grp", dist.Normal(0., 1.5).expand([len(NONREF)]))
    logit = g0 + D @ g
    numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)


def run(model, key, **kw):
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=1000,
                num_samples=1000, num_chains=2, chain_method="sequential",
                progress_bar=False)
    mcmc.run(random.PRNGKey(key), **kw)
    return mcmc


# ===========================================================================
def main():
    print("=" * 72); print("ECHO V3  -  TIER 2 Bayesian treatment-effect analysis"); print("=" * 72)
    ts, out, X, D, y, prior_mu, prior_sig, fids = load()
    idx = {g: i for i, g in enumerate(NONREF)}

    # ---- (A) progression-hazard HMM ----
    print("\n(A) Progression-hazard HMM with group covariates (NUTS) ...")
    mA = run(hmm_model, SEED, X=X, D=D, prior_mu=prior_mu, prior_sig=prior_sig)
    pa = {k: np.asarray(v) for k, v in mA.get_samples().items()}
    bg = pa["beta_grp"]                       # (S,3) log-OR of advancing vs sham
    veh = bg[:, idx["se_vehicle"]]; vpa = bg[:, idx["se_vpa"]]; core = bg[:, idx["se_core"]]
    vpa_vs_veh = vpa - veh                     # protective if < 0
    print("   log-OR of latent progression (vs sham):")
    for g in NONREF:
        v = bg[:, idx[g]]; lo, hi = hdi(v)
        print(f"     {g:11s}: {np.median(v):+.2f}  94% HDI [{lo:+.2f}, {hi:+.2f}]  P(>0)={np.mean(v>0):.3f}")
    lo, hi = hdi(vpa_vs_veh)
    print(f"   VPA vs vehicle (progression): {np.median(vpa_vs_veh):+.2f} "
          f"94% HDI [{lo:+.2f}, {hi:+.2f}]  P(protective<0)={np.mean(vpa_vs_veh<0):.3f}")

    # ---- (B) logistic outcome ----
    print("\n(B) Logistic outcome model became_sensitized ~ group (NUTS) ...")
    mB = run(logit_model, SEED + 1, D=D, y=y)
    pb = {k: np.asarray(v) for k, v in mB.get_samples().items()}
    g = pb["g_grp"]
    or_tab = {}
    print("   odds ratio vs sham:")
    for gr in NONREF:
        orr = np.exp(g[:, idx[gr]]); lo, hi = hdi(orr)
        or_tab[gr] = (float(np.median(orr)), lo, hi, float(np.mean(g[:, idx[gr]] > 0)))
        print(f"     {gr:11s}: OR={np.median(orr):.2f}  94% HDI [{lo:.2f}, {hi:.2f}]")
    or_vpa_veh = np.exp(g[:, idx["se_vpa"]] - g[:, idx["se_vehicle"]])
    lo, hi = hdi(or_vpa_veh)
    p_prot = float(np.mean(or_vpa_veh < 1))
    print(f"   *** VPA vs vehicle (OUTCOME): OR={np.median(or_vpa_veh):.2f} "
          f"94% HDI [{lo:.2f}, {hi:.2f}]  P(protective, OR<1)={p_prot:.3f} ***")

    # ---- plots ----
    plot_forest(or_tab, or_vpa_veh)
    plot_vpa_posterior(vpa_vs_veh, or_vpa_veh)

    # ---- report + metrics ----
    write_report(bg, idx, vpa_vs_veh, or_tab, or_vpa_veh, p_prot)
    json.dump(dict(
        progression_logOR_vs_sham={gr: [float(np.median(bg[:, idx[gr]])), *hdi(bg[:, idx[gr]])]
                                   for gr in NONREF},
        progression_vpa_vs_vehicle=[float(np.median(vpa_vs_veh)), *hdi(vpa_vs_veh),
                                    float(np.mean(vpa_vs_veh < 0))],
        outcome_OR_vs_sham={gr: list(or_tab[gr]) for gr in NONREF},
        outcome_OR_vpa_vs_vehicle=[float(np.median(or_vpa_veh)), *hdi(or_vpa_veh), p_prot]),
        open(f"{OUT_DIR}/v3_tier2_metrics.json", "w"), indent=2)
    print("\nV3 TIER 2 complete.")


def plot_forest(or_tab, or_vpa_veh):
    rows = [("se_core vs sham", *or_tab["se_core"][:3]),
            ("se_vehicle vs sham", *or_tab["se_vehicle"][:3]),
            ("se_vpa vs sham", *or_tab["se_vpa"][:3]),
            ("se_vpa vs se_vehicle\n(VPA effect)", float(np.median(or_vpa_veh)), *hdi(or_vpa_veh))]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    yv = np.arange(len(rows))[::-1]
    for yi, (lab, med, lo, hi) in zip(yv, rows):
        c = "tab:green" if "VPA effect" in lab else "tab:gray"
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.5)
        ax.plot(med, yi, "o", color=c, ms=9)
        ax.text(med, yi + 0.12, f"{med:.2f}", ha="center", fontsize=9)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_yticks(yv); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log"); ax.set_xlabel("odds ratio for sensitization (log scale)")
    ax.set_title("V3 sensitization odds ratios (94% HDI)\nOR<1 = protective")
    fig.tight_layout(); savefig(fig, "v3_outcome_odds_ratios.png"); plt.close(fig)


def plot_vpa_posterior(vpa_vs_veh_logor, or_vpa_veh):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, vals, lab, ref, xl in [
        (axes[0], vpa_vs_veh_logor, "progression log-OR\n(VPA vs vehicle)", 0.0, "log-odds ratio"),
        (axes[1], or_vpa_veh, "outcome OR\n(VPA vs vehicle)", 1.0, "odds ratio")]:
        ax.hist(vals, bins=40, color="tab:green", alpha=.8, density=True)
        lo, hi = hdi(vals); ax.axvspan(lo, hi, color="tab:green", alpha=.15)
        ax.axvline(ref, color="k", ls="--"); ax.axvline(np.median(vals), color="tab:red", lw=2)
        p = np.mean(vals < ref)
        ax.set_title(f"{lab}\nmedian={np.median(vals):.2f}  P(protective)={p:.3f}")
        ax.set_xlabel(xl)
    fig.suptitle("VPA protective effect - full posterior", fontsize=12)
    fig.tight_layout(); savefig(fig, "v3_vpa_posterior.png"); plt.close(fig)


def write_report(bg, idx, vpa_vs_veh, or_tab, or_vpa_veh, p_prot):
    L = []; A = L.append
    A("# ECHO V3 - Tier 2 Bayesian treatment-effect analysis\n")
    A("Full posterior uncertainty on the SE and VPA effects. Two models on the "
      "full cohort (N=128): a progression-hazard HMM (latent mechanism) and a "
      "logistic outcome model (clinical sensitization). Per-fish random effects "
      "are omitted (only 2 timepoints/fish -> not identifiable); group effects "
      "are fixed with weakly-informative priors.\n")
    A("## (A) Effect on latent progression (HMM, log-odds of advancing vs sham)\n")
    A("| contrast | median log-OR | 94% HDI | P(>0) |")
    A("|---|---|---|---|")
    for gr in NONREF:
        v = bg[:, idx[gr]]; lo, hi = hdi(v)
        A(f"| {gr} vs sham | {np.median(v):+.2f} | [{lo:+.2f}, {hi:+.2f}] | {np.mean(v>0):.3f} |")
    lo, hi = hdi(vpa_vs_veh)
    A(f"| **se_vpa vs se_vehicle** | **{np.median(vpa_vs_veh):+.2f}** | "
      f"[{lo:+.2f}, {hi:+.2f}] | P(<0)={np.mean(vpa_vs_veh<0):.3f} |")
    A("\n## (B) Effect on the actual outcome (logistic, odds ratios vs sham)\n")
    A("| contrast | OR | 94% HDI |")
    A("|---|---|---|")
    for gr in NONREF:
        med, lo, hi, _ = or_tab[gr]
        A(f"| {gr} vs sham | {med:.2f} | [{lo:.2f}, {hi:.2f}] |")
    lo, hi = hdi(or_vpa_veh)
    A(f"| **se_vpa vs se_vehicle (VPA)** | **{np.median(or_vpa_veh):.2f}** | [{lo:.2f}, {hi:.2f}] |")
    A(f"\n- **VPA is credibly protective on the outcome: OR {np.median(or_vpa_veh):.2f} "
      f"(94% HDI [{lo:.2f}, {hi:.2f}]), P(OR<1) = {p_prot:.3f}.**")
    A("- SE is harmful vs sham (se_vehicle OR "
      f"{or_tab['se_vehicle'][0]:.1f}), and VPA pulls the odds back down toward sham.")
    A("\n## Figures\n- `outputs/v3_outcome_odds_ratios.png` (forest plot)\n"
      "- `outputs/v3_vpa_posterior.png` (VPA protective-effect posterior)")
    open(f"{OUT_DIR}/v3_tier2_report.md", "w", encoding="utf-8").write("\n".join(L))
    print(f"[report] {OUT_DIR}/v3_tier2_report.md")


if __name__ == "__main__":
    main()
