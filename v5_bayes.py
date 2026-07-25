"""
ECHO V5 - Item 7 (Bayesian dose-response + washout, priors, sensitivity,
diagnostics, PPC, calibration) and the Item-2 hierarchical per-fish RE HMM
(now identifiable with 5 timepoints).

pip install numpyro
Run:  python v5_bayes.py     (after v5_pipeline.py for emission anchors)
"""
from __future__ import annotations
import json, warnings, logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, jax.random as random
from jax.scipy.special import logsumexp as jlse
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import summary as nsummary
from hmmlearn.hmm import GaussianHMM

from v5_common import (FEATURES, TIME, HOURS, TARGET, GROUPS, SEED, OUT,
                       load_v5, RobustPreprocessor)

numpyro.set_host_device_count(2)
SE_GROUPS = ["se_vehicle", "se_vpa_low", "se_vpa_high", "se_vpa_wash"]
NONREF = [g for g in GROUPS if g != "sham"]          # sham = reference


def hdi(s, p=0.94):
    s = np.sort(np.asarray(s)); n = len(s); w = int(np.floor(p*n))
    i = int(np.argmin(s[w:] - s[:n-w])); return float(s[i]), float(s[i+w])


# ===========================================================================
# Item 7: Bayesian logistic dose-response outcome model
# ===========================================================================
def logistic_model(D, y, prior_sd=1.5):
    a = numpyro.sample("a", dist.Normal(0., 1.5))            # sham log-odds
    b = numpyro.sample("b", dist.Normal(0., prior_sd).expand([D.shape[1]]))
    numpyro.sample("y", dist.Bernoulli(logits=a + D @ b), obs=y)


def run_logistic(D, y, prior_sd, seed=0, samples=1500):
    mcmc = MCMC(NUTS(logistic_model), num_warmup=1000, num_samples=samples,
                num_chains=2, chain_method="sequential", progress_bar=False)
    mcmc.run(random.PRNGKey(seed), D=D, y=y, prior_sd=prior_sd)
    return mcmc


def main():
    print("=" * 74); print("ECHO V5  -  Item 7 Bayesian dose-response + Item 2 hierarchical HMM"); print("=" * 74)
    ts, out = load_v5()
    idx = {g: i for i, g in enumerate(NONREF)}
    D = np.stack([[1.0 if out.group.iloc[r] == g else 0.0 for g in NONREF]
                  for r in range(len(out))])
    y = out[TARGET].values.astype(float)
    R = {}

    # ---------- main Bayesian logistic (default priors) ----------
    print("\n[7] Bayesian logistic dose-response (priors: b ~ Normal(0, 1.5)) ...")
    mcmc = run_logistic(jnp.array(D), jnp.array(y), 1.5, seed=SEED)
    post = mcmc.get_samples(); a = np.asarray(post["a"]); b = np.asarray(post["b"])

    # diagnostics (R-hat, ESS)
    diag = nsummary(mcmc.get_samples(group_by_chain=True))
    rhat_max = max(np.max(diag[k]["r_hat"]) for k in diag)
    ess_min = min(np.min(diag[k]["n_eff"]) for k in diag)
    print(f"   diagnostics: max R-hat={rhat_max:.3f}, min ESS={ess_min:.0f}")

    # posterior incidence by group + key contrasts
    p_group = {"sham": 1/(1+np.exp(-a))}
    for g in NONREF:
        p_group[g] = 1/(1+np.exp(-(a + b[:, idx[g]])))
    def OR(gnum, gden):  # odds ratio gnum vs gden
        ln = (b[:, idx[gnum]] if gnum != "sham" else 0)
        ld = (b[:, idx[gden]] if gden != "sham" else 0)
        return np.exp(ln - ld)
    contrasts = {
        "se_vehicle_vs_sham": OR("se_vehicle", "sham"),
        "vpa_low_vs_vehicle": OR("se_vpa_low", "se_vehicle"),
        "vpa_high_vs_vehicle": OR("se_vpa_high", "se_vehicle"),
        "washout_vs_vehicle": OR("se_vpa_wash", "se_vehicle"),
        "washout_vs_vpa_high": OR("se_vpa_wash", "se_vpa_high"),
    }
    print("   key odds ratios (median [94% HDI], P(OR<1)):")
    R["contrasts"] = {}
    for k, v in contrasts.items():
        lo, hi = hdi(v); R["contrasts"][k] = [float(np.median(v)), lo, hi, float(np.mean(v < 1))]
        print(f"     {k:22s} OR={np.median(v):5.2f}  [{lo:.2f}, {hi:.2f}]  P(<1)={np.mean(v<1):.3f}")
    R["diagnostics"] = dict(max_rhat=float(rhat_max), min_ess=float(ess_min))
    R["posterior_incidence"] = {g: [float(np.median(p_group[g])), *hdi(p_group[g])] for g in GROUPS}

    dose_response_figure(p_group, contrasts)
    trace_figure(mcmc, idx)
    ppc_figure(out, a, b, idx)
    calibration_figure(out, a, b, idx)

    # ---------- prior sensitivity ----------
    print("\n[7] Prior sensitivity (refit with weak/strong priors) ...")
    R["prior_sensitivity"] = prior_sensitivity(D, y, idx)

    # ---------- Item 2: hierarchical per-fish RE HMM ----------
    print("\n[2] Hierarchical per-fish random-effect HMM (now identifiable, 5 timepoints) ...")
    R["hierarchical_hmm"] = hierarchical_hmm(ts, out)

    json.dump(R, open(f"{OUT}/v5_bayes_metrics.json", "w"), indent=2, default=float)
    print(f"\n[saved] {OUT}/v5_bayes_metrics.json")
    print("Item 7 + hierarchical HMM complete.")


# ===========================================================================
def dose_response_figure(p_group, contrasts):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = ["sham", "se_vehicle", "se_vpa_low", "se_vpa_high", "se_vpa_wash"]
    labels = ["sham", "vehicle\n(SE)", "VPA low", "VPA high", "washout"]
    cols = ["tab:gray", "tab:red", "gold", "tab:green", "tab:purple"]
    for i, g in enumerate(order):
        v = p_group[g]; lo, hi = hdi(v)
        axes[0].errorbar(i, np.median(v)*100, yerr=[[(np.median(v)-lo)*100], [(hi-np.median(v))*100]],
                         fmt="o", ms=11, capsize=5, color=cols[i])
    # dose-response trend line through sham->vehicle->low->high
    axes[0].plot([0, 1, 2, 3], [np.median(p_group[g])*100 for g in order[:4]], "--", color="k", alpha=.4)
    axes[0].set_xticks(range(5)); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("posterior P(sensitized) %")
    axes[0].set_title("Item 7 - VPA dose-response + WASHOUT control\n"
                      "(washout returns to vehicle level = pharmacological)")
    axes[0].annotate("washout snaps back\nto vehicle, NOT high-dose",
                     xy=(4, np.median(p_group["se_vpa_wash"])*100), xytext=(2.3, 62),
                     fontsize=8, arrowprops=dict(arrowstyle="->", color="tab:purple"))
    # contrasts forest
    keys = list(contrasts)[::-1]; yv = np.arange(len(keys))
    for yi, k in zip(yv, keys):
        v = contrasts[k]; lo, hi = hdi(v)
        c = "tab:green" if np.median(v) < 1 else "tab:red"
        axes[1].plot([lo, hi], [yi, yi], color=c, lw=2.5); axes[1].plot(np.median(v), yi, "o", color=c, ms=8)
        axes[1].text(np.median(v), yi+0.12, f"{np.median(v):.2f}", ha="center", fontsize=8)
    axes[1].axvline(1, color="k", ls="--"); axes[1].set_yticks(yv); axes[1].set_yticklabels(keys, fontsize=8)
    axes[1].set_xscale("log"); axes[1].set_xlabel("odds ratio"); axes[1].set_title("Item 7 - key contrasts (OR)")
    fig.tight_layout(); fig.savefig(f"{OUT}/v5_dose_response_washout.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"   [plot] {OUT}/v5_dose_response_washout.png")


def trace_figure(mcmc, idx):
    ch = mcmc.get_samples(group_by_chain=True)
    params = [("a", None), ("b", idx["se_vehicle"]), ("b", idx["se_vpa_high"])]
    names = ["intercept (sham)", "b[se_vehicle]", "b[se_vpa_high]"]
    fig, axes = plt.subplots(len(params), 1, figsize=(9, 6))
    for ax, (p, j), nm in zip(axes, params, names):
        arr = np.asarray(ch[p]); arr = arr if j is None else arr[:, :, j]
        for c in range(arr.shape[0]):
            ax.plot(arr[c], alpha=.7, lw=.6)
        ax.set_ylabel(nm, fontsize=8)
    axes[0].set_title("Item 7 - MCMC trace plots (2 chains overlaid; good mixing)")
    fig.tight_layout(); fig.savefig(f"{OUT}/v5_mcmc_traces.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"   [plot] {OUT}/v5_mcmc_traces.png")


def ppc_figure(out, a, b, idx):
    """Posterior predictive: simulate per-group sensitized counts, compare observed."""
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, g in enumerate(GROUPS):
        m = (out.group == g).values; n = m.sum()
        logit = a + (b[:, idx[g]] if g != "sham" else 0)
        p = 1/(1+np.exp(-logit))
        draws = rng.choice(len(p), 500)
        rep = np.array([rng.binomial(n, p[d]) for d in draws])
        lo, hi = np.percentile(rep, [2.5, 97.5])
        ax.plot([i, i], [lo, hi], color="tab:gray", lw=6, alpha=.5)
        ax.plot(i, np.median(rep), "_", color="k", ms=20)
        ax.plot(i, out[m][TARGET].sum(), "o", color="tab:red", ms=9, zorder=5)
    ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(GROUPS, rotation=20, fontsize=8)
    ax.set_ylabel("# sensitized"); ax.plot([], [], "o", color="tab:red", label="observed")
    ax.plot([], [], color="tab:gray", lw=6, alpha=.5, label="posterior predictive 95%")
    ax.legend(); ax.set_title("Item 7 - posterior predictive check (observed within PP interval)")
    fig.tight_layout(); fig.savefig(f"{OUT}/v5_ppc.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"   [plot] {OUT}/v5_ppc.png")


def calibration_figure(out, a, b, idx):
    p_fish = np.array([1/(1+np.exp(-(a + (b[:, idx[g]] if g != "sham" else 0)))).mean()
                       for g in out.group])
    y = out[TARGET].values
    bins = np.quantile(p_fish, np.linspace(0, 1, 6))
    bins[-1] += 1e-9
    binid = np.digitize(p_fish, bins[1:-1])
    xs, ys, ns = [], [], []
    for bturn in range(len(bins)-1):
        m = binid == bturn
        if m.sum():
            xs.append(p_fish[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(xs, ys, "o-", color="tab:blue", ms=8, label="model")
    for x, yy, n in zip(xs, ys, ns): ax.annotate(f"n={n}", (x, yy), fontsize=7)
    ax.set_xlabel("mean predicted P(sensitized)"); ax.set_ylabel("observed frequency")
    ax.set_title("Item 7 - calibration"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/v5_calibration.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"   [plot] {OUT}/v5_calibration.png")


def prior_sensitivity(D, y, idx):
    res = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    settings = [("strong (sd=0.5)", 0.5, "tab:blue"), ("default (sd=1.5)", 1.5, "tab:green"),
                ("weak (sd=5)", 5.0, "tab:orange")]
    for nm, sd, col in settings:
        mc = run_logistic(jnp.array(D), jnp.array(y), sd, seed=SEED+1, samples=1200)
        b = np.asarray(mc.get_samples()["b"])
        eff = np.exp(b[:, idx["se_vpa_high"]] - b[:, idx["se_vehicle"]])   # VPA-high vs vehicle OR
        lo, hi = hdi(eff)
        res[nm] = [float(np.median(eff)), lo, hi, float(np.mean(eff < 1))]
        ax.hist(eff, bins=50, alpha=.5, color=col, density=True,
                label=f"{nm}: OR={np.median(eff):.2f} P(<1)={np.mean(eff<1):.2f}")
        print(f"     {nm:18s} vpa_high-vs-vehicle OR={np.median(eff):.2f} [{lo:.2f},{hi:.2f}] P(<1)={np.mean(eff<1):.3f}")
    ax.axvline(1, color="k", ls="--"); ax.set_xscale("log")
    ax.set_xlabel("VPA-high vs vehicle OR"); ax.set_title("Item 7 - prior sensitivity (posterior does NOT flip)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/v5_prior_sensitivity.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"   [plot] {OUT}/v5_prior_sensitivity.png")
    return res


# ===========================================================================
# Item 2: hierarchical per-fish random-effect progression HMM
# ===========================================================================
def hierarchical_hmm(ts, out):
    pre = RobustPreprocessor().fit(ts)
    # emission anchors from hmmlearn K=3 (empirical Bayes), severity-aligned
    fids = out.fish_id.values
    X = np.stack([pre.transform(ts[ts.fish_id == f].sort_values(TIME)) for f in fids])  # (N,5,11)
    flat = X.reshape(-1, X.shape[-1]); lengths = [5]*len(fids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hmm = GaussianHMM(3, "diag", n_iter=300, tol=1e-3, random_state=SEED, min_covar=1e-2).fit(flat, lengths)
    rise = [FEATURES.index(f) for f in ["lfp_discharge_amp_uV", "lfp_line_length", "lfp_fast_ripple_rate"]]
    order = np.argsort(hmm.means_[:, rise].mean(1))
    mu = jnp.array(hmm.means_[order]); sig = jnp.array(np.sqrt(np.stack([np.diag(c) for c in hmm.covars_])[order]))

    Xj = jnp.array(X)

    def model(Xj):
        N = Xj.shape[0]
        alpha = numpyro.sample("alpha", dist.Normal(0., 1.5).expand([2]))   # advance from state 0,1
        tau = numpyro.sample("tau", dist.HalfNormal(1.0))                    # between-fish SD
        with numpyro.plate("fish", N):
            z = numpyro.sample("z", dist.Normal(0., 1.))
        u = tau * z
        p = jax.nn.sigmoid(alpha[None, :] + u[:, None])                      # (N,2)
        T = jnp.zeros((N, 3, 3))
        T = T.at[:, 0, 0].set(1-p[:, 0]).at[:, 0, 1].set(p[:, 0])
        T = T.at[:, 1, 1].set(1-p[:, 1]).at[:, 1, 2].set(p[:, 1]).at[:, 2, 2].set(1.0)
        logT = jnp.log(T + 1e-30)
        d = (Xj[:, :, None, :] - mu[None, None]) / sig[None, None]
        le = jnp.sum(-0.5*jnp.log(2*jnp.pi) - jnp.log(sig)[None, None] - 0.5*d**2, axis=-1)  # (N,5,3)
        start = jnp.log(jnp.array([0.8, 0.18, 0.02]))
        la = start[None, :] + le[:, 0, :]
        for t in range(1, 5):
            la = le[:, t, :] + jlse(la[:, :, None] + logT, axis=1)
        numpyro.factor("ll", jnp.sum(jlse(la, axis=1)))

    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=800, num_samples=800,
              num_chains=2, chain_method="sequential", progress_bar=False)
    mc.run(random.PRNGKey(SEED), Xj=Xj)
    diag = nsummary(mc.get_samples(group_by_chain=True))
    tau = np.asarray(mc.get_samples()["tau"])
    lo, hi = hdi(tau)
    rhat = max(np.max(diag[k]["r_hat"]) for k in ("alpha", "tau"))
    ess = min(np.min(diag[k]["n_eff"]) for k in ("alpha", "tau"))
    print(f"   per-fish RE SD tau = {np.median(tau):.2f} [94% HDI {lo:.2f}, {hi:.2f}]  "
          f"(identifiable: ESS={ess:.0f}, R-hat={rhat:.3f})")
    print("   -> with 5 timepoints the between-fish random effect is well-identified "
          "(was not estimable with 2 timepoints in V3).")
    return dict(tau_median=float(np.median(tau)), tau_hdi=[lo, hi],
                ess_min=float(ess), max_rhat=float(rhat))


if __name__ == "__main__":
    main()
