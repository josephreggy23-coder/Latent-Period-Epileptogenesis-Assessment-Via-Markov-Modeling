"""
ECHO V5 - Item 6: time-to-event analysis (Kaplan-Meier + Cox proportional hazards).
Judge critique: binary logistic insufficient -> model the LATENCY with censoring.

pip install lifelines
Run:  python v5_survival.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from v5_common import load_v5, TARGET, LATENCY, STUDY_END_H, GROUPS, OUT


def survival_frame(out):
    """duration = latency if sensitized else censored at study end; event = sensitized."""
    df = out.copy()
    df["duration"] = np.where(df[TARGET] == 1, df[LATENCY], STUDY_END_H)
    df["event"] = df[TARGET].astype(int)
    # guard: any sensitized with missing latency -> drop (none expected in V5)
    df = df[~((df.event == 1) & df["duration"].isna())].copy()
    return df


def main():
    print("=" * 74); print("ECHO V5  -  Item 6: Kaplan-Meier + Cox proportional hazards"); print("=" * 74)
    ts, out = load_v5()
    df = survival_frame(out)
    n_event = int(df.event.sum()); n_cens = int((df.event == 0).sum())
    print(f"{len(df)} fish: {n_event} events (sensitized), {n_cens} censored at {STUDY_END_H:.0f}h")

    # ---- Kaplan-Meier by group + log-rank ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    kmf = KaplanMeierFitter()
    km_med = {}
    for g in GROUPS:
        m = df.group == g
        kmf.fit(df.duration[m], df.event[m], label=f"{g} (n={m.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
        km_med[g] = float(kmf.median_survival_time_)
    lr = multivariate_logrank_test(df.duration, df.group, df.event)
    ax.set_title(f"Item 6 - Kaplan-Meier: sensitization-free survival by group\n"
                 f"multivariate log-rank p={lr.p_value:.2e}")
    ax.set_xlabel("hours post-SE"); ax.set_ylabel("P(not yet sensitized)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/v5_kaplan_meier.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"   log-rank across groups: p={lr.p_value:.3e}")
    print(f"   [plot] {OUT}/v5_kaplan_meier.png")

    # ---- Cox PH (group as covariate, sham reference) + batch robustness ----
    def fit_cox(extra_cols):
        dummies = pd.get_dummies(df["group"], prefix="grp")
        ref = "grp_sham"
        cols = [c for c in dummies.columns if c != ref]
        X = pd.concat([df[["duration", "event"]].reset_index(drop=True),
                       dummies[cols].astype(float).reset_index(drop=True)], axis=1)
        if extra_cols:
            bd = pd.get_dummies(df["batch"], prefix="batch").astype(float)
            bd = bd[[c for c in bd.columns if c != bd.columns[0]]].reset_index(drop=True)
            X = pd.concat([X, bd], axis=1)
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(X, "duration", "event")
        return cph

    cph = fit_cox(False)
    print("\n   Cox PH hazard ratios (vs sham):")
    hr = {}
    for g in GROUPS:
        if g == "sham": continue
        row = cph.summary.loc[f"grp_{g}"]
        hr[g] = dict(HR=float(np.exp(row["coef"])),
                     lo=float(np.exp(row["coef lower 95%"])),
                     hi=float(np.exp(row["coef upper 95%"])), p=float(row["p"]))
        print(f"     {g:12s} HR={hr[g]['HR']:5.2f}  95% CI [{hr[g]['lo']:.2f}, {hr[g]['hi']:.2f}]  p={hr[g]['p']:.3g}")

    # batch-adjusted model: do the group HRs survive?
    cph_b = fit_cox(True)
    hr_adj = {g: float(np.exp(cph_b.summary.loc[f"grp_{g}", "coef"])) for g in GROUPS if g != "sham"}
    print("   batch-adjusted HRs (vs sham):", {k: round(v, 2) for k, v in hr_adj.items()})

    # ---- HR forest plot ----
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    order = [g for g in GROUPS if g != "sham"]
    yv = np.arange(len(order))[::-1]
    for yi, g in zip(yv, order):
        c = "tab:green" if "vpa" in g and "wash" not in g else ("tab:red" if g in ("se_vehicle", "se_vpa_wash") else "tab:gray")
        ax.plot([hr[g]["lo"], hr[g]["hi"]], [yi, yi], color=c, lw=2.5)
        ax.plot(hr[g]["HR"], yi, "o", color=c, ms=9)
        ax.text(hr[g]["HR"], yi + 0.12, f"{hr[g]['HR']:.2f}", ha="center", fontsize=9)
    ax.axvline(1, color="k", ls="--"); ax.set_yticks(yv); ax.set_yticklabels(order)
    ax.set_xscale("log"); ax.set_xlabel("hazard ratio vs sham (log scale)")
    ax.set_title("Item 6 - Cox hazard ratios (green=VPA dose, red=vehicle/washout)")
    fig.tight_layout(); fig.savefig(f"{OUT}/v5_cox_forest.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"   [plot] {OUT}/v5_cox_forest.png")

    json.dump(dict(n_event=n_event, n_censored=n_cens, logrank_p=float(lr.p_value),
                   km_median=km_med, cox_hr_vs_sham=hr, cox_hr_batch_adjusted=hr_adj,
                   cox_concordance=float(cph.concordance_index_)),
              open(f"{OUT}/v5_survival_metrics.json", "w"), indent=2)
    print(f"   Cox concordance (C-index) = {cph.concordance_index_:.3f}")
    print("Item 6 complete.")


if __name__ == "__main__":
    main()
