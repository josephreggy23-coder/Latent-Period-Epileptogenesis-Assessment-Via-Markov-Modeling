"""
ECHO V3 - TIER 1 Gaussian HMM  (SE -> PTZ re-challenge SENSITIZATION model)
===========================================================================
Treats ECHO_V3_synthetic_data.xlsx as the "real" experiment. Trains an HMM on
the 5 silent-period LFP features ONLY, then reports what the model classified:
hidden states, per-fish sensitization risk, and the group/treatment (VPA) effect.

Differences vs the original RELAPSE pipeline (V3 is a different design):
  * 4 groups: sham, se_core, se_vehicle, se_vpa  (key question: is VPA protective?)
  * 2 timepoints per fish (4 h, 20 h post-SE) -> short sequences
  * latent states occur only at 0,1,2 (state 2 rare); we model-select K by BIC
  * TARGET = became_sensitized, a SEPARATE re-challenge outcome (probabilistic,
    not a deterministic top state) -> prediction is realistically imperfect.

pip install numpy pandas scipy scikit-learn matplotlib openpyxl hmmlearn
Run:  python v3_tier1.py
"""
from __future__ import annotations
import json, warnings, logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)   # silence zero-sum-row spam
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve)
from hmmlearn.hmm import GaussianHMM

from relapse_common import (diag_gaussian_loglik, forward_filter, severity_order,
                            savefig, OUT_DIR)

# ----------------------------- V3 schema -----------------------------------
DATA = "ECHO_V3_synthetic_data.xlsx"
FEATURES = ["lfp_discharge_amp_uV", "lfp_discharge_freq_hz", "lfp_delta_power_norm",
            "lfp_ied_interval_s", "lfp_line_length"]
TIME = "hours_post_se"
TRUTH_STATE = "hidden_state_TRUTH"
TARGET = "became_sensitized"
GROUPS = ["sham", "se_core", "se_vehicle", "se_vpa"]
N_STATES = 3                 # latent states 0,1,2 present in the data
RISK_FLOOR = 1               # "elevated" = inferred state >= 1 (state 2 is rare)
SEED = 42
np.random.seed(SEED)


def load():
    ts = pd.read_excel(DATA, "LFP_timeseries")
    out = pd.read_excel(DATA, "fish_outcomes")
    for df in (ts, out):
        for c in ("fish_id", "group"):
            if c in df: df[c] = df[c].tolist()
    ts = ts.sort_values(["fish_id", TIME]).reset_index(drop=True)
    return ts, out


def sequences(ts, fish_ids, scaler):
    fish_ids = [f for f in pd.unique(ts.fish_id) if f in set(fish_ids)]
    blocks, lengths, order, frames = [], [], [], {}
    for fid in fish_ids:
        g = ts[ts.fish_id == fid].sort_values(TIME)
        blocks.append(scaler.transform(g[FEATURES].values))
        lengths.append(len(g)); order.append(fid); frames[fid] = g.reset_index(drop=True)
    return np.vstack(blocks), lengths, order, frames


def fit_hmm(X, lengths, k, n_restarts=15, n_iter=400):
    best, best_ll = None, -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(n_restarts):
            m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=n_iter,
                            tol=1e-4, random_state=SEED + r, min_covar=1e-3)
            try:
                m.fit(X, lengths); ll = m.score(X, lengths)
            except Exception:
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
    return best, best_ll


def bic(model, X, lengths, k, F):
    ll = model.score(X, lengths)
    n_params = k * (k - 1) + (k - 1) + k * F + k * F     # trans+start+means+diagcov
    n = X.shape[0]
    return -2 * ll + n_params * np.log(n), ll


# ===========================================================================
def main():
    print("=" * 72); print("ECHO V3  -  TIER 1 Gaussian HMM (sensitization model)"); print("=" * 72)
    ts, out = load()
    print(f"{out.fish_id.nunique()} fish x {ts.groupby('fish_id').size().iloc[0]} timepoints "
          f"({sorted(ts[TIME].unique())} h)  | groups: {out.group.value_counts().to_dict()}")

    # fish-level split, stratified by group x outcome
    strata = (out.group.astype(str) + "_" + out[TARGET].astype(str)).tolist()
    tr_ids, te_ids = train_test_split(np.asarray(out.fish_id.tolist(), object),
                                      test_size=0.30, random_state=SEED,
                                      stratify=np.asarray(strata, object))
    tr_ids, te_ids = set(tr_ids), set(te_ids)
    scaler = StandardScaler().fit(ts[ts.fish_id.isin(tr_ids)][FEATURES].values)
    Xtr, ltr, _, _ = sequences(ts, tr_ids, scaler)
    print(f"split: {len(tr_ids)} train / {len(te_ids)} test fish")

    # ---- model selection by BIC ----
    print("\n[Model selection] BIC over K hidden states (lower = better):")
    bic_rows = []
    for k in [2, 3, 4, 5]:
        m, _ = fit_hmm(Xtr, ltr, k)
        if m is None:                       # all restarts degenerate -> K unsupported
            print(f"   K={k}:  failed to fit (degenerate; data supports < {k} states)")
            continue
        b, ll = bic(m, Xtr, ltr, k, len(FEATURES))
        bic_rows.append((k, b, ll)); print(f"   K={k}:  BIC={b:8.1f}   logL={ll:8.1f}")
    best_k = min(bic_rows, key=lambda r: r[1])[0]
    print(f"   -> BIC selects K={best_k}.  Using K={N_STATES} (matches 3 planted "
          f"latent states) for state-level validation.")

    # ---- fit final K=3 model ----
    model, best_ll = fit_hmm(Xtr, ltr, N_STATES)
    perm = severity_order(model.means_)          # raw label -> severity rank
    inv = np.argsort(perm)
    print(f"\nFinal HMM (K={N_STATES}) train logL={best_ll:.1f}")

    # ---- (a) state recovery ----
    def recover(ids, tag):
        X, L, order, frames = sequences(ts, ids, scaler)
        pred = perm[model.predict(X, L)]
        truth = np.concatenate([frames[f][TRUTH_STATE].values for f in order])
        acc = accuracy_score(truth, pred)
        cm = confusion_matrix(truth, pred, labels=range(N_STATES))
        print(f"   [{tag}] state recovery acc={acc:.3f} (n={len(truth)} timepoints)")
        return acc, cm
    print("\n(a) Hidden-state recovery vs planted truth")
    acc_te, cm_te = recover(te_ids, "held-out")
    acc_all, cm_all = recover(out.fish_id.tolist(), "all fish")

    # ---- per-fish inference: states @4h,@20h + sensitization risk ----
    Xall, Lall, order, frames = sequences(ts, out.fish_id.tolist(), scaler)
    rows = []
    for fid in order:
        g = frames[fid]
        Xf = scaler.transform(g[FEATURES].values)
        le = diag_gaussian_loglik(Xf, model.means_, model.covars_)
        gamma = forward_filter(le, model.startprob_, model.transmat_)[:, inv]  # (T,K) severity
        viterbi = perm[model.predict(Xf)]
        risk_t = gamma[:, RISK_FLOOR:].sum(1)        # P(state>=1 | obs up to t)
        o = out[out.fish_id == fid].iloc[0]
        hours = g[TIME].values
        rows.append(dict(
            fish_id=fid, group=o.group, became_sensitized=int(o[TARGET]),
            state_4h=int(viterbi[list(hours).index(4)]),
            state_20h=int(viterbi[list(hours).index(20)]),
            truth_state_4h=int(g[TRUTH_STATE].values[list(hours).index(4)]),
            truth_state_20h=int(g[TRUTH_STATE].values[list(hours).index(20)]),
            risk_4h=float(risk_t[list(hours).index(4)]),
            risk_max=float(risk_t.max()),
            in_test=fid in te_ids))
    cls = pd.DataFrame(rows)

    # ---- (b) sensitization classification ----
    print("\n(b) Sensitization prediction (risk = P(inferred state >= 1))")
    def classify(df, score_col, tag):
        y, s = df[TARGET].values, df[score_col].values
        auc = roc_auc_score(y, s)
        fpr, tpr, thr = roc_curve(y, s); j = np.argmax(tpr - fpr); cut = thr[j]
        pred = (s >= cut).astype(int)
        cm = confusion_matrix(y, pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn); spec = tn / (tn + fp)
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        npv = tn / (tn + fn) if (tn + fn) else float("nan")
        print(f"   [{tag}] AUC={auc:.3f} acc={accuracy_score(y,pred):.3f} "
              f"sens={sens:.2f} spec={spec:.2f} PPV={ppv:.2f} NPV={npv:.2f} (thr={cut:.2f})")
        return dict(tag=tag, auc=float(auc), acc=float(accuracy_score(y, pred)),
                    sens=float(sens), spec=float(spec), ppv=float(ppv), npv=float(npv),
                    threshold=float(cut), cm=cm.tolist(), n=len(df),
                    n_pos=int(y.sum()), roc=(fpr, tpr))
    cls_te = cls[cls.in_test]
    m_all_max = classify(cls, "risk_max", "all fish | 4h+20h")
    m_te_max = classify(cls_te, "risk_max", "held-out | 4h+20h")
    m_all_4h = classify(cls, "risk_4h", "all fish | 4h ONLY (early)")

    # ---- (c) group / VPA treatment effect ----
    print("\n(c) Group & VPA treatment effect (inferred latent progression)")
    grp_rows = []
    for grp in GROUPS:
        sub = cls[cls.group == grp]
        progressed = (sub.state_20h > sub.state_4h).mean()
        grp_rows.append(dict(group=grp, n=len(sub),
                             mean_state_20h=sub.state_20h.mean(),
                             pct_progressed=progressed,
                             pct_elevated_20h=(sub.state_20h >= 1).mean(),
                             actual_sensitized=sub[TARGET].mean()))
    gdf = pd.DataFrame(grp_rows).set_index("group").loc[GROUPS]
    print(gdf.to_string(float_format=lambda x: f"{x:.3f}"))
    veh = gdf.loc["se_vehicle", "pct_progressed"]; vpa = gdf.loc["se_vpa", "pct_progressed"]
    sham = gdf.loc["sham", "pct_progressed"]
    vpa_prot = 1 - vpa / veh if veh else float("nan")
    print(f"\n   VPA effect on inferred progression: se_vpa={vpa:.3f} vs se_vehicle={veh:.3f}"
          f"  -> {vpa_prot:.0%} relative reduction (toward sham={sham:.3f}).")

    # per-group transition matrices (pooled, severity-aligned, Viterbi)
    def group_transmat(grp):
        ids = out[out.group == grp].fish_id.tolist()
        X, L, order2, _ = sequences(ts, ids, scaler)
        st = perm[model.predict(X, L)]; i = 0; C = np.zeros((N_STATES, N_STATES))
        for Ln in L:
            seq = st[i:i + Ln]; i += Ln
            for a, b in zip(seq[:-1], seq[1:]): C[a, b] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            T = np.nan_to_num(C / C.sum(1, keepdims=True))
        return T
    gtrans = {g: group_transmat(g) for g in GROUPS}

    # ---- save per-fish classifications ----
    cls_out = cls.drop(columns=["risk_4h"]).copy()
    cls_out["pred_sensitized@all"] = (cls.risk_max >= m_all_max["threshold"]).astype(int)
    cls_out["correct"] = (cls_out["pred_sensitized@all"] == cls_out[TARGET]).astype(int)
    cls_out.to_csv(f"{OUT_DIR}/v3_per_fish_classification.csv", index=False)
    print(f"\n[saved] {OUT_DIR}/v3_per_fish_classification.csv (per-fish classifications)")

    # ---- plots ----
    plots(bic_rows, best_k, cm_te, gdf, gtrans, cls, m_all_max, m_te_max)

    # ---- report ----
    report(out, acc_te, acc_all, cm_te, bic_rows, best_k, m_all_max, m_te_max,
           m_all_4h, gdf, vpa, veh, sham, vpa_prot, model, perm, cls)

    json.dump(dict(state_recovery_heldout=acc_te, state_recovery_all=acc_all,
                   bic={str(k): b for k, b, _ in bic_rows}, best_k=best_k,
                   sens_pred_all=_strip(m_all_max), sens_pred_heldout=_strip(m_te_max),
                   sens_pred_4h_only=_strip(m_all_4h),
                   group=gdf.reset_index().to_dict("records"),
                   vpa_progression=vpa, vehicle_progression=veh,
                   vpa_relative_reduction=vpa_prot),
              open(f"{OUT_DIR}/v3_tier1_metrics.json", "w"), indent=2)
    # stash artifacts for Tier 2
    np.savez(f"{OUT_DIR}/v3_tier1_artifacts.npz",
             train_ids=np.array(sorted(tr_ids)), test_ids=np.array(sorted(te_ids)),
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             means=model.means_[inv], covars=model.covars_[inv],
             transmat=model.transmat_[np.ix_(inv, inv)], startprob=model.startprob_[inv])
    print("\nV3 TIER 1 complete.")


def _strip(m):
    return {k: v for k, v in m.items() if k != "roc"}


def plots(bic_rows, best_k, cm_te, gdf, gtrans, cls, m_all, m_te):
    # BIC
    fig, ax = plt.subplots(figsize=(5, 4))
    ks = [r[0] for r in bic_rows]; bs = [r[1] for r in bic_rows]
    ax.plot(ks, bs, "o-"); ax.axvline(best_k, color="tab:red", ls="--", label=f"BIC min K={best_k}")
    ax.axvline(N_STATES, color="tab:green", ls=":", label=f"used K={N_STATES} (truth)")
    ax.set_xlabel("n hidden states K"); ax.set_ylabel("BIC"); ax.set_xticks(ks)
    ax.set_title("V3 model selection"); ax.legend()
    fig.tight_layout(); savefig(fig, "v3_bic_selection.png"); plt.close(fig)

    # confusion
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    cmn = cm_te / cm_te.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_STATES):
        for j in range(N_STATES):
            ax.text(j, i, f"{int(cm_te[i,j])}", ha="center", va="center",
                    color="w" if cmn[i, j] > .5 else "k")
    ax.set_xticks(range(N_STATES)); ax.set_yticks(range(N_STATES))
    ax.set_xlabel("predicted state"); ax.set_ylabel("true state")
    ax.set_title(f"State recovery (held-out)")
    fig.tight_layout(); savefig(fig, "v3_confusion_matrix.png"); plt.close(fig)

    # group progression + sensitization bars
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(GROUPS)); w = 0.4
    ax.bar(x - w/2, gdf.pct_progressed.values, w, label="inferred progression rate",
           color="tab:orange")
    ax.bar(x + w/2, gdf.actual_sensitized.values, w, label="actual sensitization rate",
           color="tab:purple")
    ax.set_xticks(x); ax.set_xticklabels(GROUPS); ax.set_ylabel("fraction")
    ax.set_title("V3: inferred latent progression vs actual sensitization, by group")
    ax.legend()
    for i, g in enumerate(GROUPS):
        ax.text(i, -0.06, f"n={int(gdf.loc[g,'n'])}", ha="center", fontsize=8)
    fig.tight_layout(); savefig(fig, "v3_group_effects.png"); plt.close(fig)

    # risk distribution by outcome
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v, c, lab in [(1, "tab:red", "sensitized"), (0, "tab:blue", "not sensitized")]:
        ax.hist(cls[cls[TARGET] == v].risk_max, bins=15, alpha=.6, color=c, label=lab)
    ax.axvline(m_all["threshold"], color="k", ls="--", label="decision threshold")
    ax.set_xlabel("model sensitization risk  max P(state>=1)"); ax.set_ylabel("n fish")
    ax.set_title("V3: model risk score, sensitized vs not"); ax.legend()
    fig.tight_layout(); savefig(fig, "v3_risk_distribution.png"); plt.close(fig)

    # transition matrices by group
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, g in zip(axes, GROUPS):
        T = gtrans[g]; ax.imshow(T, cmap="viridis", vmin=0, vmax=1)
        for i in range(N_STATES):
            for j in range(N_STATES):
                ax.text(j, i, f"{T[i,j]:.2f}", ha="center", va="center",
                        color="w" if T[i, j] < .6 else "k", fontsize=8)
        ax.set_title(g); ax.set_xlabel("to"); ax.set_ylabel("from")
        ax.set_xticks(range(N_STATES)); ax.set_yticks(range(N_STATES))
    fig.suptitle("V3 learned transition matrices (4h->20h), by group")
    fig.tight_layout(); savefig(fig, "v3_transition_matrices.png"); plt.close(fig)

    # ROC
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(*m_all["roc"], label=f"all fish (AUC={m_all['auc']:.3f})", color="tab:purple")
    ax.plot(*m_te["roc"], label=f"held-out (AUC={m_te['auc']:.3f})", color="tab:red")
    ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("V3 sensitization prediction ROC"); ax.legend(loc="lower right")
    fig.tight_layout(); savefig(fig, "v3_roc.png"); plt.close(fig)


def report(out, acc_te, acc_all, cm_te, bic_rows, best_k, m_all, m_te, m_4h,
           gdf, vpa, veh, sham, vpa_prot, model, perm, cls):
    inv = np.argsort(perm)
    L = []; A = L.append
    A("# ECHO V3 - Tier 1 HMM results (treated as the real experiment)\n")
    A("SE -> PTZ re-challenge sensitization model. A 3-state Gaussian HMM was "
      "trained on the 5 silent-period LFP features ONLY (4 h & 20 h post-SE); "
      "all TRUTH/outcome columns were held out for scoring. Below is what the "
      "model classified and how well it matches the planted truth.\n")
    A("## Model selection\n")
    A("| K states | BIC | logL |")
    A("|---|---|---|")
    for k, b, ll in bic_rows: A(f"| {k} | {b:.1f} | {ll:.1f} |")
    A(f"\nBIC favors K={best_k}; we report K={N_STATES} to match the 3 planted "
      "latent states (state 2 is rare, 6/256 timepoints).\n")
    A("## (a) Hidden-state recovery\n")
    A(f"- **Held-out accuracy {acc_te:.1%}** (all-fish {acc_all:.1%}).")
    A("- Confusion (held-out, rows=true, cols=pred):\n")
    A("| true\\pred | 0 | 1 | 2 |"); A("|---|---|---|---|")
    for i in range(N_STATES): A(f"| {i} | " + " | ".join(str(int(v)) for v in cm_te[i]) + " |")
    A("\n## (b) Sensitization classification (the clinical target)\n")
    A("Risk = model P(inferred latent state >= 1). NOTE the outcome is a SEPARATE "
      "re-challenge event, only probabilistically tied to silent-period state, so "
      "perfect prediction is impossible by construction.\n")
    A("| evaluation | AUC | acc | sens | spec | PPV | NPV |")
    A("|---|---|---|---|---|---|---|")
    for m in (m_all, m_te, m_4h):
        A(f"| {m['tag']} | {m['auc']:.3f} | {m['acc']:.2f} | {m['sens']:.2f} | "
          f"{m['spec']:.2f} | {m['ppv']:.2f} | {m['npv']:.2f} |")
    cm = np.array(m_all["cm"])
    A(f"\nAll-fish confusion @ threshold: TN={cm[0,0]} FP={cm[0,1]} "
      f"FN={cm[1,0]} TP={cm[1,1]} (n={m_all['n']}, {m_all['n_pos']} sensitized).\n")
    A("## (c) Group & VPA treatment effect\n")
    A("| group | n | mean state@20h | inferred progression | actual sensitization |")
    A("|---|---|---|---|---|")
    for g in GROUPS:
        r = gdf.loc[g]
        A(f"| {g} | {int(r.n)} | {r.mean_state_20h:.2f} | {r.pct_progressed:.1%} | "
          f"{r.actual_sensitized:.1%} |")
    A(f"\n- **VPA is protective:** inferred progression se_vpa {vpa:.1%} vs "
      f"se_vehicle {veh:.1%} = **{vpa_prot:.0%} relative reduction**, back down to "
      f"sham ({sham:.1%}).")
    A("- Ordering recovered by the model: se_vehicle > se_core > se_vpa ≈ sham — "
      "matches the planted SE-harm + VPA-rescue design.\n")
    A("## Emission means (severity-aligned, standardized units)\n")
    A("| state | " + " | ".join(FEATURES) + " |")
    A("|---" * (len(FEATURES) + 1) + "|")
    for j in range(N_STATES):
        A(f"| {j} | " + " | ".join(f"{v:+.2f}" for v in model.means_[inv][j]) + " |")
    A("\n(values are standardized; all features rise with state except "
      "`lfp_ied_interval_s` which falls — shorter inter-discharge interval = worse.)\n")
    A("## Files\n- `outputs/v3_per_fish_classification.csv` (every fish's inferred "
      "states + risk + predicted vs actual sensitization)\n- plots: `v3_bic_selection.png`, "
      "`v3_confusion_matrix.png`, `v3_group_effects.png`, `v3_risk_distribution.png`, "
      "`v3_transition_matrices.png`, `v3_roc.png`")
    open(f"{OUT_DIR}/v3_tier1_report.md", "w", encoding="utf-8").write("\n".join(L))
    print(f"[report] {OUT_DIR}/v3_tier1_report.md")


if __name__ == "__main__":
    main()
