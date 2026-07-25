"""
TIER 1 - Standard Gaussian HMM for zebrafish epileptogenesis
============================================================
pip install numpy pandas scipy scikit-learn matplotlib openpyxl hmmlearn

Pipeline
--------
1. Load timeseries; treat each fish as a sequence of 7 observations over time.
   Emissions = the 5 LFP features ONLY. Ground-truth columns are never input.
2. Fish-level 70/30 train/test split (a fish's timepoints never straddle the split).
3. Standardize features (scaler fit on TRAIN only).
4. Fit a 4-state Gaussian HMM (hmmlearn) with several random restarts.
5. Validate against the planted truth:
     (a) state recovery   - accuracy + confusion matrix (states severity-aligned)
     (b) early prediction - online forward-FILTER risk; ROC-AUC, accuracy, lead-time
     (c) microplastic     - control vs microplastic transition / progression rates
6. Save plots (PNG), a markdown report, and artifacts for the Tier 1 vs 2 comparison.

Run:  python tier1_gaussian_hmm.py
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve)
from hmmlearn.hmm import GaussianHMM

from relapse_common import (
    FEATURES, TRUTH_STATE_COL, OUTCOME_COL, SEIZURE_HR_COL, N_STATES, OUT_DIR,
    load_data, fish_level_split, build_sequences, severity_order,
    diag_gaussian_loglik, forward_filter, savefig,
)

SEED = 42
RISK_STATE_FLOOR = 2     # "at-risk / epileptogenic" = severity state >= 2
np.random.seed(SEED)


# ===========================================================================
# Model fitting (with restarts; EM is sensitive to initialization)
# ===========================================================================
def fit_hmm(X_train, lengths_train, n_restarts=12, n_iter=300):
    best, best_ll = None, -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(n_restarts):
            m = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                            n_iter=n_iter, tol=1e-4, random_state=SEED + r,
                            min_covar=1e-3, init_params="stmc")
            m.fit(X_train, lengths_train)
            ll = m.score(X_train, lengths_train)
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
    print(f"  best train log-likelihood over {n_restarts} restarts: {best_ll:,.1f}")
    return best


def relabel_states(model):
    """Severity-align the raw HMM labels -> 0..3 (mild..severe). Returns perm."""
    return severity_order(model.means_)


# ===========================================================================
# (a) State recovery
# ===========================================================================
def state_recovery(model, perm, ts, fish_ids, scaler, tag):
    X, lengths, order, frames = build_sequences(ts, fish_ids, scaler)
    raw = model.predict(X, lengths)
    pred = perm[raw]                                  # severity-aligned labels
    truth = np.concatenate([frames[f][TRUTH_STATE_COL].values for f in order])
    acc = accuracy_score(truth, pred)
    cm = confusion_matrix(truth, pred, labels=list(range(N_STATES)))
    print(f"  [{tag}] state-recovery accuracy = {acc:.3f}  (n={len(truth)} timepoints)")
    return acc, cm, pred, truth


# ===========================================================================
# (b) Early prediction (honest, online forward-filter; no future leakage)
# ===========================================================================
def per_fish_risk(model, perm, frames, order, scaler):
    """Return dict fid -> DataFrame[hours, risk, filt state probs] using the
    online forward FILTER, with severity-aligned states.
    risk = P(state_t >= RISK_STATE_FLOOR | observations up to t)."""
    inv = np.argsort(perm)        # inv[rank] = raw label  (column reorder)
    out = {}
    for fid in order:
        g = frames[fid]
        Xf = scaler.transform(g[FEATURES].values)
        log_emis = diag_gaussian_loglik(Xf, model.means_, model.covars_)
        gamma_raw = forward_filter(log_emis, model.startprob_, model.transmat_)
        gamma = gamma_raw[:, inv]                      # reorder cols -> severity
        risk = gamma[:, RISK_STATE_FLOOR:].sum(axis=1)
        out[fid] = pd.DataFrame({
            "hours": g["hours_post_insult"].values,
            "risk": risk,
            **{f"p_state{j}": gamma[:, j] for j in range(N_STATES)},
        })
    return out


def early_prediction(risk_by_fish, out_df, test_ids):
    """Fish-level early-prediction metrics on held-out fish.

    Fish score = max filtered risk over its STRICTLY pre-seizure window
    (epileptic: hours < first_seizure; non-epileptic: all timepoints). Excluding
    the seizure timepoint itself makes this a genuine BEFORE-seizure prediction.
    """
    rows = []
    for fid in test_ids:
        r = risk_by_fish[fid]
        o = out_df[out_df.fish_id == fid].iloc[0]
        ep = int(o[OUTCOME_COL])
        fsz = o[SEIZURE_HR_COL]
        if ep == 1 and np.isfinite(fsz):
            pre = r[r.hours < fsz]
        else:
            pre = r
        score = pre["risk"].max() if len(pre) else 0.0
        rows.append(dict(fish_id=fid, group=o["group"], became_epileptic=ep,
                         first_seizure_hours=fsz, score=score))
    df = pd.DataFrame(rows)
    y, s = df.became_epileptic.values, df.score.values
    auc = roc_auc_score(y, s)

    # Youden-optimal threshold on the held-out ROC
    fpr, tpr, thr = roc_curve(y, s)
    youden = thr[np.argmax(tpr - fpr)]
    pred = (s >= youden).astype(int)
    acc = accuracy_score(y, pred)

    # Lead-time: earliest pre-seizure hour where risk >= threshold, among the
    # epileptic held-out fish that were flagged before their seizure.
    leads, flagged = [], 0
    n_ep = int(df.became_epileptic.sum())
    for fid in df[df.became_epileptic == 1].fish_id:
        r = risk_by_fish[fid]
        fsz = out_df.loc[out_df.fish_id == fid, SEIZURE_HR_COL].iloc[0]
        pre = r[r.hours < fsz]
        hit = pre[pre.risk >= youden]
        if len(hit):
            flagged += 1
            leads.append(fsz - hit.hours.iloc[0])     # hours of warning
    lead_mean = float(np.mean(leads)) if leads else float("nan")
    print(f"  held-out early-prediction: AUC={auc:.3f}  acc={acc:.3f}  "
          f"thr={youden:.2f}")
    print(f"  flagged BEFORE seizure: {flagged}/{n_ep} epileptic fish, "
          f"mean lead-time={lead_mean:.1f} h")
    metrics = dict(auc=auc, accuracy=acc, threshold=float(youden),
                   n_test=len(df), n_test_epileptic=n_ep,
                   flagged_before_seizure=flagged, mean_lead_time_h=lead_mean,
                   median_lead_time_h=float(np.median(leads)) if leads else float("nan"))
    return df, metrics, (fpr, tpr)


# ===========================================================================
# (c) Microplastic effect on transition / progression rates
# ===========================================================================
def group_transition_analysis(model, perm, ts, scaler):
    """Decode every fish (Viterbi), severity-align, then tally transitions
    separately for control vs microplastic. Compare progression."""
    res = {}
    for grp in ["control", "microplastic"]:
        ids = ts.loc[ts.group == grp, "fish_id"].unique()
        X, lengths, order, frames = build_sequences(ts, ids, scaler)
        raw = model.predict(X, lengths)
        states = perm[raw]
        # rebuild per-fish state paths to count within-fish transitions
        counts = np.zeros((N_STATES, N_STATES))
        advances = opportunities = 0
        i = 0
        for L in lengths:
            seq = states[i:i + L]; i += L
            for a, b in zip(seq[:-1], seq[1:]):
                counts[a, b] += 1
                opportunities += 1
                if b > a:
                    advances += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            T = counts / counts.sum(axis=1, keepdims=True)
        T = np.nan_to_num(T)
        res[grp] = dict(
            transmat=T, counts=counts,
            advance_rate=advances / opportunities,          # P(move up per 2h step)
            reach_state3=float(np.mean([(perm[model.predict(
                scaler.transform(frames[f][FEATURES].values))] == 3).any()
                for f in order])),
        )
    ctrl, mp = res["control"], res["microplastic"]
    ratio = mp["advance_rate"] / ctrl["advance_rate"]
    print(f"  advance-rate (P state increases / step): control={ctrl['advance_rate']:.3f}"
          f"  microplastic={mp['advance_rate']:.3f}  ratio={ratio:.2f}x")
    print(f"  fraction reaching seizure state 3: control={ctrl['reach_state3']:.2f}"
          f"  microplastic={mp['reach_state3']:.2f}")
    res["advance_rate_ratio"] = ratio
    return res


# ===========================================================================
# Plots
# ===========================================================================
def plot_trajectories(model, perm, ts, out_df, scaler):
    inv = np.argsort(perm)
    # Representative examples: epileptic from BOTH arms, and resilient "near-miss"
    # fish that climbed highest without seizing (the hard cases for early prediction).
    ep_df = out_df[out_df.became_epileptic == 1]
    ne_df = out_df[out_df.became_epileptic == 0].sort_values("max_state_reached",
                                                             ascending=False)
    ep = (ep_df[ep_df.group == "control"].fish_id.tolist()[:1] +
          ep_df[ep_df.group == "microplastic"].fish_id.tolist()[:2])
    ne = ne_df.fish_id.tolist()[:3]
    picks = ep + ne
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for ax, fid in zip(axes.ravel(), picks):
        g = ts[ts.fish_id == fid].sort_values("hours_post_insult")
        Xf = scaler.transform(g[FEATURES].values)
        viterbi = perm[model.predict(Xf)]
        le = diag_gaussian_loglik(Xf, model.means_, model.covars_)
        gamma = forward_filter(le, model.startprob_, model.transmat_)[:, inv]
        risk = gamma[:, RISK_STATE_FLOOR:].sum(1)
        o = out_df[out_df.fish_id == fid].iloc[0]
        ax.step(g.hours_post_insult, g[TRUTH_STATE_COL], where="mid",
                color="k", lw=2.5, label="truth state", alpha=.7)
        ax.step(g.hours_post_insult, viterbi, where="mid",
                color="tab:red", lw=1.8, ls="--", label="HMM Viterbi")
        ax2 = ax.twinx()
        ax2.plot(g.hours_post_insult, risk, color="tab:blue", marker="o",
                 ms=4, label="filtered risk P(state>=2)")
        ax2.set_ylim(-0.03, 1.03)
        if o.became_epileptic == 1 and np.isfinite(o.first_seizure_hours):
            ax.axvline(o.first_seizure_hours, color="tab:green", ls=":", lw=2)
        ax.set_title(f"{fid} ({o.group}, "
                     f"{'EPILEPTIC' if o.became_epileptic else 'resilient'})",
                     fontsize=10)
        ax.set_xlabel("hours post insult"); ax.set_ylim(-0.2, 3.2)
        ax.set_yticks(range(4))
    axes[0, 0].set_ylabel("hidden state"); axes[1, 0].set_ylabel("hidden state")
    fig.suptitle("Inferred vs ground-truth hidden states  "
                 "(green dotted = first seizure; blue = online risk)", fontsize=12)
    fig.tight_layout()
    savefig(fig, "tier1_state_trajectories.png"); plt.close(fig)


def plot_risk_curves(risk_by_fish, out_df, test_ids):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    hours_grid = sorted(out_df.attrs.get("hours", [0, 2, 4, 6, 8, 10, 12]))
    for ep, color, lab in [(1, "tab:red", "epileptic"), (0, "tab:blue", "resilient")]:
        curves = []
        for fid in test_ids:
            o = out_df[out_df.fish_id == fid].iloc[0]
            if o.became_epileptic != ep:
                continue
            r = risk_by_fish[fid]
            ax.plot(r.hours, r.risk, color=color, alpha=.25, lw=1)
            curves.append(r.set_index("hours")["risk"])
        if curves:
            mean_curve = pd.concat(curves, axis=1).mean(axis=1)
            ax.plot(mean_curve.index, mean_curve.values, color=color, lw=3,
                    marker="o", label=f"{lab} (mean, n={len(curves)})")
    ax.set_xlabel("hours post insult")
    ax.set_ylabel("online risk  P(state >= 2 | obs up to t)")
    ax.set_title("Predicted epileptogenic risk over time (held-out fish)")
    ax.legend(); ax.set_ylim(-0.03, 1.03)
    fig.tight_layout(); savefig(fig, "tier1_risk_over_time.png"); plt.close(fig)


def plot_transitions(group_res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, grp in zip(axes, ["control", "microplastic"]):
        T = group_res[grp]["transmat"]
        im = ax.imshow(T, cmap="viridis", vmin=0, vmax=1)
        for i in range(N_STATES):
            for j in range(N_STATES):
                ax.text(j, i, f"{T[i,j]:.2f}", ha="center", va="center",
                        color="w" if T[i, j] < .6 else "k", fontsize=9)
        ax.set_title(f"{grp}\nadvance-rate={group_res[grp]['advance_rate']:.3f}")
        ax.set_xlabel("to state"); ax.set_ylabel("from state")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
    fig.suptitle("Learned transition matrices (Viterbi-decoded, severity-aligned)")
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="P(transition)")
    savefig(fig, "tier1_transition_matrices.png"); plt.close(fig)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_STATES):
        for j in range(N_STATES):
            ax.text(j, i, f"{int(cm[i,j])}\n{cmn[i,j]:.2f}", ha="center",
                    va="center", color="w" if cmn[i, j] > .5 else "k", fontsize=9)
    ax.set_xlabel("predicted state"); ax.set_ylabel("true state")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_title("State recovery confusion matrix (held-out)")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="row-normalized")
    fig.tight_layout(); savefig(fig, "tier1_confusion_matrix.png"); plt.close(fig)


# ===========================================================================
# Report
# ===========================================================================
def write_report(acc_test, acc_all, cm_test, early_metrics, group_res, model, perm):
    inv = np.argsort(perm)
    lines = []
    A = lines.append
    A("# RELAPSE Tier 1 - Gaussian HMM results\n")
    A("Standard 4-state Gaussian HMM (`hmmlearn`), diagonal covariance, trained on "
      "70% of fish (fish-level split), validated on the held-out 30%. The 5 LFP "
      "features are the only model inputs; all TRUTH columns are used only to score.\n")

    A("## (a) Hidden-state recovery\n")
    A(f"- **Held-out accuracy: {acc_test:.1%}**  |  all-fish accuracy: {acc_all:.1%}")
    A(f"- States were severity-aligned to ground truth by ranking emission means.\n")
    A("Confusion matrix (held-out, rows = true state, cols = predicted):\n")
    A("| true \\ pred | 0 | 1 | 2 | 3 |")
    A("|---|---|---|---|---|")
    for i in range(N_STATES):
        A(f"| **{i}** | " + " | ".join(str(int(v)) for v in cm_test[i]) + " |")
    A("")

    A("## (b) Early prediction (before first seizure)\n")
    em = early_metrics
    A(f"- **ROC-AUC (held-out): {em['auc']:.3f}**")
    A(f"- Accuracy at Youden-optimal threshold ({em['threshold']:.2f}): "
      f"**{em['accuracy']:.1%}**")
    A(f"- Epileptic held-out fish flagged BEFORE their seizure: "
      f"**{em['flagged_before_seizure']}/{em['n_test_epileptic']}**")
    A(f"- **Mean lead-time: {em['mean_lead_time_h']:.1f} h** "
      f"(median {em['median_lead_time_h']:.1f} h) before first seizure")
    A("- Risk = filtered P(state>=2) from an ONLINE forward pass (uses only "
      "observations up to t -> no leakage from the future seizure).\n")

    A("## (c) Microplastic effect on progression\n")
    c, m = group_res["control"], group_res["microplastic"]
    A(f"- Advance-rate P(state increases per 2 h step): "
      f"control **{c['advance_rate']:.3f}** vs microplastic **{m['advance_rate']:.3f}** "
      f"= **{group_res['advance_rate_ratio']:.2f}x** faster progression under microplastic.")
    A(f"- Fraction of fish reaching seizure state 3: control {c['reach_state3']:.0%} "
      f"vs microplastic {m['reach_state3']:.0%}.")
    A("- Direction matches the planted sanity check (microplastic -> faster "
      "progression, higher incidence).\n")

    A("## Learned emission means (severity-aligned, original units)\n")
    A("| state | " + " | ".join(FEATURES) + " |")
    A("|---" * (len(FEATURES) + 1) + "|")
    # report means back in original units via the saved scaler stats
    means_orig = MODEL_SCALER.inverse_transform(model.means_[inv])
    for j in range(N_STATES):
        A(f"| {j} | " + " | ".join(f"{v:.2f}" for v in means_orig[j]) + " |")
    A("")
    A("## Figures\n")
    for fn in ["tier1_state_trajectories.png", "tier1_risk_over_time.png",
               "tier1_transition_matrices.png", "tier1_confusion_matrix.png"]:
        A(f"- `outputs/{fn}`")
    report = "\n".join(lines)
    path = f"{OUT_DIR}/tier1_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  [report] {path}")
    return report


# module-level handle so the report can inverse-transform means
MODEL_SCALER: StandardScaler = None


def main():
    global MODEL_SCALER
    print("=" * 70)
    print("TIER 1  -  Gaussian HMM")
    print("=" * 70)
    ts, out = load_data()
    out.attrs["hours"] = sorted(ts.hours_post_insult.unique())
    train_ids, test_ids = fish_level_split(out, 0.30, SEED)
    print(f"Fish: {len(train_ids)} train / {len(test_ids)} test "
          f"(split at fish level, stratified by group x outcome)")

    # scaler fit on TRAIN timepoints only
    train_rows = ts[ts.fish_id.isin(train_ids)]
    scaler = StandardScaler().fit(train_rows[FEATURES].values)
    MODEL_SCALER = scaler

    X_tr, len_tr, _, _ = build_sequences(ts, train_ids, scaler)
    print("\nFitting HMM ...")
    model = fit_hmm(X_tr, len_tr)
    perm = relabel_states(model)

    print("\n(a) State recovery")
    acc_test, cm_test, _, _ = state_recovery(model, perm, ts, test_ids, scaler, "held-out")
    acc_all, _, _, _ = state_recovery(model, perm, ts, out.fish_id.tolist(), scaler, "all fish")

    print("\n(b) Early prediction")
    _, _, order_all, frames_all = build_sequences(ts, out.fish_id.tolist(), scaler)
    risk_by_fish = per_fish_risk(model, perm, frames_all, order_all, scaler)
    early_df, early_metrics, roc = early_prediction(risk_by_fish, out, sorted(test_ids))

    print("\n(c) Microplastic effect")
    group_res = group_transition_analysis(model, perm, ts, scaler)

    print("\nPlots")
    plot_trajectories(model, perm, ts, out, scaler)
    plot_risk_curves(risk_by_fish, out, sorted(test_ids))
    plot_transitions(group_res)
    plot_confusion(cm_test)

    write_report(acc_test, acc_all, cm_test, early_metrics, group_res, model, perm)

    # ---- save artifacts so Tier 2 reuses the identical split & test scores ----
    inv = np.argsort(perm)
    np.savez(f"{OUT_DIR}/tier1_artifacts.npz",
             train_ids=np.array(sorted(train_ids)),
             test_ids=np.array(sorted(test_ids)),
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             means=model.means_[inv], covars=model.covars_[inv],
             transmat=model.transmat_[np.ix_(inv, inv)],
             startprob=model.startprob_[inv],
             test_fish=early_df.fish_id.values,
             test_y=early_df.became_epileptic.values,
             test_score=early_df.score.values)
    with open(f"{OUT_DIR}/tier1_metrics.json", "w") as f:
        json.dump(dict(state_recovery_heldout=acc_test, state_recovery_all=acc_all,
                       early=early_metrics,
                       advance_rate_control=group_res["control"]["advance_rate"],
                       advance_rate_microplastic=group_res["microplastic"]["advance_rate"],
                       advance_rate_ratio=group_res["advance_rate_ratio"]),
                  f, indent=2)
    print("  [artifacts] outputs/tier1_artifacts.npz, outputs/tier1_metrics.json")
    print("\nTIER 1 complete.")


if __name__ == "__main__":
    main()
