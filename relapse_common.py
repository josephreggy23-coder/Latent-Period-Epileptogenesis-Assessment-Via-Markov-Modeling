"""
relapse_common.py
=================
Shared utilities for the RELAPSE epileptogenesis HMM pipeline (Tier 1 + Tier 2).

This module deliberately knows NOTHING about the ground-truth label columns when
it builds model inputs. The only columns ever fed to a model are the 5 LFP
features. The TRUTH / outcome columns are loaded separately and used ONLY for
validation.

Pip installs needed for the whole project:
    pip install numpy pandas scipy scikit-learn matplotlib openpyxl hmmlearn
    # Tier 2 additionally tries: numpyro/jax  OR  pymc   OR  emcee (fallback)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.special import logsumexp

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "RELAPSE_synthetic_data.xlsx")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# The ONLY columns ever used as model input:
FEATURES = [
    "lfp_spike_amp_uV",
    "lfp_spike_freq_hz",
    "lfp_hfo_power",
    "lfp_signal_entropy",
    "lfp_line_length",
]

# Validation-only columns (NEVER used as model input):
TRUTH_STATE_COL = "hidden_state_TRUTH"
OUTCOME_COL = "became_epileptic"
SEIZURE_HR_COL = "first_seizure_hours"

N_STATES = 4  # planted ground truth has 4 states (0=healthy ... 3=seizure)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_data(path: str = DATA_PATH):
    """Load the timeseries and fish-level outcome sheets.

    Returns
    -------
    ts  : DataFrame, one row per fish per timepoint, sorted by (fish_id, hours).
    out : DataFrame, one row per fish (outcomes / ground truth).
    """
    ts = pd.read_excel(path, sheet_name="LFP_timeseries")
    out = pd.read_excel(path, sheet_name="fish_outcomes")
    # pandas 3.0 makes string columns PyArrow-backed, which breaks integer-array
    # indexing (sklearn splits). Coerce id/group cols to plain numpy object dtype.
    for df in (ts, out):
        for c in ("fish_id", "group"):
            df[c] = df[c].tolist()
    ts = ts.sort_values(["fish_id", "hours_post_insult"]).reset_index(drop=True)
    return ts, out


def fish_level_split(out: pd.DataFrame, test_frac: float = 0.30, seed: int = 42):
    """Split FISH (not timepoints) into train/test, stratified by group x outcome.

    Stratifying on group x became_epileptic keeps both arms and both outcomes
    represented in train and test, which matters with only 80 fish.
    """
    from sklearn.model_selection import train_test_split

    strata = (out["group"].astype(str) + "_" + out[OUTCOME_COL].astype(str)).tolist()
    fish = out["fish_id"].tolist()
    train_ids, test_ids = train_test_split(
        np.asarray(fish, dtype=object),
        test_size=test_frac,
        random_state=seed,
        stratify=np.asarray(strata, dtype=object),
    )
    return set(train_ids), set(test_ids)


def build_sequences(ts: pd.DataFrame, fish_ids, scaler):
    """Assemble hmmlearn-style (X, lengths) from a set of fish.

    Parameters
    ----------
    ts       : full timeseries DataFrame.
    fish_ids : iterable of fish_id to include.
    scaler   : a *fitted* sklearn scaler applied to FEATURES.

    Returns
    -------
    X       : (sum_T, 5) scaled emissions, fish concatenated in `order`.
    lengths : list of per-fish sequence lengths (all 7 here).
    order   : list of fish_id in the same order they appear in X.
    frames  : dict fish_id -> per-fish DataFrame (unscaled, with truth cols).
    """
    fish_ids = [f for f in pd.unique(ts["fish_id"]) if f in set(fish_ids)]
    blocks, lengths, order, frames = [], [], [], {}
    for fid in fish_ids:
        g = ts[ts["fish_id"] == fid].sort_values("hours_post_insult")
        Xf = scaler.transform(g[FEATURES].values)
        blocks.append(Xf)
        lengths.append(len(g))
        order.append(fid)
        frames[fid] = g.reset_index(drop=True)
    X = np.vstack(blocks)
    return X, lengths, order, frames


# ----------------------------------------------------------------------------
# Severity alignment (HMM states are arbitrarily labelled -> map to 0..3)
# ----------------------------------------------------------------------------
def severity_order(means_: np.ndarray) -> np.ndarray:
    """Return a relabelling array `perm` such that perm[raw_state] = severity rank.

    Because every LFP feature increases monotonically with disease severity and
    the model is trained on standardized features, a state's severity is well
    captured by the sum of its (standardized) emission means. Ranking states by
    that sum maps arbitrary HMM labels -> 0 (mildest) .. 3 (most severe).
    """
    severity_score = means_.sum(axis=1)          # higher = more severe
    ranked = np.argsort(severity_score)          # raw indices, mild -> severe
    perm = np.empty_like(ranked)
    perm[ranked] = np.arange(len(ranked))        # perm[raw] = rank
    return perm


# ----------------------------------------------------------------------------
# Emission log-likelihood + honest forward FILTER (online, no future leakage)
# ----------------------------------------------------------------------------
def diag_gaussian_loglik(X: np.ndarray, means_: np.ndarray, covars_: np.ndarray):
    """Per-timepoint, per-state Gaussian log-likelihood for a diagonal-cov HMM.

    Computed directly from means_/covars_ (no private hmmlearn API), so it is
    robust across library versions.

    Returns (T, K) log p(x_t | state=k).
    """
    X = np.asarray(X)
    means_ = np.asarray(means_)
    var = np.asarray(covars_)
    if var.ndim == 3:                  # (K, F, F) full -> take diagonal
        var = np.stack([np.diag(c) for c in var])
    T, F = X.shape
    K = means_.shape[0]
    ll = np.empty((T, K))
    log2pi = np.log(2.0 * np.pi)
    for k in range(K):
        d = X - means_[k]              # (T, F)
        ll[:, k] = -0.5 * np.sum(log2pi + np.log(var[k]) + d * d / var[k], axis=1)
    return ll


def forward_filter(log_emis: np.ndarray, startprob: np.ndarray, transmat: np.ndarray):
    """Online forward FILTER: gamma_t = P(state_t | observations_0..t).

    Crucially this uses ONLY past and present observations (not the whole
    sequence), so a risk score computed from it never "sees" the future
    seizure. This is the right object for *early* prediction; hmmlearn's
    predict_proba returns the *smoothed* posterior (uses the full sequence)
    and would leak future information into early timepoints.

    Returns (T, K) filtered posterior probabilities (rows sum to 1).
    """
    T, K = log_emis.shape
    log_alpha = np.empty((T, K))
    log_T = np.log(transmat + 1e-300)
    log_alpha[0] = np.log(startprob + 1e-300) + log_emis[0]
    for t in range(1, T):
        # predict step then update with emission
        prev = log_alpha[t - 1][:, None] + log_T          # (K_prev, K_next)
        log_alpha[t] = logsumexp(prev, axis=0) + log_emis[t]
    # normalize each row -> filtered posterior over states at time t
    gamma = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
    return gamma


# ----------------------------------------------------------------------------
# Small plotting helper
# ----------------------------------------------------------------------------
def savefig(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  [plot] {path}")
    return path
