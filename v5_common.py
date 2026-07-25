"""
v5_common.py - shared utilities for the ECHO V5 judge-hardened LFP-only pipeline.

pip install numpy pandas scipy scikit-learn matplotlib openpyxl hmmlearn numpyro lifelines

Single-electrode optic-tectum LFP is a FIXED constraint: all 11 features come from
ONE channel. Ground-truth columns are NEVER model inputs (scoring only).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "ECHO_V5_synthetic_data.xlsx")
OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)

# 11 single-electrode LFP features (the ONLY model inputs)
FEATURES = ["lfp_discharge_amp_uV", "lfp_discharge_freq_hz", "lfp_delta_power_norm",
            "lfp_theta_power_norm", "lfp_ied_interval_s", "lfp_line_length",
            "lfp_spectral_entropy", "lfp_hjorth_mobility", "lfp_hjorth_complexity",
            "lfp_fast_ripple_rate", "lfp_pac_theta_gamma"]
# the 5 "basic" features (classic discharge metrics) for the subset comparison
BASIC = ["lfp_discharge_amp_uV", "lfp_discharge_freq_hz", "lfp_delta_power_norm",
         "lfp_ied_interval_s", "lfp_line_length"]
ADDED = [f for f in FEATURES if f not in BASIC]   # entropy, Hjorth, fast-ripple, PAC

TIME = "hours_post_se"
HOURS = [4, 8, 12, 16, 20]
TRUTH_STATE = "hidden_state_TRUTH"
TARGET = "became_sensitized"
LATENCY = "sensitization_latency_h"
STUDY_END_H = 24.0                       # censoring horizon for non-sensitized fish
GROUPS = ["sham", "se_core", "se_vehicle", "se_vpa_low", "se_vpa_high", "se_vpa_wash"]
# VPA dose axis (mg/L-ish ordinal) for dose-response; washout handled separately
VPA_DOSE = {"sham": 0.0, "se_core": 0.0, "se_vehicle": 0.0,
            "se_vpa_low": 1.0, "se_vpa_high": 2.0, "se_vpa_wash": 0.0}
SEED = 42


def load_v5():
    ts = pd.read_excel(DATA, "LFP_timeseries").sort_values(["fish_id", TIME]).reset_index(drop=True)
    out = pd.read_excel(DATA, "fish_outcomes")
    return ts, out


# ===========================================================================
# Item 1: artifact rejection (modified z-score) + robust preprocessing
# ===========================================================================
def detect_artifacts(ts: pd.DataFrame, thresh: float = 3.5):
    """Flag artifact cells via the Iglewicz-Hoaglin modified z-score
    (0.6745*(x-median)/MAD).  Returns a boolean mask DataFrame (same index, the
    FEATURE columns) and a per-feature count table."""
    mask = pd.DataFrame(False, index=ts.index, columns=FEATURES)
    counts = {}
    for f in FEATURES:
        x = ts[f].values.astype(float)
        med = np.median(x); mad = stats.median_abs_deviation(x)
        mz = 0.6745 * (x - med) / mad if mad > 0 else np.zeros_like(x)
        flagged = np.abs(mz) > thresh
        mask[f] = flagged
        counts[f] = int(flagged.sum())
    return mask, counts


class RobustPreprocessor:
    """Winsorize (robust, fit on TRAIN) then RobustScaler (median/IQR). Tames the
    ~4% injected artifacts + student-t heavy tails so the Gaussian HMM is valid."""

    def __init__(self, lo_pct=1.0, hi_pct=99.0):
        self.lo_pct, self.hi_pct = lo_pct, hi_pct

    def fit(self, ts_train: pd.DataFrame):
        X = ts_train[FEATURES].values.astype(float)
        self.lo = np.percentile(X, self.lo_pct, axis=0)
        self.hi = np.percentile(X, self.hi_pct, axis=0)
        Xw = np.clip(X, self.lo, self.hi)
        self.scaler = RobustScaler().fit(Xw)
        return self

    def transform(self, ts: pd.DataFrame) -> np.ndarray:
        X = np.clip(ts[FEATURES].values.astype(float), self.lo, self.hi)
        return self.scaler.transform(X)

    def transform_cols(self, ts, cols):
        """Transform only a subset of columns (for the feature-subset experiment)."""
        idx = [FEATURES.index(c) for c in cols]
        X = np.clip(ts[cols].values.astype(float), self.lo[idx], self.hi[idx])
        return (X - self.scaler.center_[idx]) / self.scaler.scale_[idx]


# ===========================================================================
# Fish-level sequence assembly
# ===========================================================================
def fish_sequences(ts, fish_ids, pre: RobustPreprocessor, upto_h=None, cols=None):
    """Return (X, lengths, order, frames). If upto_h given, truncate each fish's
    sequence to timepoints <= upto_h (forward-filter / temporal validation)."""
    cols = cols or FEATURES
    ids = [f for f in pd.unique(ts.fish_id) if f in set(fish_ids)]
    blocks, lengths, order, frames = [], [], [], {}
    for fid in ids:
        g = ts[ts.fish_id == fid].sort_values(TIME)
        if upto_h is not None:
            g = g[g[TIME] <= upto_h]
        Xf = pre.transform(g) if cols == FEATURES else pre.transform_cols(g, cols)
        blocks.append(Xf); lengths.append(len(g)); order.append(fid); frames[fid] = g
    return np.vstack(blocks), lengths, order, frames


def fish_strata(out):
    """Stratify on group x outcome so 5-fold / split keep both balanced."""
    return (out.group.astype(str) + "_" + out[TARGET].astype(str)).values
