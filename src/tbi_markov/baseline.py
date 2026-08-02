"""Elastic-net landmark logistic regression: the head-to-head static baseline.

The README has listed this as "planned" since the project began (Task 4). It
tests whether the HMM's latent state dynamics add predictive value over a
regularized static classifier fit on the same causal landmark feature vector,
the same reduced three-feature allowlist, and the identical fish-level split.
The comparison is reported honestly in the results regardless of outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .common import PREDICTION_CUTOFF_DPF, TARGET, build_sequences


def _landmark_matrix(
    lfp: pd.DataFrame,
    fish_ids: set[str],
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """One feature vector per fish: its last available <=5 dpf session.

    Uses the identical causal landmark rule as the HMM's forward filter
    (build_sequences with the same PREDICTION_CUTOFF_DPF), so both models see
    exactly the same information at prediction time.
    """
    sequences, order, _ = build_sequences(
        lfp, fish_ids, center, scale, cutoff_dpf=PREDICTION_CUTOFF_DPF
    )
    matrix = np.vstack([sequence[-1] for sequence in sequences])
    return matrix, order


def fit_elastic_net_baseline(
    lfp: pd.DataFrame,
    outcomes: pd.DataFrame,
    train_ids: set[str],
    test_ids: set[str],
    center: np.ndarray,
    scale: np.ndarray,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Fit on training fish only, score held-out fish, report head-to-head."""
    train_outcomes = outcomes.loc[
        outcomes["fish_id"].isin(train_ids) & outcomes[TARGET].notna()
    ].set_index("fish_id")
    test_outcomes = outcomes.loc[
        outcomes["fish_id"].isin(test_ids) & outcomes[TARGET].notna()
    ].set_index("fish_id")

    X_train, train_order = _landmark_matrix(
        lfp, set(train_outcomes.index.astype(str)), center, scale
    )
    y_train = train_outcomes.loc[train_order, TARGET].to_numpy(int)
    X_test, test_order = _landmark_matrix(
        lfp, set(test_outcomes.index.astype(str)), center, scale
    )
    y_test = test_outcomes.loc[test_order, TARGET].to_numpy(int)

    model = LogisticRegressionCV(
        Cs=8,
        cv=5,
        solver="saga",
        l1_ratios=[0.1, 0.5, 0.9],
        max_iter=5000,
        random_state=seed,
        scoring="average_precision",
        use_legacy_attributes=True,
    )
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]

    threshold = 0.5
    label = score >= threshold
    true_positive = int(np.sum((label == 1) & (y_test == 1)))
    true_negative = int(np.sum((label == 0) & (y_test == 0)))
    false_positive = int(np.sum((label == 1) & (y_test == 0)))
    false_negative = int(np.sum((label == 0) & (y_test == 1)))
    sensitivity = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)

    metrics = {
        "model": "elastic-net landmark logistic regression",
        "features": "identical causal <=5 dpf landmark vector, same reduced feature allowlist",
        "n_train_fish": int(len(y_train)),
        "n_test_fish": int(len(y_test)),
        "n_positive": int(y_test.sum()),
        "chosen_C": float(model.C_[0]),
        "chosen_l1_ratio": float(model.l1_ratio_[0]),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "average_precision": float(average_precision_score(y_test, score)),
        "brier_score": float(brier_score_loss(y_test, score)),
        "threshold": threshold,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }
    predictions = pd.DataFrame(
        {
            "fish_id": test_order,
            TARGET: y_test,
            "baseline_risk_dpf6": score,
        }
    ).sort_values("fish_id").reset_index(drop=True)
    return metrics, predictions
