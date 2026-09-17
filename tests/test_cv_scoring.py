"""Tests for stratified cross-validation scoring utilities.

Covers the full scoring pipeline: single-fold scoring, fold aggregation
with bootstrap CIs, calibration error, and net reclassification improvement.
Edge cases (degenerate folds, single-element arrays, perfect predictions)
are tested explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from tbi_markov.cv_scoring import (
    CVResult,
    FoldScore,
    aggregate_folds,
    calibration_error,
    net_reclassification_improvement,
    score_fold,
)


# ---- score_fold --------------------------------------------------------------

class TestScoreFold:
    def test_perfect_predictions(self) -> None:
        """Perfect probabilistic predictions should yield AUC=1, Brier=0."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        y_train = np.ones(10)

        fs = score_fold(y_true, y_score, y_train, fold_index=0)
        assert fs.auc == pytest.approx(1.0)
        assert fs.brier == pytest.approx(0.0)
        assert fs.balanced_accuracy == pytest.approx(1.0)
        assert fs.n_train == 10
        assert fs.n_test == 5
        assert fs.n_positive_test == 3

    def test_chance_predictions(self) -> None:
        """Constant p=0.5 predictions should give AUC=0.5, Brier=0.25."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])

        fs = score_fold(y_true, y_score, np.zeros(8), fold_index=1)
        assert fs.auc == pytest.approx(0.5)
        assert fs.brier == pytest.approx(0.25)

    def test_degenerate_fold_returns_nan_auc(self) -> None:
        """A fold with only one class should return NaN for AUC and AP."""
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.8, 0.9, 0.7])

        fs = score_fold(y_true, y_score, np.zeros(5), fold_index=0)
        assert np.isnan(fs.auc)
        assert np.isnan(fs.average_precision)

    def test_fold_index_is_stored(self) -> None:
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.7])
        fs = score_fold(y_true, y_score, np.zeros(4), fold_index=7)
        assert fs.fold == 7

    def test_custom_threshold(self) -> None:
        """A high threshold should classify fewer samples as positive."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.6, 0.7, 0.85, 0.95])

        fs_low = score_fold(y_true, y_score, np.zeros(6), fold_index=0, threshold=0.5)
        fs_high = score_fold(y_true, y_score, np.zeros(6), fold_index=0, threshold=0.9)

        # threshold=0.5: predicts [0,0,1,1,1,1] -> 2/3 neg correct, 3/3 pos correct
        # threshold=0.9: predicts [0,0,0,0,0,1] -> 3/3 neg correct, 1/3 pos correct
        assert fs_low.balanced_accuracy > fs_high.balanced_accuracy

    def test_average_precision_perfect(self) -> None:
        """Perfect ranking should yield average precision of 1.0."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])

        fs = score_fold(y_true, y_score, np.zeros(5), fold_index=0)
        assert fs.average_precision == pytest.approx(1.0)


# ---- aggregate_folds ---------------------------------------------------------

class TestAggregateFolds:
    def _make_folds(self, aucs, aps=None, briers=None, bas=None):
        """Helper to create FoldScore objects with specified AUCs."""
        n = len(aucs)
        aps = aps or [0.8] * n
        briers = briers or [0.2] * n
        bas = bas or [0.7] * n
        return [
            FoldScore(
                fold=i,
                n_train=20,
                n_test=5,
                n_positive_test=2,
                auc=aucs[i],
                average_precision=aps[i],
                brier=briers[i],
                balanced_accuracy=bas[i],
            )
            for i in range(n)
        ]

    def test_mean_auc_is_correct(self) -> None:
        folds = self._make_folds([0.7, 0.8, 0.9])
        result = aggregate_folds(folds)
        assert result.mean_auc == pytest.approx(0.8)

    def test_std_auc_is_correct(self) -> None:
        folds = self._make_folds([0.7, 0.8, 0.9])
        result = aggregate_folds(folds)
        expected_std = float(np.std([0.7, 0.8, 0.9], ddof=1))
        assert result.std_auc == pytest.approx(expected_std)

    def test_ci_contains_mean(self) -> None:
        """Bootstrap CI should contain the mean AUC."""
        folds = self._make_folds([0.65, 0.70, 0.75, 0.80, 0.85])
        result = aggregate_folds(folds)
        assert result.ci_auc[0] <= result.mean_auc <= result.ci_auc[1]

    def test_ci_width_decreases_with_more_folds(self) -> None:
        """More folds with similar values should narrow the CI."""
        folds_few = self._make_folds([0.7, 0.8, 0.9])
        folds_many = self._make_folds([0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85])

        result_few = aggregate_folds(folds_few)
        result_many = aggregate_folds(folds_many)

        width_few = result_few.ci_auc[1] - result_few.ci_auc[0]
        width_many = result_many.ci_auc[1] - result_many.ci_auc[0]
        assert width_many < width_few

    def test_nan_folds_are_excluded_from_mean(self) -> None:
        """NaN AUC folds (degenerate) should not affect the mean."""
        folds = self._make_folds([0.8, np.nan, 0.9])
        result = aggregate_folds(folds)
        assert result.mean_auc == pytest.approx(0.85)

    def test_empty_folds_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            aggregate_folds([])

    def test_single_fold_has_zero_std(self) -> None:
        folds = self._make_folds([0.85])
        result = aggregate_folds(folds)
        assert result.std_auc == 0.0
        # CI should be a point
        assert result.ci_auc[0] == pytest.approx(result.ci_auc[1])

    def test_n_folds_property(self) -> None:
        folds = self._make_folds([0.7, 0.8, 0.9, 0.85])
        result = aggregate_folds(folds)
        assert result.n_folds == 4

    def test_summary_dict_keys(self) -> None:
        folds = self._make_folds([0.7, 0.8, 0.9])
        result = aggregate_folds(folds)
        d = result.summary_dict()
        expected_keys = {
            "n_folds", "auc_mean", "auc_std", "auc_ci_low", "auc_ci_high",
            "ap_mean", "ap_std", "brier_mean", "brier_std",
            "balanced_accuracy_mean", "balanced_accuracy_std",
        }
        assert set(d.keys()) == expected_keys

    def test_summary_dict_values_are_rounded(self) -> None:
        folds = self._make_folds([0.71234, 0.82345, 0.93456])
        result = aggregate_folds(folds)
        d = result.summary_dict()
        # All float values should have at most 4 decimal places
        for key, value in d.items():
            if isinstance(value, float):
                rounded = round(value, 4)
                assert value == rounded, f"{key} not rounded to 4 places"


# ---- calibration_error -------------------------------------------------------

class TestCalibrationError:
    def test_perfectly_calibrated(self) -> None:
        """A perfectly calibrated model has ECE = 0."""
        # All predictions match the true frequency exactly
        rng = np.random.default_rng(42)
        n = 1000
        y_score = rng.uniform(0, 1, n)
        y_true = (rng.uniform(0, 1, n) < y_score).astype(float)

        ece = calibration_error(y_true, y_score, n_bins=10)
        # Not exactly 0 due to finite samples, but should be small
        assert ece < 0.1

    def test_overconfident_model_has_high_ece(self) -> None:
        """A model that predicts 0 or 1 for 50/50 data has high ECE."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_score = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

        ece = calibration_error(y_true, y_score, n_bins=10)
        assert ece > 0.3

    def test_ece_is_nonnegative(self) -> None:
        rng = np.random.default_rng(7)
        y_true = rng.choice([0, 1], size=50)
        y_score = rng.uniform(0, 1, size=50)

        ece = calibration_error(y_true, y_score)
        assert ece >= 0.0

    def test_ece_at_most_one(self) -> None:
        """ECE is bounded by 1 since it is a weighted average of |freq - pred|."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_score = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        ece = calibration_error(y_true, y_score, n_bins=5)
        assert ece <= 1.0

    def test_invalid_n_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            calibration_error(np.array([0, 1]), np.array([0.3, 0.7]), n_bins=0)

    def test_single_bin_equals_overall_gap(self) -> None:
        """With 1 bin, ECE = |mean(y) - mean(p)|."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.2, 0.3, 0.8, 0.7, 0.9])

        ece = calibration_error(y_true, y_score, n_bins=1)
        expected = abs(np.mean(y_true) - np.mean(y_score))
        assert ece == pytest.approx(expected)


# ---- net_reclassification_improvement ----------------------------------------

class TestNRI:
    def test_identical_models_give_zero_nri(self) -> None:
        """Two identical models should have NRI = 0."""
        y_true = np.array([0, 0, 1, 1])
        risk = np.array([0.3, 0.4, 0.7, 0.8])

        result = net_reclassification_improvement(y_true, risk, risk)
        assert result["nri_total"] == pytest.approx(0.0)

    def test_perfect_new_model_has_positive_nri(self) -> None:
        """A perfect new model vs. random old model gives positive NRI."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        risk_old = np.array([0.6, 0.4, 0.7, 0.3, 0.6, 0.4])
        risk_new = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.9])

        result = net_reclassification_improvement(y_true, risk_old, risk_new)
        assert result["nri_total"] > 0

    def test_worse_model_has_negative_nri(self) -> None:
        """A worse new model should have negative NRI."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        risk_old = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.9])
        risk_new = np.array([0.6, 0.4, 0.7, 0.3, 0.6, 0.4])

        result = net_reclassification_improvement(y_true, risk_old, risk_new)
        assert result["nri_total"] < 0

    def test_nri_components_sum(self) -> None:
        """NRI total should equal events + nonevents components."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        risk_old = np.array([0.3, 0.6, 0.4, 0.7, 0.5, 0.8])
        risk_new = np.array([0.2, 0.4, 0.6, 0.9, 0.3, 0.7])

        result = net_reclassification_improvement(y_true, risk_old, risk_new)
        assert result["nri_total"] == pytest.approx(
            result["nri_events"] + result["nri_nonevents"], abs=1e-4
        )

    def test_no_events_returns_nan(self) -> None:
        """If there are no positive cases, NRI should be NaN."""
        y_true = np.array([0, 0, 0])
        risk_old = np.array([0.1, 0.2, 0.3])
        risk_new = np.array([0.4, 0.5, 0.6])

        result = net_reclassification_improvement(y_true, risk_old, risk_new)
        assert np.isnan(result["nri_total"])

    def test_no_nonevents_returns_nan(self) -> None:
        """If there are no negative cases, NRI should be NaN."""
        y_true = np.array([1, 1, 1])
        risk_old = np.array([0.6, 0.7, 0.8])
        risk_new = np.array([0.5, 0.9, 0.7])

        result = net_reclassification_improvement(y_true, risk_old, risk_new)
        assert np.isnan(result["nri_total"])

    def test_antisymmetry(self) -> None:
        """Swapping old and new models should negate NRI."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        risk_a = np.array([0.3, 0.6, 0.4, 0.7, 0.2, 0.8])
        risk_b = np.array([0.2, 0.4, 0.6, 0.9, 0.3, 0.7])

        forward = net_reclassification_improvement(y_true, risk_a, risk_b)
        backward = net_reclassification_improvement(y_true, risk_b, risk_a)

        assert forward["nri_total"] == pytest.approx(
            -backward["nri_total"], abs=1e-4
        )
