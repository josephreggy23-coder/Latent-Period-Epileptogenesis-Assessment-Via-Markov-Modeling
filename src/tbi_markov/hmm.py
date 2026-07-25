"""Core dependency-light diagonal Gaussian hidden Markov model.

This implementation is intended for longitudinal zebrafish data, where each
animal contributes only a few observations.  Unlike a classifier that treats
rows independently, the model learns the initial latent-state distribution
and a full transition matrix (including both recovery and worsening).

Only NumPy and SciPy are required.  The public API follows the useful subset of
``hmmlearn``: concatenate sequences in ``X`` and pass their ``lengths``, or pass
a list of two-dimensional arrays directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import logsumexp


ArrayLike = np.ndarray | Sequence[Sequence[float]]


@dataclass
class _FitResult:
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    history: list[float]
    converged: bool

    @property
    def log_likelihood(self) -> float:
        return self.history[-1]


class DiagonalGaussianHMM:
    """Finite-state HMM with diagonal Gaussian emission distributions.

    Parameters
    ----------
    n_components:
        Number of latent states.
    n_iter, tol:
        Maximum EM iterations and convergence tolerance in log likelihood.
    n_restarts:
        Number of deterministic-from-seed initializations.  The fit with the
        highest final likelihood is retained.
    random_state:
        Seed used to derive every restart.
    min_covar:
        Elementwise lower bound on emission variances.
    variance_regularization:
        Strength of shrinkage of state variances toward the pooled variance.
        This is especially useful for two- or three-observation sequences.
    start_pseudocount, transition_pseudocount:
        Symmetric Dirichlet pseudocounts used in the EM probability updates.
        Positive transition pseudocounts preserve a full transition matrix, so
        both worsening and recovery remain possible even if rare in training.

    Notes
    -----
    ``predict_proba`` is deliberately causal: it is an alias for ``filter`` and
    returns :math:`P(z_t | x_1, ..., x_t)`.  EM itself uses the usual smoothed
    forward-backward responsibilities.
    """

    def __init__(
        self,
        n_components: int,
        *,
        n_iter: int = 200,
        tol: float = 1e-5,
        n_restarts: int = 5,
        random_state: int | None = 0,
        min_covar: float = 1e-4,
        variance_regularization: float = 1e-2,
        start_pseudocount: float = 1e-2,
        transition_pseudocount: float = 1e-2,
        verbose: bool = False,
    ) -> None:
        if int(n_components) != n_components or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if n_iter < 1 or n_restarts < 1:
            raise ValueError("n_iter and n_restarts must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if min_covar <= 0 or variance_regularization < 0:
            raise ValueError(
                "min_covar must be positive and variance_regularization non-negative"
            )
        if start_pseudocount <= 0 or transition_pseudocount <= 0:
            raise ValueError("probability pseudocounts must be positive")

        self.n_components = int(n_components)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.n_restarts = int(n_restarts)
        self.random_state = random_state
        self.min_covar = float(min_covar)
        self.variance_regularization = float(variance_regularization)
        self.start_pseudocount = float(start_pseudocount)
        self.transition_pseudocount = float(transition_pseudocount)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: ArrayLike | Sequence[np.ndarray],
        lengths: Sequence[int] | None = None,
    ) -> "DiagonalGaussianHMM":
        """Estimate parameters by Baum-Welch EM and return ``self``."""
        data, seq_lengths = self._coerce_sequences(X, lengths)
        if len(data) < self.n_components:
            raise ValueError(
                "at least n_components observations are required to fit the model"
            )

        self.n_features_in_ = data.shape[1]
        pooled_var = np.maximum(np.var(data, axis=0), self.min_covar)
        base_seed = 0 if self.random_state is None else int(self.random_state)
        best: _FitResult | None = None

        for restart in range(self.n_restarts):
            rng = np.random.default_rng(
                np.random.SeedSequence([base_seed, restart, self.n_components])
            )
            startprob, transmat, means, covars = self._initialize(
                data, pooled_var, rng
            )
            history: list[float] = []
            converged = False

            for iteration in range(self.n_iter):
                stats, log_likelihood = self._expectation(
                    data, seq_lengths, startprob, transmat, means, covars
                )
                history.append(log_likelihood)

                startprob = self._normalize(
                    stats["start"] + self.start_pseudocount
                )
                transmat = self._normalize_rows(
                    stats["trans"] + self.transition_pseudocount
                )

                weights = stats["weights"]
                safe_weights = np.maximum(weights, np.finfo(float).tiny)
                means = stats["obs"] / safe_weights[:, None]

                # Recompute centered second moments using the new means.  The
                # pooled-variance prior prevents a rare state from collapsing.
                squared_delta = (data[:, None, :] - means[None, :, :]) ** 2
                centered_ss = np.sum(
                    stats["gammas"][:, :, None] * squared_delta, axis=0
                )
                reg = self.variance_regularization
                covars = (centered_ss + reg * pooled_var) / (
                    weights[:, None] + reg
                )
                covars = np.maximum(covars, self.min_covar)

                if iteration > 0:
                    gain = history[-1] - history[-2]
                    if abs(gain) <= self.tol * (1.0 + abs(history[-2])):
                        converged = True
                        break

            # Store a likelihood corresponding to the final, updated model.
            final_ll = self._score_parameters(
                data, seq_lengths, startprob, transmat, means, covars
            )
            if not history or not np.isclose(final_ll, history[-1]):
                history.append(final_ll)
            candidate = _FitResult(
                startprob, transmat, means, covars, history, converged
            )
            if best is None or candidate.log_likelihood > best.log_likelihood:
                best = candidate
            if self.verbose:
                print(
                    f"restart {restart + 1}/{self.n_restarts}: "
                    f"log likelihood={candidate.log_likelihood:.6f}"
                )

        if best is None or not np.isfinite(best.log_likelihood):
            raise RuntimeError("HMM fitting failed to produce a finite likelihood")

        self.startprob_ = best.startprob
        self.transmat_ = best.transmat
        self.means_ = best.means
        self.covars_ = best.covars
        self.monitor_ = list(best.history)
        self.history_ = self.monitor_
        self.log_likelihood_ = float(best.log_likelihood)
        self.converged_ = bool(best.converged)
        self.n_iter_ = len(best.history)
        return self

    def score(
        self,
        X: ArrayLike | Sequence[np.ndarray],
        lengths: Sequence[int] | None = None,
    ) -> float:
        """Return the total log likelihood across independent sequences."""
        self._check_fitted()
        data, seq_lengths = self._coerce_sequences(X, lengths)
        self._check_features(data)
        return self._score_parameters(
            data,
            seq_lengths,
            self.startprob_,
            self.transmat_,
            self.means_,
            self.covars_,
        )

    def predict(
        self,
        X: ArrayLike | Sequence[np.ndarray],
        lengths: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Return the most likely (Viterbi) state path for each sequence."""
        self._check_fitted()
        data, seq_lengths = self._coerce_sequences(X, lengths)
        self._check_features(data)
        log_emissions = self._log_emission_prob(data, self.means_, self.covars_)
        log_start = self._safe_log(self.startprob_)
        log_trans = self._safe_log(self.transmat_)
        paths: list[np.ndarray] = []
        offset = 0

        for length in seq_lengths:
            emissions = log_emissions[offset : offset + length]
            delta = np.empty((length, self.n_components))
            backptr = np.zeros((length, self.n_components), dtype=np.int64)
            delta[0] = log_start + emissions[0]
            for time in range(1, length):
                candidates = delta[time - 1, :, None] + log_trans
                backptr[time] = np.argmax(candidates, axis=0)
                delta[time] = np.max(candidates, axis=0) + emissions[time]
            path = np.empty(length, dtype=np.int64)
            path[-1] = int(np.argmax(delta[-1]))
            for time in range(length - 2, -1, -1):
                path[time] = backptr[time + 1, path[time + 1]]
            paths.append(path)
            offset += length
        return np.concatenate(paths)

    def filter(
        self,
        X: ArrayLike | Sequence[np.ndarray],
        lengths: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Return causal state probabilities based only on each prefix."""
        self._check_fitted()
        data, seq_lengths = self._coerce_sequences(X, lengths)
        self._check_features(data)
        log_emissions = self._log_emission_prob(data, self.means_, self.covars_)
        log_start = self._safe_log(self.startprob_)
        log_trans = self._safe_log(self.transmat_)
        probabilities: list[np.ndarray] = []
        offset = 0

        for length in seq_lengths:
            emissions = log_emissions[offset : offset + length]
            log_filter = np.empty((length, self.n_components))
            first = log_start + emissions[0]
            log_filter[0] = first - logsumexp(first)
            for time in range(1, length):
                current = emissions[time] + logsumexp(
                    log_filter[time - 1, :, None] + log_trans, axis=0
                )
                log_filter[time] = current - logsumexp(current)
            probabilities.append(np.exp(log_filter))
            offset += length
        return np.vstack(probabilities)

    def predict_proba(
        self,
        X: ArrayLike | Sequence[np.ndarray],
        lengths: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Alias for :meth:`filter`; probabilities are forward-only."""
        return self.filter(X, lengths)

    # ------------------------------------------------------------------
    # EM and numerical internals
    # ------------------------------------------------------------------
    def _initialize(
        self,
        data: np.ndarray,
        pooled_var: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """K-means++ emissions plus mildly persistent random dynamics."""
        n_samples = len(data)
        means = np.empty((self.n_components, data.shape[1]), dtype=float)
        means[0] = data[rng.integers(n_samples)]
        closest_sq = np.sum((data - means[0]) ** 2, axis=1)
        for state in range(1, self.n_components):
            total = closest_sq.sum()
            if total <= np.finfo(float).tiny:
                choice = rng.integers(n_samples)
            else:
                choice = rng.choice(n_samples, p=closest_sq / total)
            means[state] = data[choice]
            closest_sq = np.minimum(
                closest_sq, np.sum((data - means[state]) ** 2, axis=1)
            )

        # A few Lloyd steps make each restart a strong but still seed-dependent
        # candidate without introducing a scikit-learn dependency.
        for _ in range(8):
            distances = np.sum((data[:, None, :] - means[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            for state in range(self.n_components):
                members = data[labels == state]
                if len(members):
                    means[state] = members.mean(axis=0)

        covars = np.tile(pooled_var, (self.n_components, 1))
        startprob = rng.dirichlet(np.full(self.n_components, 1.5))
        transmat = np.empty((self.n_components, self.n_components))
        for state in range(self.n_components):
            concentration = np.full(self.n_components, 1.0)
            concentration[state] = 3.0
            transmat[state] = rng.dirichlet(concentration)
        return startprob, transmat, means, covars

    def _expectation(
        self,
        data: np.ndarray,
        lengths: np.ndarray,
        startprob: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
    ) -> tuple[dict[str, np.ndarray | list[np.ndarray]], float]:
        log_emissions = self._log_emission_prob(data, means, covars)
        log_start = self._safe_log(startprob)
        log_trans = self._safe_log(transmat)
        start_counts = np.zeros(self.n_components)
        trans_counts = np.zeros((self.n_components, self.n_components))
        all_gamma = np.empty((len(data), self.n_components))
        total_ll = 0.0

        starts = np.r_[0, np.cumsum(lengths[:-1])]
        for length in np.unique(lengths):
            group_starts = starts[lengths == length]
            indices = group_starts[:, None] + np.arange(length)[None, :]
            emissions = log_emissions[indices]

            alpha = np.empty_like(emissions)
            alpha[:, 0] = log_start + emissions[:, 0]
            for time in range(1, length):
                alpha[:, time] = emissions[:, time] + logsumexp(
                    alpha[:, time - 1, :, None] + log_trans[None, :, :],
                    axis=1,
                )
            log_likelihood = logsumexp(alpha[:, -1], axis=1)

            beta = np.zeros_like(emissions)
            for time in range(length - 2, -1, -1):
                beta[:, time] = logsumexp(
                    log_trans[None, :, :]
                    + emissions[:, time + 1, None, :]
                    + beta[:, time + 1, None, :],
                    axis=2,
                )

            log_gamma = alpha + beta - log_likelihood[:, None, None]
            log_gamma -= logsumexp(log_gamma, axis=2, keepdims=True)
            gamma = np.exp(log_gamma)
            all_gamma[indices] = gamma
            start_counts += gamma[:, 0].sum(axis=0)

            for time in range(length - 1):
                log_xi = (
                    alpha[:, time, :, None]
                    + log_trans[None, :, :]
                    + emissions[:, time + 1, None, :]
                    + beta[:, time + 1, None, :]
                    - log_likelihood[:, None, None]
                )
                log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
                trans_counts += np.exp(log_xi).sum(axis=0)
            total_ll += float(log_likelihood.sum())

        stats: dict[str, np.ndarray | list[np.ndarray]] = {
            "start": start_counts,
            "trans": trans_counts,
            "weights": all_gamma.sum(axis=0),
            "obs": all_gamma.T @ data,
            "gammas": all_gamma,
        }
        return stats, float(total_ll)

    @staticmethod
    def _forward(
        log_emissions: np.ndarray,
        log_start: np.ndarray,
        log_trans: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        alpha = np.empty_like(log_emissions)
        alpha[0] = log_start + log_emissions[0]
        for time in range(1, len(log_emissions)):
            alpha[time] = log_emissions[time] + logsumexp(
                alpha[time - 1, :, None] + log_trans, axis=0
            )
        return alpha, float(logsumexp(alpha[-1]))

    @staticmethod
    def _backward(
        log_emissions: np.ndarray, log_trans: np.ndarray
    ) -> np.ndarray:
        beta = np.zeros_like(log_emissions)
        for time in range(len(log_emissions) - 2, -1, -1):
            beta[time] = logsumexp(
                log_trans
                + log_emissions[time + 1, None, :]
                + beta[time + 1, None, :],
                axis=1,
            )
        return beta

    def _score_parameters(
        self,
        data: np.ndarray,
        lengths: np.ndarray,
        startprob: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
    ) -> float:
        log_emissions = self._log_emission_prob(data, means, covars)
        log_start = self._safe_log(startprob)
        log_trans = self._safe_log(transmat)
        total = 0.0
        starts = np.r_[0, np.cumsum(lengths[:-1])]
        for length in np.unique(lengths):
            group_starts = starts[lengths == length]
            indices = group_starts[:, None] + np.arange(length)[None, :]
            emissions = log_emissions[indices]
            alpha = log_start[None, :] + emissions[:, 0]
            for time in range(1, length):
                alpha = emissions[:, time] + logsumexp(
                    alpha[:, :, None] + log_trans[None, :, :], axis=1
                )
            total += float(logsumexp(alpha, axis=1).sum())
        return float(total)

    @staticmethod
    def _log_emission_prob(
        data: np.ndarray, means: np.ndarray, covars: np.ndarray
    ) -> np.ndarray:
        n_features = data.shape[1]
        delta = data[:, None, :] - means[None, :, :]
        mahalanobis = np.sum(delta * delta / covars[None, :, :], axis=2)
        log_determinant = np.sum(np.log(covars), axis=1)
        return -0.5 * (
            n_features * np.log(2.0 * np.pi)
            + log_determinant[None, :]
            + mahalanobis
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_sequences(
        X: ArrayLike | Sequence[np.ndarray],
        lengths: Sequence[int] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if lengths is None and isinstance(X, (list, tuple)) and len(X):
            first = np.asarray(X[0])
            if first.ndim == 2:
                sequences = [np.asarray(sequence, dtype=float) for sequence in X]
                if any(sequence.ndim != 2 for sequence in sequences):
                    raise ValueError("every sequence must be a two-dimensional array")
                feature_counts = {sequence.shape[1] for sequence in sequences}
                if len(feature_counts) != 1:
                    raise ValueError("all sequences must have the same feature count")
                if any(len(sequence) < 1 for sequence in sequences):
                    raise ValueError("empty sequences are not allowed")
                data = np.vstack(sequences)
                seq_lengths = np.asarray([len(sequence) for sequence in sequences])
            else:
                data = np.asarray(X, dtype=float)
                seq_lengths = np.asarray([len(data)])
        else:
            data = np.asarray(X, dtype=float)
            if data.ndim == 1:
                data = data[:, None]
            if data.ndim != 2:
                raise ValueError("X must be a two-dimensional array")
            if lengths is None:
                seq_lengths = np.asarray([len(data)])
            else:
                raw_lengths = np.asarray(lengths)
                if raw_lengths.ndim != 1 or not np.issubdtype(
                    raw_lengths.dtype, np.integer
                ):
                    raise ValueError("lengths must be a one-dimensional integer array")
                seq_lengths = raw_lengths.astype(np.int64, copy=False)

        if data.ndim != 2 or data.shape[1] < 1:
            raise ValueError("X must contain at least one feature")
        if not np.all(np.isfinite(data)):
            raise ValueError("X must contain only finite values")
        if len(data) < 1 or len(seq_lengths) < 1:
            raise ValueError("at least one observation and sequence are required")
        if np.any(seq_lengths <= 0) or int(seq_lengths.sum()) != len(data):
            raise ValueError("lengths must be positive and sum to len(X)")
        return np.ascontiguousarray(data, dtype=float), seq_lengths

    def _check_fitted(self) -> None:
        required = ("startprob_", "transmat_", "means_", "covars_")
        if not all(hasattr(self, attribute) for attribute in required):
            raise RuntimeError("model is not fitted; call fit before inference")

    def _check_features(self, data: np.ndarray) -> None:
        if data.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {data.shape[1]} features; expected {self.n_features_in_}"
            )

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        return values / values.sum()

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        return values / values.sum(axis=1, keepdims=True)

    @staticmethod
    def _safe_log(values: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(values)


# Short aliases make the class easy to discover in scripts and notebooks.
GaussianHMM = DiagonalGaussianHMM
TBIHMM = DiagonalGaussianHMM

__all__ = ["DiagonalGaussianHMM", "GaussianHMM", "TBIHMM"]
