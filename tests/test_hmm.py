from __future__ import annotations

import numpy as np

from tbi_markov.hmm import DiagonalGaussianHMM


def _make_short_sequences(
    n_sequences: int = 240, seed: int = 812
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    start = np.array([0.72, 0.22, 0.06])
    transition = np.array(
        [
            [0.76, 0.22, 0.02],
            [0.13, 0.65, 0.22],  # recovery and worsening are both possible
            [0.03, 0.18, 0.79],
        ]
    )
    means = np.array([[-2.5, -1.5], [0.0, 0.2], [2.8, 2.0]])
    scales = np.array([[0.35, 0.45], [0.40, 0.35], [0.45, 0.40]])
    sequences: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for index in range(n_sequences):
        length = 2 + index % 2
        latent = np.empty(length, dtype=int)
        latent[0] = rng.choice(3, p=start)
        for time in range(1, length):
            latent[time] = rng.choice(3, p=transition[latent[time - 1]])
        observations = rng.normal(means[latent], scales[latent])
        sequences.append(observations)
        states.append(latent)
    return sequences, states


def _fit_model() -> tuple[DiagonalGaussianHMM, list[np.ndarray], list[np.ndarray]]:
    sequences, states = _make_short_sequences()
    model = DiagonalGaussianHMM(
        3,
        random_state=91,
        n_restarts=5,
        n_iter=120,
        tol=1e-6,
        transition_pseudocount=0.05,
    ).fit(sequences)
    return model, sequences, states


def test_probability_parameters_and_filter_are_normalized() -> None:
    model, sequences, _ = _fit_model()
    probabilities = model.predict_proba(sequences[:12])

    np.testing.assert_allclose(model.startprob_.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        model.transmat_.sum(axis=1), np.ones(3), atol=1e-12
    )
    np.testing.assert_allclose(
        probabilities.sum(axis=1), np.ones(len(probabilities)), atol=1e-12
    )
    assert np.all(model.startprob_ > 0)
    assert np.all(model.transmat_ > 0)
    assert np.all(model.covars_ >= model.min_covar)


def test_fit_recovers_well_separated_short_sequence_states() -> None:
    model, sequences, true_states = _fit_model()
    predicted = model.predict(sequences)
    truth = np.concatenate(true_states)

    # Emission means are ordered by their first feature to resolve label
    # non-identifiability before comparing with the planted severity states.
    learned_order = np.argsort(model.means_[:, 0])
    raw_to_ordered = np.empty(3, dtype=int)
    raw_to_ordered[learned_order] = np.arange(3)
    recovered = raw_to_ordered[predicted]

    assert np.mean(recovered == truth) > 0.88
    assert np.isfinite(model.score(sequences))


def test_variable_lengths_and_list_input_are_equivalent() -> None:
    model, sequences, _ = _fit_model()
    subset = [sequences[0], sequences[1], sequences[2], sequences[3]]
    concatenated = np.vstack(subset)
    lengths = [len(sequence) for sequence in subset]

    np.testing.assert_allclose(
        model.filter(subset), model.filter(concatenated, lengths), atol=1e-12
    )
    assert model.predict(concatenated, lengths).shape == (sum(lengths),)
    separate_score = sum(model.score(sequence) for sequence in subset)
    np.testing.assert_allclose(
        model.score(concatenated, lengths), separate_score, atol=1e-10
    )


def test_forward_filter_is_prefix_invariant() -> None:
    model, sequences, _ = _fit_model()
    sequence = sequences[1]  # this generated sequence has length three
    full = model.predict_proba(sequence)

    for prefix_length in range(1, len(sequence) + 1):
        prefix = model.filter(sequence[:prefix_length])
        np.testing.assert_allclose(
            prefix, full[:prefix_length], rtol=1e-12, atol=1e-12
        )
