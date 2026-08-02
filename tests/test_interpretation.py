"""Tests for Task 6 state naming: the contract that no bare state index
survives into user-facing prose after this module runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tbi_markov.interpretation import (
    named_state,
    state_names,
    write_state_interpretation_report,
)


def test_state_names_defined_for_two_and_three_states():
    assert set(state_names(3)) == {0, 1, 2}
    assert set(state_names(2)) == {0, 1}
    assert all(isinstance(name, str) and name for name in state_names(3).values())


def test_state_names_rejects_unpreregistered_k():
    with pytest.raises(ValueError, match="PREREGISTRATION"):
        state_names(4)


def test_named_state_pairs_index_with_name():
    label = named_state(2, 3)
    assert label.startswith("2 (")
    assert state_names(3)[2] in label


def test_write_state_interpretation_report_three_states(tmp_path):
    means = np.array(
        [
            [-0.5, -0.4, 0.0, -0.5],
            [0.3, 0.3, 0.0, 0.3],
            [0.8, 0.9, 1.3, 0.9],
        ]
    )
    transition = np.array(
        [[0.8, 0.15, 0.05], [0.05, 0.5, 0.45], [0.01, 0.4, 0.59]]
    )
    start = np.array([0.6, 0.3, 0.1])
    occupancy = pd.DataFrame(
        {
            "group": ["sham", "tbi_high"] * 3,
            "dpf": [4, 4, 5, 5, 6, 6],
            "fraction_state_0": [0.9, 0.0] * 3,
            "fraction_state_1": [0.1, 0.3] * 3,
            "fraction_state_2": [0.0, 0.7] * 3,
        }
    )
    output = tmp_path / "STATE_INTERPRETATION.md"
    write_state_interpretation_report(output, 3, means, transition, start, occupancy)
    text = output.read_text(encoding="utf-8")
    for name in state_names(3).values():
        assert name in text
    assert "does not fit" in text.lower() or "does **not**" in text
    # Every "State N" occurrence must appear on the same line as its name --
    # never bare, per Task 6's naming contract.
    for line in text.splitlines():
        for index, name in state_names(3).items():
            if f"State {index}" in line:
                assert name in line, f"bare 'State {index}' without its name: {line!r}"
