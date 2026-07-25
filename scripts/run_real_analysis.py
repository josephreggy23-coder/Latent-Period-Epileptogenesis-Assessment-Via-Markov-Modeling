"""Command-line wrapper for the REAL-recording HMM analysis.

Normalizes the real workbooks into `data/real/`, then runs the same causal
4-5 dpf to 6 dpf forecast used by the synthetic benchmark. State recovery is
not scored: real animals have no planted latent state.
"""

from tbi_markov.real_data import main


if __name__ == "__main__":
    main()
