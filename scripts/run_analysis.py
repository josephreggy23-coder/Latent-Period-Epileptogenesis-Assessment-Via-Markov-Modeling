"""Command-line wrapper: ingest the source workbooks and run the analysis.

Normalizes the LFP and behavioral workbooks into `data/measured/`, then fits the
HMM and scores the causal 4-5 dpf to 6 dpf forecast into `results/`.
"""

from tbi_markov.dataset import main


if __name__ == "__main__":
    main()
