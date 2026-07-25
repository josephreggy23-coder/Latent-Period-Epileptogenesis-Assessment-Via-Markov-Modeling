"""Explicitly reset synthetic template data and run the demonstration benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    DATA_DIR,
    DLC_CSV,
    LFP_CSV,
    OUTCOMES_CSV,
    RESULTS_DIR,
    SEED,
    load_dataset,
)
from .modeling import run_analysis
from .template_data import write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-per-arm", type=int, default=60)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--states", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument(
        "--force-reset-demo",
        action="store_true",
        help=(
            "Required to overwrite template data. Back up measured records "
            "before using this option."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.force_reset_demo:
        raise SystemExit(
            "Refusing to reset template data without --force-reset-demo. "
            "Use tbi-analyze --allow-placeholder-data for a non-resetting demo."
        )
    manifest = write_dataset(
        args.data_dir,
        seed=args.seed,
        n_per_arm=args.n_per_arm,
        force=True,
    )
    lfp, outcomes, dlc = load_dataset(
        args.data_dir / LFP_CSV.name,
        args.data_dir / OUTCOMES_CSV.name,
        args.data_dir / DLC_CSV.name,
    )
    metrics = run_analysis(
        lfp,
        outcomes,
        dlc,
        output_dir=args.output_dir,
        seed=args.seed,
        candidates=args.states,
        restarts=args.restarts,
        cv_folds=args.cv_folds,
        bootstrap_iterations=args.bootstrap_iterations,
        allow_placeholder_data=True,
    )
    print(
        f"SYNTHETIC PLACEHOLDER DEMO ONLY: initialized {manifest['n_fish']} fish; selected "
        f"K={metrics['selected_states']} statistical microstates; held-out "
        f"DPF6 forecast AUC={metrics['early_prediction']['roc_auc']:.3f}."
    )


if __name__ == "__main__":
    main()
