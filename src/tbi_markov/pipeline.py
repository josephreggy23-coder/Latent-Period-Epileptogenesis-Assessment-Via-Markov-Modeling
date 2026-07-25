"""Regenerate the default synthetic TBI CSVs and run the Markov analysis."""
from .common import load_dataset
from .modeling import run_analysis
from .synthetic import write_dataset


def main() -> None:
    manifest = write_dataset()
    lfp, outcomes, dlc = load_dataset()
    metrics = run_analysis(lfp, outcomes, dlc)
    print(
        f"Generated {manifest['n_fish']} synthetic fish; selected "
        f"K={metrics['selected_states']} statistical microstates; held-out "
        f"DPF6 forecast AUC={metrics['early_prediction']['roc_auc']:.3f}."
    )


if __name__ == "__main__":
    main()
