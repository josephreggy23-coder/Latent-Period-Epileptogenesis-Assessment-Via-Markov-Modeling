# Reproducibility

## Deterministic default run

The committed analysis uses:

- seed: `42`;
- 70%/30% fish-level split;
- HMM candidates: 2 and 3 states (see `docs/PREREGISTRATION.md`);
- three train-only cross-validation folds;
- 1,000 bootstrap iterations for forecast intervals.

Run the complete workflow with:

```bash
python -m tbi_markov
```

This re-ingests the source workbooks, so `data/measured/` and `results/` are
both regenerated from the raw inputs on every run.

## Output contract

```text
data/measured/
├── tbi_4_6dpf_lfp_timeseries.csv
├── tbi_4_6dpf_fish_outcomes.csv
├── tbi_4_6dpf_behavior.csv
└── tbi_4_6dpf_manifest.json

results/
├── figures/
├── tables/
├── TBI_MODEL_RESULTS.md
└── tbi_model_metrics.json
```

The source workbooks at the repository root are the only non-regenerable
inputs. Everything under `data/measured/` and `results/` is derived and safe to
delete; the next run rebuilds it.

Because these are real animals with no latent-state ground truth, the run
reports no state-recovery metric and writes no confusion-matrix figure. That
absence is intentional, not a missing output.

## Integrity controls

- The manifest records cohort scope, arm counts, and the endpoint definition
  with its positive/negative/unresolved counts.
- Feature allowlisting prevents protocol, dose, behavior, or endpoint leakage.
- Validation asserts the recorded `qc_pass` flag reproduces the published
  electrode-shift/noise rule exactly.
- Preprocessing is fit on training fish only.
- Train/test fish IDs are disjoint.
- QC gaps terminate sequences instead of shortening time.
- Prefix invariance ensures a 5 dpf forecast does not depend on 6 dpf features.
- Forecast tests verify the transition-matrix propagation horizon.
- Fish never observed at 6 dpf carry an `NA` endpoint and are excluded from
  scoring rather than counted as negatives.

## Verification

```bash
python -m pytest
python -m compileall -q src scripts
```

Tests that require the source workbooks skip automatically when the workbooks
are absent, so the suite runs on a clone without them.

## Changing the configuration

```bash
tbi-analyze --seed 101 --test-fraction 0.25 --states 2 3 4 5
```

Any change to the seed, split, or model-selection grid changes the committed
metrics; regenerate `results/` and note the change in the changelog.
