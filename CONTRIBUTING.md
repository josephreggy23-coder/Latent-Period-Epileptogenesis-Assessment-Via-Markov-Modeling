# Contributing

Contributions should preserve the project's central boundary: committed data
and benchmark conclusions must remain explicitly synthetic.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

## Pull-request checklist

- Keep scientific claims distinct from simulator assumptions.
- Add or update tests for generator, schema, temporal, or model changes.
- Preserve fish-level splitting and train-only preprocessing.
- Do not add truth, dose, group, QC, or behavior columns to the HMM feature
  matrix.
- Regenerate committed results when numerical behavior changes.
- Update `README.md`, `docs/`, and `CHANGELOG.md` when interfaces change.
- Confirm `python -m pytest` and `python -m compileall -q src scripts` pass.

## Style

- Prefer type annotations and focused functions.
- Use deterministic random-number generators with explicit seeds.
- Keep output paths relative to the repository root.
- Document units, provenance, and whether a value is measured or simulated.
- Avoid claims of biological efficacy or disease prediction from synthetic
  benchmark performance.
