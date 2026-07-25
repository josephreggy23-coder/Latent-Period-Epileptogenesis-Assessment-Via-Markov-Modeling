# Contributing

Contributions should preserve the project's central boundary: committed data
and placeholder conclusions must remain explicitly demonstration-only.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

## Pull-request checklist

- Keep scientific claims distinct from template assumptions.
- Add or update tests for template initialization, schema, temporal, or model changes.
- Preserve fish-level splitting and train-only preprocessing.
- Do not add truth, dose, group, QC, or behavior columns to the HMM feature
  matrix.
- Regenerate committed results when numerical behavior changes.
- Update `README.md`, `docs/`, and `CHANGELOG.md` when interfaces change.
- Confirm `python -m pytest` and `python -m compileall -q src scripts` pass.

## Style

- Prefer type annotations and focused functions.
- Use deterministic template initialization with explicit seeds.
- Keep output paths relative to the repository root.
- Document units, provenance, and whether a value is measured or a placeholder.
- Avoid claims of biological efficacy or disease prediction from placeholder
  benchmark performance.
