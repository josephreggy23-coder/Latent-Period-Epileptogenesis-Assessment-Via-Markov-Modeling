# Contributing

Contributions should preserve the project's central boundary: this is a
retrospective analysis of a single measured cohort, and conclusions must not be
stated more strongly than that.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

## Pull-request checklist

- Keep scientific claims within what one retrospective cohort supports.
- Add or update tests for ingestion, schema, temporal, or model changes.
- Preserve fish-level splitting and train-only preprocessing.
- Do not add endpoint, dose, group, QC, or behavior columns to the HMM feature
  matrix.
- Keep unobserved outcomes as `NA`; never code an unchecked animal as negative.
- Regenerate committed results when numerical behavior changes.
- Update `README.md`, `docs/`, and `CHANGELOG.md` when interfaces change.
- Confirm `python -m pytest` and `python -m compileall -q src scripts` pass.

## Style

- Prefer type annotations and focused functions.
- Use deterministic random-number generators with explicit seeds.
- Keep output paths relative to the repository root.
- Document units and provenance, and state when a quantity is derived rather
  than directly measured.
- Avoid claims of biological efficacy or clinical prediction.
