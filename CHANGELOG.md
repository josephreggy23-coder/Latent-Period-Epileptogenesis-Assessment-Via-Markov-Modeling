# Changelog

All notable project changes are documented here.

## Unreleased

### Fixed

- Made `--help` side-effect free and added explicit overwrite protection to
  template initialization and the combined reset workflow.
- Evaluated rank-based forecast metrics at a documented six-decimal score
  precision so JSON and CSV results reproduce exactly.
- Strengthened schema validation for keys, domains, booleans, temporal
  relations, QC rules, and cross-table consistency.
- Renamed the synthetic electrode QC field to
  `electrode_resistance_change_pct` to match the source protocol.

### Changed

- Redesigned the README, figures, result narrative, and artifact navigation.
- Expanded machine-readable outputs with model parameters, convergence,
  configuration, software versions, normalized input hashes, cohort flow, and
  numerical-precision sensitivity.
- Clarified that behavior is generated pose-style concordance rather than
  independent validation and that raw feature extraction is out of scope.

## 1.1.0 — 2026-07-25

### Changed

- Renamed the editable dataset and workbook around a neutral template workflow.
- Replaced ambiguous provenance fields with `record_status` and `template_seed`.
- Added an analysis gate that blocks rows still marked
  `placeholder_pending_replacement` unless demonstration mode is explicitly
  requested.
- Renamed the initializer command to `tbi-initialize`.

## 1.0.0 — 2026-07-24

### Added

- Deterministic 4–6 dpf larval-zebrafish TBI placeholder template.
- Eimon-inspired LFP acquisition, QC, and feature interface.
- DeepLabCut-style behavioral validation table.
- Dependency-light diagonal-Gaussian HMM with worsening and recovery.
- Fish-level held-out evaluation and transition-propagated DPF6 forecast.
- Installable `src/` package, command-line entry points, CI, documentation, and
  reproducibility tests.

### Changed

- Replaced the former PTZ/pilocarpine project scope with a TBI-only benchmark.
- Organized data, figures, tables, reports, and package source into standard
  research-software directories.

### Removed

- Legacy PTZ, pilocarpine, valproate, microplastic, and status-epilepticus
  datasets, scripts, reports, and generated artifacts.
