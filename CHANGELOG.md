# Changelog

All notable project changes are documented here.

## 1.1.0 — 2026-07-25

### Added

- Real-data analysis path (`tbi_markov.real_data`, `tbi-real`,
  `scripts/run_real_analysis.py`): normalizes a measured 240-fish weight-drop
  TBI recording into the same three tables and runs the identical HMM,
  preprocessing, causal 4–5 dpf prefix rule, and 6 dpf propagation.
- Real 6 dpf endpoint derived from the blinded behavioral Event Log
  (Baraban stage ≥ 2 with passing pose QC), independent of every LFP input.
- Per-session aggregation of the per-event behavioral log, retaining zero-event
  sessions as observations rather than dropping them.
- Calibration block in the early-prediction metrics (observed positive rate vs
  mean/median forecast risk and the count above threshold), so a low sensitivity
  at the fixed 0.5 threshold cannot be misread as poor ranking.
- Real-data tests covering provenance enforcement in both directions, absence of
  planted truth, and endpoint independence; they skip when the source workbooks
  are absent.

### Changed

- `validate_dataset` takes `expect_synthetic` and `require_truth`; the
  `is_synthetic` flag is now asserted in both directions so synthetic and
  measured rows can never be mixed.
- `state_recovery_metrics` returns `None` when no planted truth exists, and the
  confusion-matrix figure is skipped rather than fabricated.
- `run_analysis` accepts `benchmark_type`, `critical_caveat`, `behavior_note`,
  `figure_labels`, and `report_writer`; figure captions are parameterized so a
  measured result is never labeled "synthetic" or "planted".
- Scope broadened from a synthetic-only benchmark to a synthetic benchmark plus
  a real-recording analysis path; README badge updated accordingly.

## 1.0.0 — 2026-07-24

### Added

- Deterministic 4–6 dpf larval-zebrafish TBI simulator.
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
