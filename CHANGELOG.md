# Changelog

All notable project changes are documented here.

## 2.0.0 — 2026-07-25

**Breaking.** The synthetic simulator and every artifact derived from it are
removed. The simulator existed to validate that the estimator recovers known
latent states; that job is done, and keeping it alongside a measured cohort
invited confusion about which numbers describe real animals. The repository is
now a single-purpose analysis of one measured recording.

### Removed

- `tbi_markov.synthetic` (simulator), `tbi_markov.pipeline`, `data/synthetic/`,
  and the simulator's committed results.
- `tbi-generate` and `tbi-pipeline` entry points; `scripts/generate_dataset.py`
  and `scripts/run_pipeline.py`.
- The `is_synthetic` provenance flag, the `hidden_state_TRUTH` column, and
  `state_recovery_metrics` — with no simulator there is no latent-state ground
  truth, so state recovery is not measurable and no proxy is reported.
- The state-recovery confusion-matrix figure.

### Changed

- `tbi_markov.real_data` → `tbi_markov.dataset`; `data/real/` →
  `data/measured/`; `results_real/` → `results/`. Names contrasted with the
  simulator, which no longer exists.
- The endpoint column is renamed `high_burden_state_dpf6`. It was previously
  suffixed `_TRUTH` for schema compatibility with planted data, which misstated
  a behavioral observation as ground truth.
- `tbi-analyze` now runs the full ingest-and-analyze workflow; `python -m
  tbi_markov` does the same.
- `run_analysis` loses the provenance-switching parameters (`benchmark_type`,
  `critical_caveat`, `figure_labels`, `report_writer`) that existed only to keep
  one code path serving two data sources. Figure captions and the report are no
  longer parameterized.
- The behavioral-validation note no longer describes values as simulated; it
  states that locomotor speed is non-monotone in dose because pressures above
  ~300 kPa suppress movement.
- Tests build small in-memory fixtures instead of calling the simulator. The
  leakage, split, prefix, and propagation tests are unchanged in substance.
- README, `docs/METHODS.md`, `docs/REPRODUCIBILITY.md`, `data/README.md`,
  `CONTRIBUTING.md`, `CITATION.cff`, and the PR template are rewritten for a
  measured-data project.

### Note

Results are unchanged: ROC-AUC 0.749 (95% CI 0.642–0.853) on 71 held-out fish.
Removing the simulator removed no evidence bearing on the measured cohort.
Earlier commits remain in git history; this release changes the working tree,
not the past.

## 1.2.0 — 2026-07-25

### Added

- `docs/EXPERIMENTAL_PROTOCOL.md`: the full wet-lab protocol — apparatus,
  pressure calibration and conversion, plate layout, MCAM imaging, pose
  estimation and Baraban staging, LFP acquisition, and required metadata — with
  an explicit map of where the three published methods end and the new
  integration begins.
- README sections for the real apparatus (20 mL syringe, three-prong clamp,
  108 cm, single drop at 100/200/300 g, measured 115/210/319 kPa) and the three
  constraints that bound the real-data claims.
- `n_negative`, `n_unresolved`, and per-group unresolved counts in the real
  manifest and report.

### Fixed

- **The 6 dpf endpoint is now three-valued.** Fish never observed at 6 dpf were
  being coded `0` (no high-burden state) rather than `NA`. Seven fish had no
  6 dpf LFP session and no 6 dpf behavioural row, so they were counted as
  confirmed negatives despite never having been checked. They are now `NA` and
  excluded from endpoint scoring.

  This corrects the headline: the held-out forecast is **ROC-AUC 0.749**
  (95% CI 0.642–0.853) on 71 fish with 19 positives, not the previously
  reported 0.830 on 72 fish with 18 positives. The earlier figure was inflated
  by unobserved animals padding the negative class.

### Changed

- Documented that the electrode metadata (forebrain, 1 M chloride, 2.45–3.57 MΩ)
  matches the Eimon penetrating preparation, which was demonstrated at 7 dpf and
  never validated as a recoverable repeated measurement at 4–6 dpf — so per-fish
  longitudinal state transitions rest on an unverifiable assumption.
- Noted that `insult_batch_id` is absent, so the drop batch cannot enter the
  model as the experimental unit the protocol requires.
- Citation guidance now separates the synthetic benchmark from the retrospective
  single-cohort real analysis.

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
