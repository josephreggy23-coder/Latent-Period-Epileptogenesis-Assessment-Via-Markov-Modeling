# Data

This directory holds two clearly separated datasets:

- **`synthetic/`** — simulator output. No row represents an experimental animal.
- **`real/`** — a normalized real 240-fish weight-drop TBI recording. Every row
  represents a real animal.

Every table in both directories carries an `is_synthetic` flag, and the loader
asserts it in **both** directions: a synthetic load rejects measured rows and a
real load rejects simulated rows. The two can never be silently mixed.

The real tables are generated from the source workbooks by `tbi-real`; only the
normalized CSVs and a provenance manifest are written here. The `real/` tables
carry **no** `hidden_state_TRUTH` column, because no planted latent state
exists — state recovery is unmeasurable on real data, not merely unreported.

## Authoritative files (synthetic)

The normalized analysis tables are stored in `data/synthetic/`:

| File | Rows | Purpose |
|---|---:|---|
| `tbi_4_6dpf_lfp_timeseries.csv` | 706 | Session protocol, QC, seven HMM features, planted state |
| `tbi_4_6dpf_fish_outcomes.csv` | 240 | Injury arm, attrition, and planted DPF6 endpoint |
| `tbi_4_6dpf_dlc_behavior.csv` | 706 | Synthetic pose-derived validation features |
| `tbi_4_6dpf_dataset_manifest.json` | — | Seed, scope, definitions, sources, and SHA-256 hashes |
| `TBI_4_6dpf_synthetic_data.xlsx` | — | Formatted seed-42 workbook for human review |

The CSV tables and manifest are authoritative after regeneration. The workbook
is a formatted snapshot of the default seed-42 cohort.

## Table relationships

```text
fish_outcomes (one row/fish)
    ├── LFP_timeseries (zero to three rows/fish)
    └── DLC_behavior (zero to three rows/fish)
```

`fish_id` is the primary key in the outcomes table and combines with `dpf` to
form the session-table keys.

## Feature isolation

Only these LFP columns enter the HMM:

- `lfp_mean_uv`
- `lfp_variance_uv2`
- `lfp_skewness`
- `lfp_kurtosis`
- `lfp_fourth_power_mean_uv4`
- `lfp_seizure_event_rate_per_h`
- `lfp_ica_complexity`

Columns containing `TRUTH`, injury metadata, dose, group, batch, QC results, or
behavior are excluded from model inputs.

The same seven columns, and only those, enter the HMM for the real recording.

## Real files (`real/`)

| File | Rows | Purpose |
|---|---:|---|
| `tbi_4_6dpf_real_lfp_timeseries.csv` | 706 | Session protocol, QC, and the seven HMM features |
| `tbi_4_6dpf_real_fish_outcomes.csv` | 240 | Arm, dose, survival, and the behavioral 6 dpf endpoint |
| `tbi_4_6dpf_real_behavior.csv` | 706 | Per-session aggregate of the blinded Event Log |
| `tbi_4_6dpf_real_manifest.json` | — | Provenance, endpoint definition, and arm counts |

### The real endpoint

`high_burden_state_dpf6_TRUTH` keeps its name so the two datasets share one
schema, but on real data it is **not planted truth**: a fish is positive if the
blinded scorer logged at least one qualifying event (Baraban stage ≥ 2 with
passing pose QC) in the 6 dpf session. It is derived purely from behavior and
shares no variable with the LFP feature matrix.

### Zero-event behavior sessions

The source Event Log contains only scored events, so a session in which the
scorer logged nothing is absent from it. Those sessions are written out with
zero event rates rather than dropped: "no scored behavior" is an observation,
and omitting them would restrict the behavioral validation to the abnormal
subset and bias it.

## Regeneration

```bash
tbi-generate --seed 42 --n-per-arm 60   # synthetic
tbi-real                                # real, from the source workbooks
```

`tbi-generate` rewrites the synthetic CSV tables and manifest
deterministically; the manifest hashes are checked by the test suite.
`tbi-real` rewrites `real/` from `actualdata1(lfp).xlsx` and
`actualdata(behavioral).xlsx` at the repository root.
