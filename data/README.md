# Data

All committed data are synthetic. No row represents an experimental animal.

## Authoritative files

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

## Regeneration

```bash
tbi-generate --seed 42 --n-per-arm 60
```

The command rewrites the CSV tables and manifest deterministically. The
manifest hashes are checked by the test suite.
