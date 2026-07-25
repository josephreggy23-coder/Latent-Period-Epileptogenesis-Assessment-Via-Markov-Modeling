# Data

All committed rows are deterministically generated synthetic placeholders. No
current row represents an experimental animal. The current benchmark also
expects reference state labels and a binary 6 dpf endpoint; it is not a
drop-in analysis for unlabeled measured data.

## Authoritative files

The editable analysis tables are stored in `data/template/`:

| File | Rows | Purpose |
|---|---:|---|
| `tbi_4_6dpf_lfp_timeseries.csv` | 706 | Session protocol, QC, seven HMM features, planted state |
| `tbi_4_6dpf_fish_outcomes.csv` | 240 | Injury arm, attrition, and planted DPF6 endpoint |
| `tbi_4_6dpf_dlc_behavior.csv` | 706 | Generated pose-style concordance features |
| `tbi_4_6dpf_dataset_manifest.json` | — | Initialized seed, scope, sources, and SHA-256 hashes |
| `TBI_4_6dpf_data_template.xlsx` | — | Formatted reference snapshot for human review |

The CSV tables are the analysis inputs. The manifest and workbook describe the
initialized seed-42 template; manual CSV edits invalidate the manifest hashes.

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

The repository consumes precomputed session summaries. It does not extract
seizure events, ICA complexity, higher moments, or pose summaries from raw LFP
or video. A measured-data adapter must define and validate those operations
before marking records analysis-ready.

`electrode_resistance_change_pct` is the generated absolute resistance-change
QC proxy adapted from Eimon et al.; it is not a physical displacement measure.

## Regeneration

```bash
tbi-initialize --seed 42 --n-per-arm 60
```

The command refuses to overwrite an existing template. To make an intentional
reset after backing up manual entries, add `--force`. The manifest hashes are
checked by the test suite.

Normal analysis refuses to run while any row remains
`placeholder_pending_replacement`. After replacing every value in a completed
row, set that row to `analysis_ready`.
