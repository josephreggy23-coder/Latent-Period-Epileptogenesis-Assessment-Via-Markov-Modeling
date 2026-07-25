# Data

Every row in this directory represents a real larval zebrafish.

`measured/` holds the normalized analysis tables, generated from the source
workbooks by `tbi-analyze`. The workbooks at the repository root are the
authoritative inputs; everything here is derived and rebuilt on each run.

## Files

| File | Rows | Purpose |
|---|---:|---|
| `tbi_4_6dpf_lfp_timeseries.csv` | 706 | Session protocol, QC, and the seven HMM features |
| `tbi_4_6dpf_fish_outcomes.csv` | 240 | Arm, dose, survival, and the 6 dpf endpoint |
| `tbi_4_6dpf_behavior.csv` | 706 | Per-session aggregate of the blinded Event Log |
| `tbi_4_6dpf_manifest.json` | — | Cohort scope, arm counts, and endpoint definition |

## Table relationships

```text
fish_outcomes (one row/fish)
    ├── lfp_timeseries (one to three rows/fish)
    └── behavior       (one row per LFP session)
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

Injury metadata, dose, group, batch, QC results, behavior, and the endpoint are
excluded from model inputs.

## The endpoint

`high_burden_state_dpf6` is behavioral and **three-valued**:

| Value | Meaning |
|---|---|
| `1` | ≥ 1 qualifying blinded event (Baraban stage ≥ 2, passing pose QC) at 6 dpf |
| `0` | observed at 6 dpf, no qualifying event |
| `NA` | no evidence the fish was observed at 6 dpf |

Seven fish are `NA`. An unobserved animal has an unknown outcome, not a negative
one; coding absence as `0` would pad the negative class with animals nobody
checked. Those fish are excluded from endpoint scoring.

Evidence of observation is a 6 dpf LFP session **or** any 6 dpf behavioral row,
including a normal one — the Event Log records normal swim bouts, so presence
proves observation while absence alone proves nothing.

The endpoint is derived purely from behavior and shares no variable with the LFP
feature matrix.

## Zero-event behavior sessions

The source Event Log contains only scored events, so a session in which the
scorer logged nothing is absent from it. Those sessions are written out with
zero event rates rather than dropped: "no scored behavior" is an observation,
and omitting them would restrict the behavioral validation to the abnormal
subset and bias it.

## Regeneration

```bash
tbi-analyze
```

Rewrites `measured/` from `actualdata1(lfp).xlsx` and
`actualdata(behavioral).xlsx` at the repository root, then runs the analysis.
