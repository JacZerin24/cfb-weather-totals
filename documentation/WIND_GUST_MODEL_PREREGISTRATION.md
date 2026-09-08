# Wind Gust Model Preregistration

**Branch:** `research/wind-gust-value`

**Status:** frozen before the first full-sample gust model run. Retrospective research only. No production, live-board, weekly-pick, shadow, or prospective-ledger effect.

## Purpose

This addendum freezes the final model comparison after source feasibility established ECMWF IFS as the viable same-model historical wind+gust source. It replaces the earlier placeholder `era5_sustained_control` name for the actual first model run; it does not change the scientific question or evidence gate.

## Population

The first gust-value model run is restricted to the exact common population:

- seasons 2017-2025;
- outdoor games;
- FBS vs FBS;
- closing total present;
- final score present;
- valid scheduled kickoff;
- complete same-model ECMWF IFS kickoff-aligned sustained wind and gust row;
- `fetch_status = ok`;
- `source_stack = ecmwf_ifs`.

All ladder variants use identical training and test games. The shorter 2017-2025 period applies to the baseline as well so the gust variants do not receive a different training sample.

## Target and evaluation

Primary target remains:

```text
market_residual = actual_total_points - closing_total
```

Evaluation remains chronological walk-forward:

- each test season is predicted using only earlier seasons;
- minimum training games: 1,000;
- minimum test games: 100;
- preprocessing and temperature-context references are fit from the training fold only;
- prediction error is primary evidence;
- betting outcomes cannot override the prediction-error gate.

The estimator is the repository's existing GENERAL HistGradientBoostingRegressor configuration. This research does not tune a new estimator.

## Frozen model ladder

### 1. `baseline`

Existing GENERAL HGB feature set only, including the existing CFBD sustained-wind feature.

### 2. `ifs_sustained_control`

Baseline plus:

```text
ifs_wind_mph
```

This is the mandatory external-source control. Gusts do not receive credit merely because IFS sustained wind differs from CFBD sustained wind.

### 3. `gust_magnitude`

`ifs_sustained_control` plus:

```text
ifs_gust_mph
```

### 4. `gust_spread`

`ifs_sustained_control` plus:

```text
ifs_gust_spread_mph
```

The spread remains the returned IFS gust minus IFS instantaneous sustained wind. Negative values are retained.

### 5. `gust_core`

`ifs_sustained_control` plus both:

```text
ifs_gust_mph
ifs_gust_spread_mph
```

### 6. `gust_interactions`

`gust_core` plus only these preregistered interactions:

```text
ifs_gust_spread_x_wind
ifs_gust_spread_x_market_total
ifs_gust_x_temperature_anomaly
```

The market-total interaction uses `(closing_total - 56) / 10` as the context scale. The temperature interaction uses a temperature anomaly built from prior-season training data only and scaled by 10 F.

No additional gust interactions are permitted in the first run.

## Frozen pairwise comparisons

The first run reports exactly these comparisons:

1. `ifs_sustained_control` vs `baseline`
2. `gust_magnitude` vs `ifs_sustained_control`
3. `gust_spread` vs `ifs_sustained_control`
4. `gust_core` vs `ifs_sustained_control`
5. `gust_interactions` vs `gust_core`

The primary question "does gust add value beyond sustained wind?" is answered by comparisons 2-4.

## Evidence gate

For a primary gust challenger to be labeled `SUPPORTED_RETROSPECTIVELY`, every condition must hold against `ifs_sustained_control`:

1. mean paired MAE delta < 0;
2. 95% game-level paired bootstrap CI upper bound < 0;
3. 95% season-cluster bootstrap CI upper bound < 0;
4. MAE improves in at least 70% of evaluated test seasons.

Otherwise the result is `NOT_PROVEN`.

The source-control comparison and interaction-vs-core comparison use the same statistical gate for transparency, but they are not substitutes for the three primary gust-vs-sustained comparisons.

## Fixed descriptive thresholds

Threshold diagnostics remain descriptive only and are fixed before results.

IFS gust magnitude:

- >= 20 mph
- >= 25 mph
- >= 30 mph
- >= 35 mph

IFS gust spread:

- >= 5 mph
- >= 10 mph
- >= 15 mph

For each threshold, report sample size, seasons represented, average market residual, under rate excluding pushes, baseline MAE, IFS sustained-control MAE, gust-core MAE, and gust-core MAE delta vs sustained control.

No threshold may be promoted because it has the best historical ROI or under rate.

## Outputs for the first model run

The first model run writes only:

```text
outputs/wind_gust_value/coverage.csv
outputs/wind_gust_value/walk_forward_predictions.csv
outputs/wind_gust_value/walk_forward_diagnostics.csv
outputs/wind_gust_value/model_summary.csv
outputs/wind_gust_value/incremental_comparisons.csv
outputs/wind_gust_value/model_by_season.csv
outputs/wind_gust_value/threshold_diagnostics.csv
outputs/wind_gust_value/summary.md
```

No robustness/regime expansion is run automatically in the first pass.

## Stop / advance rule

If `gust_magnitude`, `gust_spread`, and `gust_core` all fail the gate against `ifs_sustained_control`, stop the primary gust study. A favorable threshold slice, ROI slice, or one-season result cannot rescue it.

If at least one primary gust challenger clears the gate, the next scientific stage is fixed-lead archived pregame forecast testing. Reanalysis/historical IFS evidence alone cannot justify production use or a prospective betting shadow.

Production remains unchanged until a future explicit promotion decision after forecast-based and prospective evidence.