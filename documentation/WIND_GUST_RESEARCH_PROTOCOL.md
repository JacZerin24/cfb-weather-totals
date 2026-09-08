# Wind Gust Incremental-Value Research Protocol

**Status: preregistered retrospective research only. No production, live-board, weekly-pick, shadow, or prospective-ledger effect.**

## Research question

Does historical wind-gust information add useful out-of-sample predictive value for college-football market residuals **beyond sustained wind**, after controlling for the existing GENERAL HGB feature set and for differences between weather data sources?

Primary target remains:

```text
market_residual = actual_total_points - closing_total
```

The primary question is incremental value, not whether gusty games score differently in-sample.

## Current repository state

Historical game weather is currently pulled from CFBD `/games/weather`. The modeling dataset retains `windSpeed` as `wind_mph` and `windDirection` as `wind_direction_degrees`; no gust field is currently present in that historical feature path.

The configured retrospective range is 2014-2025 regular seasons.

## Isolation boundary

This study remains separate from the frozen/live 2026 system.

It must not:

- change GENERAL HGB production features;
- change live thresholds or weekly classifications;
- write `outputs/weekly_board.csv` or `outputs/weekly_picks.csv`;
- modify any official prospective ledger;
- modify the existing orientation or joint-core shadows;
- call `run_live_week`;
- auto-promote any challenger.

Development remains on `research/wind-gust-value` until retrospective work is complete and reviewed.

## Historical source definition

Open-Meteo's current Historical Weather API documentation does **not** expose the three needed variables from one long-history ERA5 product:

- ERA5 provides 10 m sustained wind speed and direction but its current Open-Meteo availability table does not list wind gusts.
- ERA5-Land provides 10 m gusts but does not independently provide the same sustained-wind/direction fields.
- ECMWF IFS provides sustained wind, direction, and gust from one model, but the Open-Meteo historical IFS record begins in 2017 rather than covering the full 2014-2025 period.

Therefore the acquisition phase uses two explicitly labeled source stacks instead of pretending a single-source long-history series exists.

### Primary long-history stack

```text
source_stack = era5_era5_land
wind_model   = era5
gust_model   = era5_land
```

Request:

- ERA5 `wind_speed_10m`;
- ERA5 `wind_direction_10m`;
- ERA5-Land `wind_gusts_10m`;
- mph wind units;
- GMT timestamps;
- committed stadium coordinates.

This stack preserves 2014-2025 coverage but is **not a same-model wind/gust comparison**. That fact must remain attached to every row and considered when interpreting results.

### Same-model sensitivity stack

```text
source_stack = ecmwf_ifs
wind_model   = ecmwf_ifs
gust_model   = ecmwf_ifs
```

For 2017 onward, optionally acquire:

- `wind_speed_10m`;
- `wind_direction_10m`;
- `wind_gusts_10m`.

This shorter sample is useful as a same-model sensitivity check if the primary long-history result is promising. It does not replace the long-history analysis and cannot erase time-period/model-version limitations.

Reference documentation:

- https://open-meteo.com/en/docs/historical-weather-api
- https://github.com/open-meteo/open-meteo/blob/main/openapi/historical-weather.yml
- https://apinext.collegefootballdata.com/api/games

## Critical timing / leakage boundary

The Open-Meteo historical products used here are reanalysis/historical analyses, not the forecast actually available to a bettor before kickoff.

A positive result can establish that gust information contains physical/statistical signal, but **cannot by itself establish deployable pregame value**.

### Primary retrospective track: kickoff snapshot

For each game, use the latest hourly value whose timestamp is **at or before scheduled kickoff**.

The acquisition code:

- converts scheduled kickoff to UTC;
- requests GMT hourly data;
- refuses post-kickoff hours;
- records the age of the selected wind and gust observations;
- rejects a selected value if it is more than two hours old by default.

No later game-hour value may be substituted into the primary kickoff snapshot.

### Secondary oracle diagnostic

A future separately labeled diagnostic may examine maximum/mean gust during an approximate game window.

That would be hindsight-only and can never satisfy a promotion gate.

If kickoff-snapshot retrospective evidence is supported, the next stage must use archived forecasts at fixed pregame lead times and then a frozen prospective shadow.

## Data acquisition artifact

The acquisition module is:

```text
src/pull_historical_gusts.py
```

It reads:

```text
data/processed/modeling_dataset.csv
data/reference/stadium_locations.csv
```

and writes an isolated research table:

```text
data/processed/wind_gust_kickoff.csv
```

with a resumable cache:

```text
data/processed/wind_gust_kickoff_cache.csv
```

Neither file changes the existing modeling dataset.

Minimum retained fields include:

```text
game_id
season
start_date
kickoff_utc
venue_id
venue_name
latitude
longitude
cfbd_wind_mph
game_indoors_bool
source_stack
weather_provider
wind_model
gust_model
same_model_components
wind_valid_time_utc
gust_valid_time_utc
wind_age_minutes
gust_age_minutes
component_time_delta_minutes
wind_grid_latitude
wind_grid_longitude
wind_grid_elevation_m
gust_grid_latitude
gust_grid_longitude
gust_grid_elevation_m
component_grid_distance_km
wind_mph
wind_direction_degrees
gust_mph
gust_spread_mph
gust_factor
fetch_status
fetch_detail
```

Define:

```text
gust_spread_mph = gust_mph - wind_mph
```

and, only when sustained wind is at least 5 mph:

```text
gust_factor = gust_mph / wind_mph
```

Negative gust spread is **not clamped or deleted** during acquisition. It is retained and reported as a QC anomaly.

## Request strategy

To reduce external API load, games are grouped by:

```text
source_stack
venue_id
season
```

For each venue-season group, the code requests one continuous hourly date range covering that group's games plus a one-day buffer on each side, rather than making one API request per game.

The client uses retries/backoff for 429 and transient 5xx responses and supports a small configurable request delay.

The default acquisition stack is `era5_era5_land`.

Optional same-model IFS acquisition can be added with:

```text
python -m src.pull_historical_gusts --include-ifs-sensitivity
```

A smoke test can cap the number of venue-season groups:

```text
python -m src.pull_historical_gusts --max-groups 5
```

## QC outputs

The acquisition phase writes **only data/QC outputs**, not model results:

```text
outputs/wind_gust/data_coverage.csv
outputs/wind_gust/source_comparison.csv
outputs/wind_gust/qc_summary.csv
outputs/wind_gust/distribution_summary.csv
outputs/wind_gust/extreme_values.csv
outputs/wind_gust/request_log.csv
```

QC includes:

- overall/by-season source coverage;
- missing venue-coordinate rate;
- complete sustained/gust coverage;
- source out-of-range rows;
- partial/missing API rows;
- kickoff-to-selected-hour age;
- wind/gust component timestamp mismatches;
- source grid-point separation;
- sustained-wind and gust distributions;
- gust-spread distribution;
- negative gust-spread count/rate;
- impossible negative wind/gust values;
- extreme gust/spread checks;
- correlation of external sustained wind with CFBD `wind_mph`;
- external-minus-CFBD sustained-wind bias;
- mean/median absolute external-vs-CFBD wind difference.

Large external-vs-CFBD differences are surfaced for manual inspection rather than automatically "corrected."

## Deterministic self-test

The acquisition/QC code has a network-free self-test:

```text
python -m src.wind_gust_data_selftest
```

It verifies:

- latest-at-or-before-kickoff alignment;
- no post-kickoff leakage;
- stale-hour rejection;
- preservation of negative gust spread;
- QC summary accounting;
- explicit mixed-source vs same-model metadata.

The self-test is a code-integrity check, not evidence that Open-Meteo historical coverage is adequate. Coverage must be demonstrated by the actual acquisition outputs.

## Model ladder after QC passes

No model is fit during the current acquisition phase.

If data QC is acceptable, the predeclared next ladder remains:

1. `baseline`
   - Existing GENERAL HGB features.

2. `era5_sustained_control`
   - Baseline + external ERA5 10 m sustained wind.
   - Measures source substitution before crediting gusts.

3. `gust_magnitude`
   - Sustained control + 10 m gust magnitude.

4. `gust_spread`
   - Sustained control + `gust_spread_mph`.

5. `gust_core`
   - Sustained control + gust magnitude + gust spread.

6. `gust_interactions`
   - Gust core + a deliberately small preregistered interaction set.

Primary gust comparisons remain:

```text
gust_magnitude vs era5_sustained_control
gust_spread    vs era5_sustained_control
gust_core      vs era5_sustained_control
```

Because the long-history stack mixes ERA5 sustained wind with ERA5-Land gust, a promising result should also be checked on the same-model IFS sensitivity sample before making a strong causal/source-independent claim.

## Continuous features first

Primary evidence comes from continuous features:

- `gust_mph`;
- `gust_spread_mph`.

`gust_factor` is diagnostic only because ratios become unstable in near-calm conditions.

## Predeclared threshold diagnostics

To avoid threshold fishing, initial descriptive thresholds remain fixed.

Gust magnitude:

- >= 20 mph
- >= 25 mph
- >= 30 mph
- >= 35 mph

Gust spread:

- >= 5 mph
- >= 10 mph
- >= 15 mph

Threshold tables are descriptive/sensitivity diagnostics unless separately validated out of sample.

## Limited interaction candidates

Only after the primary ladder is run:

- gust spread x sustained wind;
- gust spread x market-total context;
- gust magnitude x temperature-anomaly context.

A separate orientation-ready analysis may later test gust spread with sustained crosswind/along-field magnitude.

Field orientation is not required for the primary gust study because that would unnecessarily shrink the common sample.

## Chronological evaluation

Any later model phase must mirror the repository's existing leak-safe research pattern:

- train each test season using only earlier seasons;
- fit imputation/scaling/context references on training data only;
- score every ladder variant on the same paired test games;
- keep the target as market residual;
- make prediction error the primary evidence;
- treat betting outcomes as secondary diagnostics.

## Evidence gate

A gust challenger is `SUPPORTED_RETROSPECTIVELY` against the sustained-wind control only if all of the following hold:

1. mean paired MAE delta is below zero;
2. the 95% game-level paired bootstrap interval is entirely below zero;
3. the 95% season-cluster interval is entirely below zero;
4. the challenger improves MAE in at least 70% of evaluated test seasons.

RMSE, signed bias, qualifier hit rate, ROI, and regime stability cannot override this gate.

## Stop / advance rules

### Stop

If gust magnitude, gust spread, and gust core do not show robust incremental improvement over sustained wind, stop. Do not add gusts to the live model because one threshold subgroup or ROI slice looks attractive.

### Physics-only finding

If only a hindsight game-window gust analysis shows value, document the physical relationship but do not treat it as deployable evidence.

### Advance

If kickoff-snapshot gust features clear the retrospective gate and source sensitivity is acceptable, next test **archived pregame forecasts** at fixed lead times.

Only after forecast-based evidence is defensible should a separately versioned frozen prospective shadow be considered.

Production remains unchanged unless that future prospective evidence succeeds and an explicit promotion decision is made.
