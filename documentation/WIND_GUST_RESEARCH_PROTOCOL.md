# Wind Gust Incremental-Value Research Protocol

**Status: preregistered retrospective research plan only. No production, live-board, weekly-pick, shadow, or prospective-ledger effect.**

## Research question

Does historical wind-gust information add useful out-of-sample predictive value for college-football market residuals **beyond sustained wind**, after controlling for the existing GENERAL HGB feature set and the possibility that an external weather source is simply better than the current CFBD weather field?

Primary target remains:

```text
market_residual = actual_total_points - closing_total
```

The primary question is incremental value, not whether windy games score differently in-sample.

## Current repository state

Historical game weather is currently pulled from CFBD `/games/weather`. The modeling dataset retains `windSpeed` as `wind_mph` and `windDirection` as `wind_direction_degrees`; no gust field is currently present in the historical feature pipeline.

The current research range is 2014-2025 regular seasons.

## Isolation boundary

This study must remain separate from the frozen/live 2026 system during retrospective development.

It must not:

- change GENERAL HGB production features;
- change live thresholds or weekly classifications;
- write `outputs/weekly_board.csv` or `outputs/weekly_picks.csv`;
- modify any official prospective ledger;
- modify the existing orientation or joint-core shadows;
- call `run_live_week`;
- auto-promote any challenger.

Development should occur on `research/wind-gust-value` until the retrospective study is complete and reviewed.

## Candidate historical gust source

CFBD's current `GameWeather` schema exposes sustained wind speed and direction but not a gust field.

For the first retrospective feasibility study, use a single internally consistent external source for **both sustained wind and gust** so source quality is not confused with gust value.

Preferred initial candidate:

- Open-Meteo Historical Weather API;
- explicitly pin the reanalysis model to ERA5 rather than using a changing "Best Match" blend;
- request hourly `wind_speed_10m`, `wind_direction_10m`, and `wind_gusts_10m` in mph;
- use committed stadium coordinates to query venue locations;
- retain source/model metadata with every row.

Reference documentation:

- https://open-meteo.com/en/docs/historical-weather-api
- https://apinext.collegefootballdata.com/api/games

Direct ECMWF ERA5 access can be considered later if API reproducibility or source-definition questions require it.

## Critical timing / leakage boundary

ERA5 is reanalysis, not the forecast that was actually available to a bettor before kickoff. Therefore a positive ERA5 result can establish that gust information contains physical/statistical signal, but **cannot by itself establish deployable pregame value**.

Two timing tracks may be evaluated, but they must never be conflated:

### Primary retrospective track: kickoff snapshot

For each game, use the latest hourly ERA5 value whose timestamp is **at or before scheduled kickoff**. Do not use a later game-hour value for the primary test.

This is still reanalysis and therefore not production-deployable evidence, but it avoids the more severe error of using post-kickoff weather from the game window.

### Secondary oracle diagnostic: game-window gust

A separately labeled diagnostic may examine maximum/mean gust during an approximate game window. This is hindsight-only and may answer whether gustiness physically matters during play.

**Oracle game-window results can never satisfy a promotion gate.**

If the kickoff-snapshot retrospective test is supported, the next stage must use archived forecasts at fixed pregame lead times and then a frozen prospective shadow.

## Data construction and QC

Create a separate external-weather research table rather than modifying the historical CFBD raw file in place.

Minimum fields:

```text
game_id
season
start_date
venue_id
latitude
longitude
era5_valid_time_utc
era5_wind_mph
era5_wind_direction_degrees
era5_gust_mph
gust_spread_mph
weather_source
weather_model
```

Define:

```text
gust_spread_mph = era5_gust_mph - era5_wind_mph
```

Do not silently clamp negative spreads. Report them as QC anomalies first and decide on handling before model fitting.

Recommended QC outputs:

- overall and by-season coverage;
- venue coverage;
- missingness by field;
- timestamp offset from kickoff;
- sustained-wind and gust distributions;
- gust-spread distribution;
- count/rate of negative gust spreads;
- correlation of ERA5 sustained wind with current CFBD `wind_mph`;
- mean/median absolute difference between ERA5 and CFBD sustained wind;
- extreme-value inspection for gust and spread.

All challenger models must be scored on the exact same common paired sample.

## Predeclared model ladder

Use the existing HistGradientBoosting regression framework and current GENERAL baseline features.

1. `baseline`
   - Existing GENERAL HGB features only.

2. `era5_sustained_control`
   - Baseline + ERA5 10 m sustained wind.
   - Purpose: measure whether the external source itself adds value before crediting gusts.

3. `gust_magnitude`
   - `era5_sustained_control` + ERA5 10 m gust magnitude.

4. `gust_spread`
   - `era5_sustained_control` + `gust_spread_mph`.

5. `gust_core`
   - `era5_sustained_control` + gust magnitude + gust spread.

6. `gust_interactions`
   - `gust_core` + a deliberately small set of preregistered physically/contextually motivated interactions.

### Primary pairwise comparisons

The key comparisons are:

```text
era5_sustained_control vs baseline
gust_magnitude vs era5_sustained_control
gust_spread vs era5_sustained_control
gust_core vs era5_sustained_control
gust_interactions vs gust_core
```

The main gust claim must be based on improvement over `era5_sustained_control`, not merely over `baseline`.

## Continuous features first

Primary evidence should come from continuous features. Threshold rules are diagnostics, not the main model-selection mechanism.

Primary continuous candidates:

- `era5_gust_mph`;
- `gust_spread_mph`.

Optional diagnostic only:

```text
gust_factor = era5_gust_mph / era5_wind_mph
```

Only calculate gust factor when sustained wind is at least 5 mph because ratios become unstable in near-calm conditions.

## Predeclared threshold diagnostics

To avoid threshold fishing, begin with a small fixed grid.

Gust magnitude:

- >= 20 mph
- >= 25 mph
- >= 30 mph
- >= 35 mph

Gust spread:

- >= 5 mph
- >= 10 mph
- >= 15 mph

For each threshold, report sample size, market residual, over/under rate, and model error conditional on the regime. These are descriptive/sensitivity diagnostics unless a threshold is separately validated out of sample.

Do not promote a threshold because it happens to maximize ROI in the full historical sample.

## Limited interaction candidates

HistGradientBoosting already learns nonlinear structure, so explicit interactions should be limited.

Initial interaction candidates:

- gust spread x sustained wind;
- gust spread x market-total context;
- gust magnitude x temperature-anomaly context.

A separate secondary orientation-ready analysis may test:

- gust spread x sustained crosswind magnitude;
- gust spread x sustained along-field wind magnitude.

Do not require field orientation for the primary gust study because doing so would unnecessarily shrink the common sample.

## Chronological evaluation

Mirror the repository's existing leak-safe weather-context research pattern:

- train each test season using only earlier seasons;
- fit any imputation/scaling/context references on training data only;
- score every ladder variant on the same test games;
- keep the target as market residual;
- make prediction error the primary evidence, with betting outcomes secondary.

## Evidence gate for "gust adds value beyond sustained wind"

A gust challenger is `SUPPORTED_RETROSPECTIVELY` against `era5_sustained_control` only if all of the following hold:

1. mean paired MAE delta is below zero;
2. the 95% game-level paired bootstrap interval is entirely below zero;
3. the 95% season-cluster interval is entirely below zero;
4. the challenger improves MAE in at least 70% of evaluated test seasons.

Secondary diagnostics may include RMSE, signed bias, qualifier hit rate, ROI, and regime stability, but they cannot override the MAE evidence gate.

## Robustness checks if and only if the primary gate is promising

Do not expand the search space before the primary ladder is evaluated.

If a gust challenger is supported or narrowly misses the gate, then examine:

- performance by test season;
- early vs recent eras;
- high/low market-total regimes;
- sustained-wind regimes;
- gust/spread regimes;
- latitude/climate regimes;
- venue concentration;
- sensitivity to removing one test season at a time;
- sensitivity to kickoff-hour alignment choices.

## Stop / promotion rules

### Stop

If gust magnitude, gust spread, and gust core do not show robust incremental improvement over the same-source sustained-wind control, stop. Do not add gusts to the live model merely because a threshold subgroup or ROI slice looks attractive.

### Physics-only finding

If only the oracle game-window gust analysis shows value, document the physical relationship but do not treat it as deployable predictive evidence.

### Advance

If kickoff-snapshot gust features clear the retrospective gate, next test **archived pregame forecasts** at fixed lead times. Open-Meteo's archived forecast products can support a shorter recent-era study, but coverage is much shorter than the 2014-2025 reanalysis period.

Only after a forecast-based challenger is defensible should a separately versioned, frozen prospective shadow be considered. Production remains unchanged unless that future prospective evidence succeeds and an explicit promotion decision is made.

## Proposed repository outputs when implementation begins

Keep all outputs isolated under:

```text
outputs/wind_gust/
```

Suggested outputs:

```text
coverage.csv
source_comparison.csv
qc_summary.csv
threshold_diagnostics.csv
walk_forward_predictions.csv
walk_forward_diagnostics.csv
model_summary.csv
incremental_comparisons.csv
model_by_season.csv
regime_stability.csv
summary.md
```

Suggested implementation modules, only after this protocol is reviewed:

```text
src/pull_historical_gusts.py
src/wind_gust_research.py
src/wind_gust_research_selftest.py
.github/workflows/manual-wind-gust-research.yml
```

The future manual workflow should restore the latest historical modeling artifact exactly as the existing isolated research workflows do, run deterministic self-tests, produce an artifact containing the external weather cache and research outputs, and commit only `outputs/wind_gust/` if output commits are desired.
