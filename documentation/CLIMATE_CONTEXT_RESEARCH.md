# Climate-Context and Latitude Research

**Status: retrospective research only. No production or prospective-ledger effect.**

## Purpose

This study tests whether the relationship between weather and college-football totals changes by venue latitude and whether locally unusual weather adds information beyond raw temperature and wind speed.

The target remains:

```text
market_residual = actual_total_points - closing_total
```

That distinction matters. The study is not asking only whether cold or windy games score fewer points. It is asking whether location-conditioned weather helps explain what the closing market total did not already price.

## Retrospective scope

- Historical period: the existing 2014-2025 modeling dataset.
- Primary track: outdoor FBS-vs-FBS games.
- Paired evaluation requires venue coordinates, kickoff month, temperature, wind, closing total, and final score.
- Venue coordinates come from the CollegeFootballData `/venues` endpoint and are saved to `data/reference/stadium_locations.csv` by the historical research run.
- Coordinate coverage is reported before any result is interpreted.

The descriptive tables use four predeclared latitude bands:

- south of 30 degrees North;
- 30 to less than 35 degrees North;
- 35 to less than 40 degrees North;
- 40 degrees North and higher.

Temperature and wind retain the existing project bins so latitude comparisons remain compatible with the earlier weather research.

## Local weather context

Raw latitude is only a geographic proxy. The more meteorologically useful question is whether the forecast weather is unusual for that location and time of season.

For each walk-forward training fold, the script learns:

- a venue-month historical game-temperature baseline;
- a venue-month empirical wind distribution.

When a venue-month sample is too small, the calculation falls back to latitude-band/month and then calendar-month information. The resulting research features are:

- `venue_latitude`;
- `temperature_anomaly_f`, relative to the training-only venue/month hierarchy;
- `wind_local_percentile`, based on training-only outdoor game winds;
- explicit temperature-anomaly-by-latitude and wind-percentile-by-latitude interaction terms.

These are historical football-game weather references, not official NOAA climate normals. The terminology in code and reports intentionally preserves that distinction.

## Model comparison

The study compares four HGB residual models on the same eligible games:

1. `baseline`: the existing GENERAL HGB feature set;
2. `latitude_only`: baseline plus venue latitude;
3. `local_weather_context`: baseline plus temperature anomaly and local wind percentile;
4. `full_latitude_context`: all context features plus the explicit latitude interactions.

Each test season is predicted using only earlier seasons. Weather-context references are also fitted only on those earlier seasons. This prevents a future season's weather distribution from leaking into its own test features.

Primary evidence:

- paired projected-total MAE;
- paired MAE delta versus the baseline;
- 95 percent paired bootstrap interval;
- RMSE and signed projection bias;
- number of test seasons with lower MAE than the baseline.

UNDER qualifier hit rate and ROI are reported only as secondary context using the existing 3.5-point edge and 56-point minimum-total thresholds.

## Generated outputs

The dedicated `Manual Climate Context Research` workflow restores the latest successful historical modeling-dataset artifact, runs `python -m src.climate_context_research`, and writes:

- `outputs/climate_context/coordinate_coverage.csv`;
- `outputs/climate_context/latitude_weather_interactions.csv`;
- `outputs/climate_context/walk_forward_predictions.csv`;
- `outputs/climate_context/walk_forward_diagnostics.csv`;
- `outputs/climate_context/model_summary.csv`;
- `outputs/climate_context/model_by_season.csv`;
- `outputs/climate_context/summary.md`.

`src.climate_context_selftest` supplies a deterministic synthetic end-to-end check and runs automatically for relevant pull requests.

The dedicated workflow commits only `data/reference/stadium_locations.csv` and `outputs/climate_context/`. It does not run the live board, rebuild the website, change weekly picks, regenerate the main research summaries, or touch the prospective ledger.

## Prospective decision boundary

This retrospective script does not choose, freeze, or deploy a 2026 challenger. Results should first be reviewed for:

- coordinate and context coverage;
- negative paired MAE delta with uncertainty considered;
- improvement across multiple test seasons rather than one isolated year;
- stability across latitude bands and weather regimes;
- no evidence that a small qualifier subset is driving the conclusion.

If a specification is selected, it must receive a new version, freeze timestamp, research-only shadow output path, and prospective evaluation protocol. Only games after that freeze may count as prospective evidence. As with the orientation challenger, it must not modify the live board status, weekly picks, or official prospective ledger.
