# Joint Weather-Context Research

**Status: retrospective research only. No production or 2026 prospective-ledger effect.**

## Purpose

This study tests the combined hypothesis that college-football weather effects are conditional rather than additive: the value of wind may depend on field orientation, local climate, latitude, temperature regime, and the market total at the same time.

The prediction target remains:

```text
market_residual = actual_total_points - closing_total
```

The question is therefore not merely whether bad weather lowers scoring. It is whether a joint weather-context representation explains information that the closing market total did not already price.

## Isolation from 2026 live operations

This research path is intentionally separate from the frozen 2026 system.

It does not:

- change GENERAL HGB production features;
- change live thresholds or weekly classifications;
- write `outputs/weekly_board.csv` or `outputs/weekly_picks.csv`;
- modify the official prospective ledger;
- modify the existing frozen orientation challenger;
- call `run_live_week`;
- promote a model automatically.

The manual workflow restores the existing historical modeling artifact, reads the committed stadium-location and stadium-orientation references, runs only the joint research module, and commits only `outputs/joint_weather_context/`.

The workflow shares the repository's data-write concurrency group solely to avoid simultaneous Git writes with live jobs. That serialization does not connect the research model to live scoring.

## Paired model ladder

Every model is evaluated on the exact same outdoor FBS-vs-FBS games that have the inputs required for both climate context and field orientation.

1. `baseline`
   - Existing GENERAL HGB feature set.

2. `orientation_crosswind`
   - Baseline plus field-relative crosswind magnitude.

3. `orientation_vector`
   - Baseline plus crosswind and along-field wind magnitudes.

4. `climate_full`
   - Baseline plus venue latitude, training-only local temperature anomaly, training-only local wind percentile, and the existing latitude-context interactions.

5. `joint_core`
   - Orientation vector and climate context together.

6. `joint_interactions`
   - Joint core plus a deliberately small set of physically motivated interaction terms:
     - crosswind × temperature anomaly;
     - along-field wind × temperature anomaly;
     - crosswind × local wind percentile;
     - along-field wind × local wind percentile;
     - crosswind × latitude;
     - along-field wind × latitude;
     - crosswind × market-total context.

HistGradientBoosting already learns nonlinear splits and higher-order structure. Explicit interactions are therefore limited rather than exhaustively enumerated.

## Field-relative wind

Field orientation is treated as an undirected 0-180 degree axis. Historical meteorological wind direction is projected into:

```text
crosswind_mph = wind_mph * sin(field-relative angle)
alongwind_mph = wind_mph * cos(field-relative angle)
```

Both are magnitudes because reversing the direction along the same football-field axis does not create a different orientation class for this first joint study.

## Leak-safe local climate context

For each walk-forward fold, temperature and wind context is learned from training seasons only.

Temperature uses the same hierarchy as the climate-context study:

1. venue/month historical game-weather mean;
2. latitude-band/month fallback;
3. calendar-month fallback;
4. training-sample global fallback.

Wind uses the analogous empirical distribution hierarchy and converts the observed game wind to a local percentile.

No tested season contributes to its own local weather reference.

## Chronological evaluation

For each test season:

- all model training uses only earlier seasons;
- local climate references use only earlier seasons;
- all challenger variants are scored on the same joint-ready test games;
- the target is market residual, not raw points.

This preserves paired comparisons and prevents a challenger from improving by selecting an easier subset.

## Evidence gate

A challenger is labeled `SUPPORTED_RETROSPECTIVELY` against a reference only if all four conditions are satisfied:

1. mean paired MAE delta is below zero;
2. the 95% game-level paired bootstrap interval lies entirely below zero;
3. the 95% season-cluster interval lies entirely below zero;
4. the challenger improves MAE in at least 70% of evaluated test seasons.

The season-cluster interval treats each test season as a higher-level unit so a large season cannot by itself create a claim of statistical support.

UNDER qualifier hit rate and ROI remain secondary diagnostics. They cannot override the prediction-error gate.

## Stability diagnostics

The full joint challenger is also summarized across:

- latitude bands;
- crosswind regimes;
- temperature-anomaly regimes;
- local-wind-percentile regimes;
- closing-total regimes;
- individual test seasons.

These are diagnostic checks, not separate opportunities to search for a winning rule.

## Promotion boundary

A retrospective `SUPPORTED_RETROSPECTIVELY` result is not a production decision.

If the joint study clears the evidence gate and the improvement is reasonably stable, the next step is a separately versioned, frozen prospective shadow challenger. Only predictions made after that future freeze could count as prospective evidence.

The frozen 2026 GENERAL model, current live outputs, official ledger, and existing orientation shadow remain unchanged unless a later, explicit decision is made after statistically supported retrospective evidence and prospective validation.

## Generated outputs

`python -m src.joint_weather_context_research` writes only:

- `outputs/joint_weather_context/coverage.csv`;
- `outputs/joint_weather_context/walk_forward_predictions.csv`;
- `outputs/joint_weather_context/walk_forward_diagnostics.csv`;
- `outputs/joint_weather_context/model_summary.csv`;
- `outputs/joint_weather_context/incremental_comparisons.csv`;
- `outputs/joint_weather_context/model_by_season.csv`;
- `outputs/joint_weather_context/regime_stability.csv`;
- `outputs/joint_weather_context/summary.md`.

`python -m src.joint_weather_context_selftest` supplies a deterministic synthetic validation path for the field-relative geometry, common paired sample, model ladder, uncertainty summaries, and regime diagnostics.
