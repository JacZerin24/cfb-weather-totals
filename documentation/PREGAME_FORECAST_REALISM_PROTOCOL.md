# Archived Pregame Forecast Realism Research Protocol

## Purpose

This research asks whether the weather information that would actually have been available before kickoff translates the existing retrospective college-football weather model into a realistic operational setting.

The study is intentionally isolated from production. No live model, weekly board, picks, prospective ledger, or `main` branch behavior may change unless a later evidence gate is passed in a separate promotion decision.

## Primary questions

1. How accurate were archived pregame weather forecasts at fixed decision times of 24, 12, and 6 hours before kickoff?
2. How much does GENERAL HGB out-of-sample accuracy change when retrospective weather is replaced by weather that was actually forecast before kickoff?
3. Do weather effects that appear useful with retrospective weather persist when only archived forecast information is available?
4. How late does weather guidance need to be updated before a material model-accuracy improvement appears?
5. Is any degradation caused by forecast error/distribution shift recoverable by training the model on archived forecast inputs rather than retrospective weather inputs?

## Research population

Primary population:

- regular-season games
- seasons 2019 through 2025
- FBS vs FBS
- outdoor games only
- valid kickoff time
- closing total available
- final score available
- venue coordinates available
- archived forecast source and station mapping pass all QC gates

2018 is excluded from the primary sample because the preferred NBM/NBS archive begins on 7 November 2018, which would create a partial-season source boundary.

All lead-time variants must score the exact same common sample for paired comparisons.

## Preferred forecast source

### Primary candidate: archived NBM short-range text guidance (NBS)

The preferred source is Iowa State University's Iowa Environmental Mesonet archive of National Blend of Models short-range text guidance (`NBS`). The NBM is a calibrated blend intended to provide a nationally consistent starting point for NWS/NDFD forecasts, making it more operationally comparable to the NWS forecasts used by the live project than a single raw deterministic model.

Source feasibility must verify before acquisition:

- archive coverage across the primary seasons
- station coverage near the football venues
- exact semantics of the IEM `runtime` field
- forecast valid-time fields and cadence
- availability and units of temperature, wind speed, wind direction, dew point, precipitation probability/QPF, snowfall, and gust if present
- whether archived runtime is an issuance timestamp or a nominal model-cycle timestamp

If runtime is an issuance timestamp, the latest runtime at or before each decision cutoff is eligible. If runtime is only a nominal model-cycle timestamp, a conservative documented dissemination lag must be frozen before model results are viewed.

### Fallback candidate: GFS MOS

If NBS coverage or semantics fail, GFS MOS from the same IEM archive is the preferred fallback because its archive extends well before the research period and provides site-specific operational guidance. A source change requires a documented feasibility decision before model fitting.

## Decision-time cutoffs

For each game, define exact UTC cutoffs:

- `cutoff_24h = kickoff_utc - 24 hours`
- `cutoff_12h = kickoff_utc - 12 hours`
- `cutoff_6h = kickoff_utc - 6 hours`

No forecast bulletin whose eligible availability time is later than the cutoff may be used for that lead.

The selected forecast must have been available in its entirety by the decision cutoff. Post-cutoff runs are prohibited even if they verify the kickoff more accurately.

## Kickoff valid-time alignment

NBS short-range bulletins are expected to have approximately 3-hourly valid times. For continuous variables, the primary alignment is the forecast valid time nearest to kickoff, provided the absolute valid-time offset is no more than 90 minutes.

Using a forecast valid time after kickoff is not look-ahead leakage if that forecast value was issued before the decision cutoff; it is still a pregame prediction of the future state. The issuance/availability cutoff, not the forecast valid time, controls leakage.

If the source cadence or fields differ from this assumption, the alignment rule must be revised and frozen during source feasibility, before any model result is calculated.

## Forecast variables

Primary continuous forecast fields:

- temperature
- sustained wind speed
- wind direction
- dew point
- derived relative humidity when possible

Primary precipitation fields:

- precipitation probability, preferably a 6-hour probability covering kickoff
- quantitative precipitation forecast if consistently available
- snowfall/categorical snow if consistently available

Gust is diagnostic only in this project and is not a candidate production feature; the separate wind-gust study already failed its preregistered promotion gate.

## Verification against retrospective weather

For continuous variables, report at each lead:

- paired count
- mean error / bias (`forecast - retrospective`)
- MAE
- RMSE
- Pearson correlation
- median absolute error
- p90 absolute error
- error by season

For wind direction, use circular angular error.

For precipitation probability, do not treat probability as a deterministic amount. Evaluate:

- Brier score against retrospective precipitation occurrence
- calibration/reliability bins when sample size permits
- discrimination by forecast-probability bins

For QPF, if a consistent quantitative field can be mapped to the retrospective precipitation variable, report amount-error metrics separately.

Because station-based guidance and stadium retrospective weather are not colocated, also report station-to-venue distance and stratify verification by distance. Forecast error versus retrospective stadium weather is therefore interpreted as *operational input discrepancy*, not pure atmospheric model error.

## Station mapping

Map each stadium to the nearest eligible NBS/MOS station using great-circle distance.

Primary maximum station distance: 30 miles (48.28 km).

Sensitivity bands:

- <= 10 miles
- >10 to 20 miles
- >20 to 30 miles

Games farther than 30 miles from an eligible station are excluded from the primary paired model comparison and reported explicitly as source-coverage failures. The threshold may not be widened after seeing model results.

Station identity must remain fixed by venue across all lead times for a season unless the station has no archived guidance for that season. Any fallback station must be recorded.

## Model experiments

All experiments use chronological walk-forward testing, training on prior seasons only. No random train/test split is permitted.

### A. Current-production realism test

This is the primary operational test.

For every test fold:

1. Train the existing GENERAL HGB exactly as it is trained today using the existing retrospective historical feature columns.
2. Score the same test games four ways:
   - retrospective-weather control
   - 24-hour archived forecast weather substituted into the weather fields
   - 12-hour archived forecast weather substituted into the weather fields
   - 6-hour archived forecast weather substituted into the weather fields
3. Keep all non-weather inputs identical across the four scores.

This directly measures the train-on-retrospective / score-on-forecast distribution shift that exists in the live workflow.

### B. Forecast-trained realism test

Secondary experiment.

Train separate HGB variants where the weather features in both training and test data come from the same archived lead-time source:

- forecast-trained 24h
- forecast-trained 12h
- forecast-trained 6h

This tests whether forecast-input training can recover skill lost by the production-style distribution shift.

### C. Weather-effect persistence

On the common sample, repeat the already-used weather feature-group/interaction comparisons with:

- retrospective weather
- 24h forecast weather
- 12h forecast weather
- 6h forecast weather

The purpose is not to discover new thresholds. It is to determine whether previously observed weather effects retain direction and magnitude when the input is genuinely pregame.

## Primary model metrics

For every paired model comparison:

- MAE
- RMSE
- signed projection bias
- game-level paired MAE delta
- per-season MAE delta
- game-level bootstrap 95% CI of paired MAE delta
- season-cluster bootstrap 95% CI
- number and percentage of test seasons improved

Positive `challenger - reference` MAE delta is worse.

## Lead-time improvement tests

Primary lead comparisons:

- 12h vs 24h
- 6h vs 12h
- 6h vs 24h

A later update is considered *materially useful* only if all are true:

1. mean paired MAE delta < 0
2. game-level 95% bootstrap CI is entirely below 0
3. season-cluster 95% CI is entirely below 0
4. later lead improves MAE in at least 70% of test seasons
5. absolute MAE improvement is at least 0.03 points

The 0.03-point floor prevents statistically detectable but operationally trivial improvements from being over-interpreted.

## Production-realism evidence gate

The current retrospective-weather training setup is considered operationally robust at a lead time only if its forecast-weather version does not materially degrade relative to the retrospective-weather control.

A degradation warning is triggered when all are true:

1. mean paired MAE delta > 0
2. game-level 95% bootstrap CI entirely above 0
3. season-cluster 95% CI entirely above 0
4. worse MAE in at least 70% of test seasons
5. absolute degradation >= 0.03 points

If this warning triggers, no production change is made automatically. The forecast-trained experiment determines whether the mismatch can be reduced by retraining.

## Forecast-trained promotion gate

A forecast-trained lead-time model may only advance to a later shadow-study proposal if it beats the current production-realism implementation at the same lead and satisfies all of:

1. mean paired MAE delta < 0
2. game-level 95% bootstrap CI entirely below 0
3. season-cluster 95% CI entirely below 0
4. improvement in at least 70% of test seasons
5. absolute MAE improvement >= 0.03 points

Passing this gate does not modify production. It only justifies a separate prospective shadow test.

## Stop rules

Stop without model fitting if source feasibility shows:

- inadequate multi-season archive coverage
- ambiguous run/issuance timing that cannot be made leak-safe
- station mapping coverage below 90% of otherwise eligible games
- key variables are inconsistently populated
- the archive cannot be acquired reproducibly within reasonable request limits

Stop without promotion if:

- forecast-trained variants do not pass the preregistered gate
- only isolated seasons or post-hoc weather slices improve
- results depend on widening the station-distance threshold
- results require changing lead times after outcome inspection

## Output boundary

All files must remain under dedicated research paths such as:

- `outputs/pregame_forecast_realism/`
- `data/processed/pregame_forecast_*.csv`

No workflow may modify:

- live weekly board/pick/card outputs
- prospective ledgers
- model promotion files
- production configuration
- `main`

## Research sequence

1. Source feasibility and runtime-semantics smoke test.
2. Stadium-to-station mapping and coverage audit.
3. Small geographically distributed acquisition pilot.
4. Forecast-vs-retrospective weather verification.
5. Full 2019-2025 archived acquisition if QC passes.
6. Frozen current-production realism model experiment.
7. Frozen forecast-trained experiment.
8. Lead-time materiality tests.
9. Document advance/stop decision.
