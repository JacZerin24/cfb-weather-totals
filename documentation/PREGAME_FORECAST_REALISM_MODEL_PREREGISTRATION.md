# Pregame Forecast Realism Model Preregistration

This note freezes implementation details before the full archived NBM dataset is available to the model-analysis code. It supplements `PREGAME_FORECAST_REALISM_PROTOCOL.md` and does not change production.

## Source and leakage rules

- Source: IEM archived National Blend of Models short-range text guidance (`NBS`).
- Primary seasons: 2019-2025 regular season.
- Primary population: outdoor FBS-vs-FBS games with a closing total, final score, valid kickoff, venue coordinates, and an archived NBM station within 30 miles.
- Decision leads: exactly 24h, 12h, and 6h before kickoff.
- Conservative availability time: `NBM runtime + 2 hours`.
- A run is eligible only when `runtime + 2h <= decision cutoff`.
- Kickoff forecast valid time must be within +/-90 minutes of kickoff.
- All lead-time comparisons use the exact same complete three-lead game sample.

## Current-production realism experiment

For each test season, the existing GENERAL HGB is fit on **all available historical modeling rows from prior seasons only**, matching the current production training convention. The test population is the frozen common outdoor FBS-vs-FBS NBM sample for that season.

Each fitted fold is scored four ways on identical games:

1. retrospective-weather control;
2. 24h NBM forecast weather;
3. 12h NBM forecast weather;
4. 6h NBM forecast weather.

No model is refit between those four test representations.

### Primary forecast substitution

Replace the model's historical weather inputs with NBM values for:

- `temperature_f`
- `dewpoint_f`
- `humidity`
- `wind_mph`
- `precipitation`
- `snowfall`

Recompute `wind_bin` and `temp_bin` from the substituted values. Set `pressure` missing (`NaN`) because the live NWS forecast path does not supply pressure; realized historical pressure may not be retained in a forecast-weather challenger.

NBM wind direction and precipitation probability are retained for verification/diagnostics but are not added as new GENERAL HGB features in this study.

### Core-continuous sensitivity

A sensitivity version substitutes only temperature, dew point, humidity, and sustained wind, while leaving historical precipitation/snowfall unchanged. Pressure is still set missing. This isolates the main continuous-weather distribution shift from the different temporal semantics of NBM six-hour precipitation fields.

The sensitivity is descriptive and cannot override the primary result.

## Forecast-trained experiment

For each lead, a separate GENERAL HGB is trained and tested using archived forecast weather at that same lead. Because forecast inputs begin in 2019, forecast-trained folds require at least **1,000 prior-season common-sample training games**, matching the existing model-bakeoff minimum-training convention. This is expected to make 2021 the first eligible forecast-trained test season.

Two controls are reported:

1. **production-style control:** the existing all-history retrospective-trained HGB scored with forecast weather at that lead;
2. **matched-population control:** an HGB trained on the same prior common-sample games but with retrospective weather, then scored with forecast weather.

The matched-population control helps distinguish training-population effects from weather-representation effects. The preregistered promotion gate remains the comparison against the production-style control.

## Frozen metrics

For every model variant:

- MAE
- RMSE
- signed prediction bias
- per-season MAE

For every paired comparison:

- challenger minus reference absolute-error delta by game
- mean paired MAE delta
- 5,000-resample game bootstrap 95% CI
- 5,000-resample season-cluster bootstrap 95% CI
- number and fraction of seasons improved/worsened

Positive MAE delta is worse.

## Frozen comparisons

### Production realism

At each lead:

`production_forecast_{lead}h - retrospective_control`

A material degradation warning requires all of:

- mean delta > 0
- game-bootstrap CI entirely above 0
- season-cluster CI entirely above 0
- worse in >=70% of test seasons
- absolute degradation >=0.03 points

### Update timing

Primary lead comparisons:

- 12h vs 24h
- 6h vs 12h
- 6h vs 24h

A later update is materially useful only when all of:

- mean delta < 0
- both 95% CIs entirely below 0
- improves >=70% of test seasons
- MAE improvement >=0.03 points

### Forecast-trained model

At each lead:

`forecast_trained_{lead}h - production_forecast_{lead}h`

Advance to a future prospective shadow proposal only if all of:

- mean delta < 0
- both 95% CIs entirely below 0
- improves >=70% of shared test seasons
- MAE improvement >=0.03 points

No passing result automatically changes production.

## Weather-effect persistence diagnostics

The following already-used, pre-existing weather screens are reported under retrospective, 24h, 12h, and 6h representations without optimizing thresholds:

- sustained wind >=15 mph
- sustained wind >=20 mph
- temperature <=35 F
- temperature <=40 F and wind >=12 mph
- closing total >=60 and wind >=10 mph
- closing total >=58 and wind >=12 mph
- closing total >=60 and relative humidity >=80%
- any six-hour NBM QPF / retrospective precipitation >0 (diagnostic because temporal definitions differ)

For each screen report game count, average market residual, and under rate when n>=25. These diagnostics cannot override the primary paired model evidence gates.

## Safety boundary

No workflow or script in this study may alter live boards, picks, cards, prospective ledgers, production configuration, or `main`. Model outputs remain under `outputs/pregame_forecast_realism/`.