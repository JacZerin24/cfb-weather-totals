# Pregame Forecast Full Acquisition Plan

This plan is frozen after source/station/pilot feasibility but **before any forecast-realism model fitting**.

## Feasibility evidence entering full acquisition

- IEM archived NBM short-range text guidance (NBS) returned structured historical data in 2019, 2021, 2023, and 2025 source-smoke cases.
- The actual 2019-2025 outdoor FBS-vs-FBS modeling scope contains 4,833 games.
- 4,829/4,833 games (99.917%) have an active ASOS/NBM station candidate within the preregistered 30-mile radius.
- A geographically distributed 64-game pilot reconstructed all three 24h/12h/6h decision-time forecasts for all 64 games.
- Five deterministic timing/alignment tests passed before the pilot.

## Full acquisition population

The acquisition population is the 4,829 games that pass the station-distance gate. The four >30-mile games remain excluded and are reported as source-coverage exclusions; the radius will not be widened.

For each game, the same station must supply all three lead times. The primary candidate is the nearest season-active station. If it cannot supply all three lead times, candidate stations are tried in preregistered nearest-distance order, still limited to 30 miles. The selected station and fallback rank are recorded.

## Request strategy

To avoid downloading irrelevant model cycles, each game request only asks for NBS runtimes from:

- `kickoff - 40 hours`
- through `kickoff - 8 hours`

The latest permissible 6h decision-time cycle is `kickoff - 8h` because the frozen two-hour availability lag requires `runtime + 2h <= kickoff - 6h`.

The 40-hour lower bound leaves more than a full NBM cycle interval before the latest permissible 24h runtime and is intentionally wider than the minimum required.

Requests are resumable. Successful game/station/lead reconstructions are cached and never re-requested in later recovery chunks unless the user explicitly starts a new research version.

## Frozen timing guards

For every selected lead row:

- `forecast_availability_utc = runtime + 2 hours`
- availability must be <= the decision cutoff
- forecast valid-time absolute offset from kickoff must be <=90 minutes
- selected station distance must be <=30 miles
- all three leads for a game must use the same selected station

Any violation fails closed.

## Primary forecast fields

Continuous kickoff fields:

- temperature
- dew point
- derived relative humidity
- sustained wind speed
- wind direction

Six-hour period fields covering kickoff:

- precipitation probability (P06)
- quantitative precipitation (Q06)
- snowfall (S06)

NBS gust is retained only as diagnostic metadata and is not a candidate production feature.

## Temporal-support caveat for precipitation

NBS P06/Q06/S06 are six-hour period products, whereas the live NWS grid values and retrospective CFBD-derived precipitation fields do not necessarily have identical temporal support. Therefore:

- precipitation verification is interpreted as **operational input discrepancy**, not pure forecast error
- the primary model-realism experiment still substitutes these NBS period values because they are the best available archived pregame proxy
- a **continuous-weather-only sensitivity** is frozen now, before outcomes, using only temperature, dew point/humidity, sustained wind, and the existing non-weather features while leaving precipitation/snow unavailable in forecast variants
- this sensitivity may diagnose temporal-support dependence but may not replace the primary result post hoc

## Pressure treatment

The current live NWS forecast path does not supply `pressure`, while the historical GENERAL HGB feature list can include pressure. To mimic the operational distribution:

- forecast-weather scoring variants set pressure to missing (`NaN`), allowing the existing pipeline imputer to handle it
- forecast-trained variants also use missing pressure in both training and testing
- retrospective-weather control retains the existing retrospective pressure field

This rule is frozen before model outcomes.

## Full-acquisition QC gate

Model fitting is prohibited unless all conditions are met:

1. At least **95%** of the 4,829 station-mapped games have complete 24h/12h/6h reconstructions on one station.
2. No season has common three-lead coverage below **90%**.
3. No selected row violates the two-hour availability rule.
4. No selected kickoff valid-time offset exceeds 90 minutes.
5. No selected station exceeds 30 miles.
6. No duplicate `game_id + lead_hours` rows exist.
7. Core continuous fields (temperature, dew point, sustained wind, wind direction) are present in at least 99% of otherwise complete selected lead rows.
8. Request failures and station fallbacks are explicitly reported.

The common-sample model dataset is the intersection of games complete at all three leads.

## Full verification outputs before model fitting

For 24h, 12h, and 6h, report overall and by season:

- temperature bias/MAE/RMSE/correlation
- dew-point bias/MAE/RMSE/correlation
- humidity bias/MAE/RMSE/correlation
- sustained-wind bias/MAE/RMSE/correlation
- circular wind-direction absolute error
- precipitation amount discrepancy where paired
- P06 Brier score against retrospective precipitation occurrence
- station-distance distributions
- runtime age at decision cutoff and kickoff
- fallback-station rate

## Frozen model test-season handling

### Current-production realism

The existing GENERAL HGB may train on all prior retrospective seasons available in the historical modeling artifact. Forecast-weather scoring is evaluated on the common NBS sample in 2019-2025, subject to the existing minimum training/test-game requirements.

### Forecast-trained realism

Because full-season NBS data begins in 2019, forecast-trained variants can only train on archived forecast-weather seasons. With 2019 (713 games before station exclusions) plus 2020 (490), the first test season expected to satisfy the existing 1,000-game minimum is **2021**. Therefore the forecast-trained primary paired evaluation is expected to cover 2021-2025.

The exact scored seasons are determined only by the frozen minimum-game rules, not by outcome quality.

## No production implication

Passing acquisition QC only authorizes retrospective model fitting on the research branch. It does not alter `main`, live forecasts, weekly cards, picks, or prospective ledgers.
