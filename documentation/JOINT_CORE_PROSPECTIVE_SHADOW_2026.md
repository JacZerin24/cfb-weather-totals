# Frozen 2026 Joint-Core Prospective Shadow

## Status

Research only. This challenger cannot alter the 2026 production model, weekly board, official picks, thresholds, or official prospective ledger.

## Why this exists

The retrospective research sequence found that `joint_core` was the strongest weather-context challenger and that the real along-field component became statistically distinguishable from randomized stadium axes only inside the fixed joint weather/climate context. The final reconciliation result justified one next step only: prospective validation on genuinely unseen post-freeze games.

This shadow is that prospective test. It does not reopen retrospective feature search.

## Freeze

- Challenger version: `joint-core-weather-context-hgb-v0.1`
- Evaluation protocol: `joint-core-eval-2026.1`
- Freeze / prospective start: `2026-09-04T14:00:00Z`
- No snapshot before that instant may ever become an official joint-core evaluation entry.
- No 2026 outcomes may enter the challenger training sample.

## Frozen challenger

The challenger uses the same GENERAL `HistGradientBoostingRegressor` specification as production:

- `max_iter=250`
- `learning_rate=0.04`
- `l2_regularization=0.5`
- `min_samples_leaf=35`
- `random_state=42`

It uses the frozen GENERAL baseline feature set plus exactly these seven numeric features:

1. `crosswind_mph`
2. `alongwind_mph`
3. `venue_latitude`
4. `temperature_anomaly_f`
5. `wind_local_percentile`
6. `temperature_latitude_interaction`
7. `wind_latitude_interaction`

There are no manually engineered joint interaction terms. The retrospective `joint_interactions` variant was worse than `joint_core` and is not part of this challenger.

## Historical context construction

The model and climate reference are built only from restored historical modeling data with `season < 2026`.

Temperature context uses outdoor historical venue-month means with latitude-band/month, calendar-month, then global fallbacks. Local wind context uses the analogous empirical wind distributions. The construction is frozen for this challenger version.

## Geometry

Stadium axes are undirected 0-180 degree axes. Wind direction is the meteorological FROM direction. The minimum field-relative separation is constrained to 0-90 degrees.

- `crosswind_mph = wind_mph * sin(field_relative_angle)`
- `alongwind_mph = wind_mph * cos(field_relative_angle)`

The live shadow obtains wind direction separately from the NWS kickoff forecast and combines it with the committed stadium orientation reference. Missing geometry makes the game ineligible for the primary paired evaluation.

## Official prospective cohort

A game can enter the official joint-core shadow sample only when all of the following are true:

- FBS track
- FBS vs FBS
- outdoor
- climate context ready
- field orientation ready
- baseline and challenger predictions both exist
- final game result exists when graded
- snapshot was captured on or after the freeze time
- snapshot came from an eligible scheduled Thu/Fri/Sat safety or freshness run
- snapshot was at least 120 minutes before kickoff

Monday, push, manual, and watchdog-dispatched captures may be archived for diagnostics but can never become official evaluation entries.

For each game, the evaluator uses the latest eligible snapshot at least 120 minutes before kickoff. If the same GitHub run is retried, the earliest successful attempt is retained before that game-level selection.

## Run order and isolation

The weekly workflow performs the official prospective-ledger step first. Only afterward does it run research-only shadows. The joint-core capture is `continue-on-error`, so it cannot block production or the official ledger.

Files are written only under:

`outputs/joint_core_shadow/2026/`

Each snapshot is immutable and content-hashed. `latest.csv` is only a convenience view; formal grading reselects from immutable snapshots under the frozen protocol.

## Primary metric

The primary metric is paired projected-total MAE on the same official games:

- baseline error = `abs((entry total + baseline predicted residual) - actual total)`
- challenger error = `abs((entry total + challenger predicted residual) - actual total)`
- reported delta = challenger MAE minus baseline MAE

Negative is better. Uncertainty is a fixed 10,000-repetition paired game bootstrap.

Prospective labels are predeclared:

- `INSUFFICIENT_SAMPLE`: fewer than 5 graded paired games
- `PROSPECTIVE_MAE_SUPPORTED`: mean delta < 0 and 95% bootstrap upper bound < 0
- `PROSPECTIVE_MAE_FAVORABLE_UNCERTAIN`: mean delta < 0 but CI includes 0
- `PROSPECTIVE_MAE_NOT_FAVORABLE`: mean delta >= 0

These labels never trigger automatic promotion.

## Secondary evidence

The evaluator also reports RMSE, signed error, weekly paired MAE, status disagreements, qualifier migrations, -110 UNDER economics, and CLV against the immutable official prospective benchmark close when available.

These are supporting diagnostics. They cannot override the paired MAE result.

## Review points

Formal reviews are predeclared for:

- after Week 6, if at least 25 paired games are graded
- after Week 10
- after the 2026 regular season

No midseason tuning is allowed. A production decision for 2027, if any, requires a separate versioned decision using the completed retrospective evidence plus this prospective sample.

## What remains frozen

Throughout this shadow version, do not change based on 2026 results:

- features
- HGB hyperparameters
- training cutoff
- climate reference construction
- stadium geometry formula
- readiness rules
- UNDER qualifier thresholds
- official entry timing
- primary metric
- bootstrap specification
- grading rules

Any substantive change creates a new challenger version and a new prospective start time. It cannot rewrite this ledger.
