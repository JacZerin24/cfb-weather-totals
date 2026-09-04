# Joint-Core Robustness and Ablation Research

## Status

Retrospective research only. This work has no production, live-board, weekly-pick, orientation-shadow, threshold, or official 2026 prospective-ledger effect.

The parent joint weather-context study found `joint_core` to be the strongest retrospective challenger, but it did not pass the original predeclared evidence gate because the game-level paired bootstrap interval still crossed zero. This robustness study does not alter that gate or retroactively relabel the result.

## Research questions

1. Which parts of `joint_core` actually contribute predictive information?
2. Does the observed improvement depend heavily on one test season?
3. Does the result survive different amounts of historical training data?
4. Is the signal stable enough to justify considering a separately frozen prospective shadow challenger?

## Parent model

`joint_core` equals the existing GENERAL HGB feature set plus:

- field-relative crosswind;
- field-relative along-field wind;
- venue latitude;
- leak-safe venue/local temperature anomaly;
- leak-safe local wind percentile;
- temperature × latitude context;
- wind-percentile × latitude context.

HistGradientBoosting remains the estimator so nonlinear relationships can be learned without enumerating a large interaction library.

## Predeclared component ablations

The study fits the following models on the same joint-ready test games:

- `baseline`
- `joint_core`
- `joint_no_orientation`
- `joint_no_climate`
- `joint_no_crosswind`
- `joint_no_alongwind`
- `joint_no_temperature_context`
- `joint_no_local_wind_context`
- `joint_no_latitude_context`

Each ablation is compared directly with `joint_core`.

`mae_penalty_when_removed > 0` means removing the component made the model worse.

A component is labeled `COMPONENT_SUPPORTED` only if:

- removing it increases paired MAE on average;
- the 95% game-level interval for that removal penalty is entirely above zero;
- the 95% season-cluster interval is entirely above zero;
- `joint_core` beats the ablation in at least 70% of test seasons.

A component may still matter jointly even if it is not individually isolated because nonlinear tree models can contain redundant or substitutable information.

## Alternate temporal training windows

The same baseline-vs-`joint_core` comparison is repeated with:

1. expanding history;
2. rolling six-season history;
3. rolling four-season history.

For every test season, climate context references are rebuilt only from the eligible training window. No future season contributes to a fold.

This is a robustness test only. It does not select a new production training window.

## Leave-one-test-season-out sensitivity

Using the expanding-history walk-forward predictions, the study removes each evaluation season one at a time and recomputes the paired `joint_core` minus baseline result.

This directly checks whether one strong season such as 2018 or 2020 is carrying the overall result.

It does not refit the models after removing the held-out evaluation season from older folds, because the purpose is sensitivity of the observed out-of-sample evidence rather than a new model specification.

## Secondary robustness label

`ROBUSTNESS_STRENGTHENED` requires all of the following:

- the mean `joint_core` MAE delta is negative in every training-window specification;
- the mean delta remains negative after omitting every individual test season;
- `joint_core` improves at least 70% of test seasons in every training-window specification;
- at least two training-window season-cluster 95% intervals are entirely below zero.

Otherwise the result is `ROBUSTNESS_MIXED`.

This label is deliberately separate from `SUPPORTED_RETROSPECTIVELY`. It cannot override the original evidence gate and cannot promote anything into the live model.

## Outputs

All files are written only under:

`outputs/joint_weather_context_robustness/`

Expected outputs:

- `component_ablations.csv`
- `leave_one_test_season_out.csv`
- `training_window_summary.csv`
- `robustness_indicators.csv`
- `walk_forward_diagnostics.csv`
- one prediction file per training-window specification
- `summary.md`

## Promotion boundary

Even a strong robustness result would only justify discussing a newly versioned, frozen prospective shadow challenger. The live 2026 GENERAL model, decision thresholds, weekly outputs, and official prospective ledger remain unchanged unless later prospective evidence supports a separately reviewed change.
