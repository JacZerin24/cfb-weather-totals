# NFL Team-Context Challenger — Preregistered Research Plan

**Research branch:** `research/nfl-team-context-v2`  
**Baseline protocol:** `nfl_totals_paper_v1` (unchanged)  
**Purpose:** Test whether strictly pregame team-performance information adds repeatable out-of-sample value beyond the frozen market/context + forecast-weather model.

## Non-negotiable isolation

This experiment must not modify the frozen v1 protocol, v1 thresholds, live NFL board scoring, or prospective v1 ledger. Any challenger that appears superior remains research-only until separately frozen and prospectively tracked.

## Primary question

Does pregame football context improve prediction of:

`market_residual = actual_total_points - market_total`

beyond the existing market/context + JMA forecast-weather model?

## Data

- Game/market baseline: existing NFL forecast-native dataset, 2018-2025.
- Pregame football context: nflverse play-by-play, using 2017 only as prior-history warmup and 2018-2025 as modeled seasons.
- 2025 decision-time test: existing public archived market database and the same time-matched 24h decision snapshot logic used by the v1 holdout.
- No same-game or future-game information may enter a pregame feature.

## Pregame team features

For each team-game, calculate game-level offense and defense-allowed values from scrimmage plays:

- EPA/play
- success rate
- pass EPA/dropback
- rush EPA/rush
- explosive-play rate (20+ yards)
- pass rate
- sack rate
- turnover rate
- early-down EPA/play (downs 1-2)

Pregame values are the mean of the **previous 8 completed games**, shifted by one game. The 2017 season supplies warmup history so early 2018 games do not use future information. At least four prior games are required for both teams.

No team identity, season-end ratings, final-season aggregates, current-game outcome statistics, or postgame injury information are allowed.

## Matchup representation

For each metric, create:

- two-team mean
- absolute team difference

For EPA/play, success rate, pass EPA, and rush EPA, also create offense-vs-opposing-defense matchup averages.

This representation is intentionally compact and mostly symmetric for a totals problem.

## Fixed model families

Use the same Random Forest architecture as v1:

- regression: `reg_rf_leaf25`
- classifier: `cls_rf_leaf25`

No hyperparameter search is permitted in this first challenger test.

Four fixed systems are compared:

1. **v1_baseline** — market/context + forecast weather
2. **football_context** — market/context + pregame football features, no weather
3. **combined** — market/context + forecast weather + pregame football features
4. **equal_blend** — arithmetic 50/50 blend of v1_baseline and football_context regression signals and probabilities

The 50/50 blend is fixed before results are observed. It is not optimized on the test seasons.

## Validation

Primary forecast horizon: **24 hours**.

Chronological expanding-window testing only:

- train on seasons before the test season;
- test on the next season;
- never randomly split games across time.

Development out-of-sample seasons: 2021-2024.  
Final historical holdout: **2025**.

The 2025 decision-time analysis must replace the final closing total with the archived market total actually available at the 24-hour decision snapshot, while preserving the known limitation that historical training targets use closing-market residuals.

## Fixed betting screens

Do not re-optimize thresholds.

- QUALIFIES: predicted residual >= +3.0 and P(OVER) >= 0.60
- STRONG: predicted residual >= +4.0 and P(OVER) >= 0.60

Also evaluate strict agreement sets where v1 and a challenger independently meet the same qualifier rule.

## Metrics

For every model and season report:

- residual MAE
- Brier score for Over probability
- qualifier count
- wins / losses / pushes
- hit rate
- flat -110 ROI
- Wilson 95% interval for hit rate
- one-sided binomial p-value versus -110 break-even

For the 2025 decision-time holdout additionally report ROI at captured consensus and best same-line Over prices.

## Robustness / anti-overfit checks

- development seasons and 2025 holdout reported separately;
- no threshold or hyperparameter tuning after inspecting 2025;
- paired bootstrap confidence intervals for MAE and Brier improvement versus v1;
- feature ablations for weather-only / football-only / combined;
- fixed equal-weight blend rather than optimized stacking;
- season-by-season results to expose dependence on one anomalous year;
- qualifier-intersection results to test whether independent model agreement improves selectivity.

## Interpretation

A challenger is **not** promoted merely because aggregate ROI is higher.

Evidence is considered materially encouraging only if it:

1. improves 2025 holdout MAE or Brier versus v1;
2. does not rely on a single season for the apparent advantage;
3. shows competitive or improved qualifier ROI with non-trivial volume;
4. remains directionally useful in the true 2025 decision-time market test; and
5. has bootstrap uncertainty consistent with a real improvement rather than obvious noise.

Even if all five are met, the result remains a research candidate. A new protocol version and prospective tracking are required before replacing v1.
