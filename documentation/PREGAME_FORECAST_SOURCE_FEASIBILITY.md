# Pregame Forecast Source Feasibility

## Status

Primary source candidate: **IEM archived National Blend of Models short-range text guidance (NBS)**.

This document freezes source/timing decisions before any forecast-realism model results are calculated.

## Why NBS

The current live project trains GENERAL HGB on retrospective historical weather but scores upcoming games with an NWS kickoff forecast. NBS is not a literal archive of each WFO-edited NWS forecast, but it is a calibrated National Blend of Models text product designed as a nationally consistent operational starting point for NWS/NDFD forecasts. It is therefore a much closer operational proxy than a single raw deterministic model.

The IEM archive begins 7 November 2018. The primary research period is therefore 2019-2025 so that every included season has full-season source availability.

## Source smoke test

GitHub Actions source smoke run: `34211343146`.

Four geographically distributed archive requests were tested:

| Case | Station | Date | HTTP | Rows |
| --- | --- | --- | ---: | ---: |
| New Orleans | KMSY | 2019-09-07 | 200 | 105 |
| Seattle | KSEA | 2021-09-04 | 200 | 92 |
| Denver | KDEN | 2023-09-02 | 200 | 92 |
| Boston | KBOS | 2025-09-06 | 200 | 92 |

All four returned structured JSON records with explicit `runtime` and `ftime` timestamps.

Observed primary fields included:

- `tmp`: temperature, degrees F
- `dpt`: dew point, degrees F
- `wdr`: wind direction, degrees in the IEM parsed output
- `wsp`: sustained wind speed, knots
- `gst`: gust, knots
- `p06`, `p12`: precipitation probabilities
- `q06`, `q12`: quantitative precipitation guidance
- `s06`: six-hour snow guidance when available
- precipitation-type probabilities (`pra`, `pzr`, `psn`, `ppl`)

The forecast valid-time cadence in the tested NBS records was three hours and the short-range horizon was sufficient for the 24h, 12h, and 6h decision-time experiment.

## Frozen availability-time rule

The IEM record `runtime` is treated as the **nominal NBM text-product cycle/runtime**, not as proof that the guidance was instantaneously available to an operational user.

NOAA documentation shows NBM text-product dissemination generally occurs within roughly the first hour surrounding/after the nominal cycle, with cycle-specific timing that has changed between NBM versions.

To make the retrospective experiment leak-safe across versions, the primary study freezes the following conservative rule:

> `eligible_availability_time = runtime + 2 hours`

A forecast cycle may be selected for a 24h/12h/6h cutoff only when this eligible availability time is at or before the cutoff.

This two-hour lag is deliberately more conservative than documented dissemination timing. It may make the selected forecast older than what a real forecaster could often have accessed, but it prevents nominal-cycle timestamps from being mistaken for instantaneous availability.

This lag may not be reduced after model outcomes are viewed. Any zero-lag or shorter-lag analysis would be explicitly labeled a sensitivity analysis and cannot replace the primary result.

## Valid-time alignment

NBS short-range guidance is three-hourly. For the continuous kickoff variables, the primary selected forecast valid time is the valid time nearest kickoff with an absolute offset no greater than 90 minutes.

A valid time after kickoff is allowed when the entire forecast bulletin was already eligible before the decision cutoff. The leak-safety boundary is forecast availability, not forecast valid time.

## Remaining feasibility gates

Before full forecast acquisition or model fitting:

1. Map eligible research stadiums to nearby NBM-capable station candidates.
2. Require at least 90% of otherwise eligible games to have a station within the preregistered 30-mile radius.
3. Run a geographically distributed station/archive pilot across multiple seasons and all three lead times.
4. Confirm selected cycles obey the `runtime + 2h <= cutoff` rule.
5. Confirm kickoff valid-time offset <=90 minutes.
6. Confirm key fields have adequate completeness.
7. Only then begin full 2019-2025 acquisition.

No production or live-model files are changed by this feasibility work.
