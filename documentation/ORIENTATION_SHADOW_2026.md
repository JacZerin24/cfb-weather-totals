# 2026 Stadium-Orientation Shadow Evaluation

**Status: research only. No production decision effect.**

## Purpose

This track prospectively evaluates whether stadium-relative crosswind adds incremental value to the existing GENERAL HGB college-football totals model.

## Frozen challenger

- Version: `orientation-crosswind-hgb-v0.1`
- Track: FBS vs FBS only
- Baseline: existing GENERAL HGB feature set and model
- Only added feature: `crosswind_mph`
- Crosswind formula: `wind_mph * abs(sin(relative field/wind angle))`
- Qualifier logic is intentionally unchanged for comparison: UNDER prediction of at least 3.5 points with market total at least 56
- No 2026 result may be used to tune this version. A changed feature set or threshold requires a new version and a new prospective start time.

## Operational isolation

The live workflow first builds the normal weekly board and archives/verifies the official prospective ledger. Only after that step completes does `src.orientation_shadow_capture` run.

The shadow step is `continue-on-error: true`. It writes only under `outputs/orientation_shadow/2026/` and does not modify:

- `outputs/weekly_board.csv`
- `outputs/weekly_picks.csv`
- the Qualifier/Lean/No Play classification used operationally
- the official prospective 2026 ledger
- the FCS model or FCS qualifier rules

A shadow failure therefore cannot block or alter the existing operational workflow.

## Captured fields

Each snapshot retains the baseline and challenger prediction/status side by side plus the market total and environmental context, including:

- NWS kickoff wind speed and shadow-captured wind direction
- measured field axis
- relative wind/field angle
- crosswind and along-field wind components
- baseline predicted market residual and status
- challenger predicted market residual and status
- whether the challenger would change the status
- GitHub run metadata and capture time

## Review criteria

The challenger should be judged on prospective prediction error/calibration and on decision-support value, especially baseline/challenger disagreement games and status migrations. Betting hit rate, ROI, and CLV are secondary economic outcomes, not substitutes for model robustness.

No production promotion is implied by this shadow track. Any future promotion requires a separately documented versioned decision and applies prospectively only.
