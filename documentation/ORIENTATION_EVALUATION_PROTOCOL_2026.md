# 2026 Baseline vs Orientation Prospective Evaluation Protocol

**Frozen 2026-08-31 before any official orientation-shadow game outcome. Research only.**

This protocol evaluates `orientation-crosswind-hgb-v0.1` against the existing GENERAL HGB baseline without changing the live board, Qualifier/Lean decisions, or the official prospective ledger.

## Official comparison entry

The orientation comparison deliberately mirrors the official 2026 prospective-entry timing:

- scheduled weekly workflow only
- eligible schedules: Thursday 14Z, Friday 14Z, Saturday 13Z
- snapshot must be at least 120 minutes before kickoff
- for each game, use the latest eligible snapshot meeting that lead-time requirement
- for duplicate attempts of one GitHub run, retain the earliest successful attempt
- push/manual snapshots are archived but never count as official comparison entries

The baseline and challenger are scored from the **same selected shadow snapshot and same stored market total**. Only FBS games with a usable measured field orientation, wind direction/crosswind value, and both model predictions enter the paired comparison.

## Primary metric

The primary metric is paired projected-total mean absolute error (MAE):

- baseline projected total = snapshot total + baseline predicted market residual
- challenger projected total = snapshot total + challenger predicted market residual
- primary delta = challenger MAE minus baseline MAE
- negative delta favors the challenger

Uncertainty is a predeclared 10,000-replicate paired bootstrap over games using seed `20260831`. RMSE and signed projection bias are secondary model diagnostics.

## Decision-support metrics

Statuses are ordered `QUALIFIES > LEAN > NO PLAY/WATCH/NO LINE` for migration analysis. A challenger move toward stronger UNDER support is scored correct when the game finishes under the selected entry total; a move toward weaker UNDER support is scored correct when the game finishes over. Pushes are neutral.

The report separately tracks all status disagreements, qualifier disagreements, and migration accuracy.

## Qualifier economics

Baseline and challenger qualifiers are graded separately at the frozen -110 paper price. Reports include W-L-P, hit rate excluding pushes, ROI per 1-unit stake, and average CLV when the existing immutable near-kickoff benchmark is available. For an UNDER, CLV is entry total minus benchmark close total, so positive is favorable.

These economic metrics are supporting evidence. They cannot override the primary paired prediction comparison on their own.

## Review points and freeze

Descriptive grades may be rebuilt after games finish. Formal model-promotion reviews are predeclared for after Week 6, after Week 10, and after the 2026 regular season.

No automatic promotion occurs. `orientation-crosswind-hgb-v0.1` cannot be tuned from 2026 outcomes. A different feature set, model specification, or threshold becomes a new challenger version with a new prospective start time. Any production promotion must be separately documented and applies only prospectively from its declared effective time.
