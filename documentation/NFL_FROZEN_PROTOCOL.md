# NFL Frozen Prospective Protocol — v1

**Frozen:** September 19, 2026  
**Status:** prospective paper tracking only  
**Protocol ID:** `nfl_totals_paper_v1`

This file records the NFL totals rule before prospective 2026 tracking begins. The purpose is to prevent outcome-driven threshold changes after games are observed.

## Official decision rule

The only official decision horizon is **24 hours before scheduled kickoff**.

A game is an official **QUALIFIES** play only when all of the following are true:

- NFL regular-season game;
- side is **OVER**;
- RF residual regression (`reg_rf_leaf25`) predicts at least **+3.0 points** versus the captured market total;
- RF OVER classifier (`cls_rf_leaf25`) gives at least **60% OVER probability**;
- a valid market total and OVER price are captured at the decision snapshot;
- required weather input is available for a fixed outdoor venue.

A qualifying play is upgraded to **STRONG** when the residual regression edge is at least **+4.0 points**, while the classifier remains at or above **60%**.

There is no forced minimum number of plays.

## Weather and roof handling

- Forecast source for the frozen research architecture: **JMA GSM**.
- Fixed outdoor stadium: use the archived/live forecast weather available at the decision time.
- Fixed dome: withhold weather features.
- Retractable roof: represent the venue as retractable and withhold weather features rather than leaking the eventual open/closed roof decision.

## Other horizons

- **48 hours:** research-only early look. It cannot create an official prospective play under v1.
- **72 hours:** disabled for official signals.

## Grading

- Paper tracking only.
- Flat **1.0 unit** per official play.
- Grade ROI using the captured OVER price.
- Also retain a flat -110 reference ROI for comparability.
- Capture a near-kickoff/closing market number to measure closing-line value.

## Change control

The thresholds and official horizon are frozen for this protocol version. They are not to be changed because of 2026 wins or losses.

Any future threshold, model, side, lead-time, roof-policy, or grading-rule change requires:

1. a new protocol version;
2. a documented reason that does not use future outcomes to rewrite v1;
3. separate validation;
4. prospective results for v1 preserved unchanged.

This protocol does **not** claim that the model has a proven real-money edge. It freezes the candidate that will now be tested prospectively.
