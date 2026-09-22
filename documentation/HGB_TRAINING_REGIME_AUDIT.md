# HGB Training-Regime Audit — September 22, 2026

## Problem

The production GENERAL HGB used `HistGradientBoostingRegressor(max_iter=250, ...)` without explicitly setting `early_stopping`. With scikit-learn's `early_stopping='auto'`, the estimator changes behavior when the training sample exceeds 10,000 rows.

The walk-forward training set first crossed that boundary for the 2025 test season:

| Test season | Training games | auto n_iter | auto avg abs edge | auto FBS under edges >=3.5 |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 8,710 | 250 | 1.931 | 82 |
| 2025 | 10,213 | 15 | 0.628 | 0 |

This was a software/model-integrity change caused by sample size, not an intentional research decision.

## Corrective training regime

The GENERAL HGB and FCS-only HGB now explicitly set:

```python
early_stopping=False
```

Under the corrected 2025 walk-forward fit, the HGB runs all 250 iterations, average absolute predicted edge is 1.716 points, and 70 FBS games have an UNDER prediction of at least 3.5 points before applying a total filter.

## FBS production-screen audit

A research-only workflow restored the exact live historical training artifact and compared FBS-vs-FBS UNDER screens across edge thresholds 2.5–5.0 and total floors 52–60. Selection used historical data only through 2025; no 2026 game outcomes were used.

The chosen production candidate is **HGB UNDER edge >=4.0 with market total >=56**.

| Evaluation | Graded | Wins | Losses | Hit rate | Net units at -110 | ROI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All walk-forward seasons | 643 | 376 | 267 | 58.48% | +74.82 | +11.64% |
| 2022–2025 | 172 | 106 | 66 | 61.63% | +30.36 | +17.65% |
| 2025 holdout | 29 | 16 | 13 | 55.17% | +1.55 | +5.33% |

The 4.0/56 screen was profitable in **10 of 10** walk-forward test seasons and **4 of 4** seasons from 2022–2025. A 4.0/58 screen had slightly higher aggregate ROI but was profitable in 9 of 10 seasons, so 4.0/56 was preferred for stability.

## Current-slate preview

A research-only live rescore of the Week 4 slate with the corrected HGB produced:

- **Ole Miss at Florida — UNDER 60.5**: model projection 54.86, residual **-5.64** → meets 4.0/56.
- **Notre Dame at Purdue — UNDER 57.5**: model projection 53.64, residual **-3.86** → does not qualify under 4.0/56 and is retained as a LEAN.

The preview is not an official prospective entry. Official status begins only after protocol 2026.6 is merged and a normally eligible scheduled snapshot is captured.

## Change-control decision

Protocol 2026.6 applies the corrected estimator and 4.0/56 GENERAL HGB screen prospectively. It does not reclassify Weeks 1–3, rewrite immutable snapshots, or use 2026 outcomes to select the new threshold. FCS qualifying rules remain 7.5/56.
