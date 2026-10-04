# 2026 Baseline vs Joint-Core Shadow Evaluation

**Research only. This report cannot alter the operational weekly board, weekly picks, thresholds, or official prospective ledger.**

Protocol: `joint-core-eval-2026.2`  
Challenger: `joint-core-weather-context-hgb-v0.1`  
Prospective signal: **PROSPECTIVE_MAE_FAVORABLE_UNCERTAIN**  
Graded paired joint-core-ready games: **205**

## Primary paired model metric

- Baseline MAE: **12.261** points
- Challenger MAE: **12.156** points
- Challenger minus baseline MAE: **-0.105** points (negative is better)
- 95% paired bootstrap interval: **[-0.312, +0.109]**

## Decision-support diagnostics

- Status disagreements: **8**
- Qualifier disagreements: **4**
- Scorable status migrations: **8**
- Status-migration accuracy: **0.375**

## Qualifier economics (supporting evidence)

- Baseline: 2-1-0, ROI +0.273, avg CLV +nan
- Challenger: 0-3-0, ROI -1.000, avg CLV +0.250

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, weekly splits, and individual disagreement games are secondary evidence. No result here changes 2026 production automatically; any 2027 consideration requires a separate versioned decision after the predeclared reviews.