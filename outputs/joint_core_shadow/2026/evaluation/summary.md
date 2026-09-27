# 2026 Baseline vs Joint-Core Shadow Evaluation

**Research only. This report cannot alter the operational weekly board, weekly picks, thresholds, or official prospective ledger.**

Protocol: `joint-core-eval-2026.2`  
Challenger: `joint-core-weather-context-hgb-v0.1`  
Prospective signal: **PROSPECTIVE_MAE_FAVORABLE_UNCERTAIN**  
Graded paired joint-core-ready games: **152**

## Primary paired model metric

- Baseline MAE: **12.012** points
- Challenger MAE: **11.964** points
- Challenger minus baseline MAE: **-0.047** points (negative is better)
- 95% paired bootstrap interval: **[-0.249, +0.150]**

## Decision-support diagnostics

- Status disagreements: **3**
- Qualifier disagreements: **2**
- Scorable status migrations: **3**
- Status-migration accuracy: **0.333**

## Qualifier economics (supporting evidence)

- Baseline: 0-0-0, ROI +nan, avg CLV +nan
- Challenger: 0-2-0, ROI -1.000, avg CLV +0.250

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, weekly splits, and individual disagreement games are secondary evidence. No result here changes 2026 production automatically; any 2027 consideration requires a separate versioned decision after the predeclared reviews.