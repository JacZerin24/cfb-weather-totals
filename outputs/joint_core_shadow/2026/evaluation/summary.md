# 2026 Baseline vs Joint-Core Shadow Evaluation

**Research only. This report cannot alter the operational weekly board, weekly picks, thresholds, or official prospective ledger.**

Protocol: `joint-core-eval-2026.2`  
Challenger: `joint-core-weather-context-hgb-v0.1`  
Prospective signal: **PROSPECTIVE_MAE_FAVORABLE_UNCERTAIN**  
Graded paired joint-core-ready games: **357**

## Primary paired model metric

- Baseline MAE: **12.714** points
- Challenger MAE: **12.654** points
- Challenger minus baseline MAE: **-0.060** points (negative is better)
- 95% paired bootstrap interval: **[-0.239, +0.118]**

## Decision-support diagnostics

- Status disagreements: **16**
- Qualifier disagreements: **7**
- Scorable status migrations: **16**
- Status-migration accuracy: **0.500**

## Qualifier economics (supporting evidence)

- Baseline: 3-4-0, ROI -0.182, avg CLV +0.750
- Challenger: 1-7-0, ROI -0.761, avg CLV +0.583

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, weekly splits, and individual disagreement games are secondary evidence. No result here changes 2026 production automatically; any 2027 consideration requires a separate versioned decision after the predeclared reviews.