# 2026 Baseline vs Joint-Core Shadow Evaluation

**Research only. This report cannot alter the operational weekly board, weekly picks, thresholds, or official prospective ledger.**

Protocol: `joint-core-eval-2026.2`  
Challenger: `joint-core-weather-context-hgb-v0.1`  
Prospective signal: **PROSPECTIVE_MAE_FAVORABLE_UNCERTAIN**  
Graded paired joint-core-ready games: **521**

## Primary paired model metric

- Baseline MAE: **12.790** points
- Challenger MAE: **12.679** points
- Challenger minus baseline MAE: **-0.111** points (negative is better)
- 95% paired bootstrap interval: **[-0.267, +0.042]**

## Decision-support diagnostics

- Status disagreements: **23**
- Qualifier disagreements: **11**
- Scorable status migrations: **23**
- Status-migration accuracy: **0.609**

## Qualifier economics (supporting evidence)

- Baseline: 4-6-0, ROI -0.236, avg CLV +0.250
- Challenger: 3-8-0, ROI -0.479, avg CLV +0.583

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, weekly splits, and individual disagreement games are secondary evidence. No result here changes 2026 production automatically; any 2027 consideration requires a separate versioned decision after the predeclared reviews.