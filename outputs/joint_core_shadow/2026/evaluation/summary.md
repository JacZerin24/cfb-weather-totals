# 2026 Baseline vs Joint-Core Shadow Evaluation

**Research only. This report cannot alter the operational weekly board, weekly picks, thresholds, or official prospective ledger.**

Protocol: `joint-core-eval-2026.2`  
Challenger: `joint-core-weather-context-hgb-v0.1`  
Prospective signal: **PROSPECTIVE_MAE_FAVORABLE_UNCERTAIN**  
Graded paired joint-core-ready games: **458**

## Primary paired model metric

- Baseline MAE: **12.726** points
- Challenger MAE: **12.640** points
- Challenger minus baseline MAE: **-0.086** points (negative is better)
- 95% paired bootstrap interval: **[-0.249, +0.076]**

## Decision-support diagnostics

- Status disagreements: **20**
- Qualifier disagreements: **10**
- Scorable status migrations: **20**
- Status-migration accuracy: **0.550**

## Qualifier economics (supporting evidence)

- Baseline: 4-6-0, ROI -0.236, avg CLV +0.250
- Challenger: 2-8-0, ROI -0.618, avg CLV +0.583

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, weekly splits, and individual disagreement games are secondary evidence. No result here changes 2026 production automatically; any 2027 consideration requires a separate versioned decision after the predeclared reviews.