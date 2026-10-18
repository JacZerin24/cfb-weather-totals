# 2026 Baseline vs Joint-Core Shadow Evaluation

**Research only. This report cannot alter the operational weekly board, weekly picks, thresholds, or official prospective ledger.**

Protocol: `joint-core-eval-2026.2`  
Challenger: `joint-core-weather-context-hgb-v0.1`  
Prospective signal: **PROSPECTIVE_MAE_FAVORABLE_UNCERTAIN**  
Graded paired joint-core-ready games: **307**

## Primary paired model metric

- Baseline MAE: **12.365** points
- Challenger MAE: **12.316** points
- Challenger minus baseline MAE: **-0.049** points (negative is better)
- 95% paired bootstrap interval: **[-0.230, +0.135]**

## Decision-support diagnostics

- Status disagreements: **14**
- Qualifier disagreements: **6**
- Scorable status migrations: **14**
- Status-migration accuracy: **0.429**

## Qualifier economics (supporting evidence)

- Baseline: 3-2-0, ROI +0.145, avg CLV +0.750
- Challenger: 1-6-0, ROI -0.727, avg CLV +0.583

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, weekly splits, and individual disagreement games are secondary evidence. No result here changes 2026 production automatically; any 2027 consideration requires a separate versioned decision after the predeclared reviews.