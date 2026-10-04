# 2026 Baseline vs Orientation Shadow Evaluation

**Research only. This report cannot alter the operational weekly board or official prospective ledger.**

Protocol: `orientation-eval-2026.3`  
Challenger: `orientation-crosswind-hgb-v0.1`  
Graded paired orientation-ready games: **242**

## Primary paired model metric

- Baseline MAE: **12.198** points
- Challenger MAE: **12.173** points
- Challenger minus baseline MAE: **-0.025** points (negative is better)
- 95% paired bootstrap interval: **[-0.177, +0.129]**

## Decision-support diagnostics

- Status disagreements: **11**
- Qualifier disagreements: **6**
- Scorable status migrations: **11**
- Status-migration accuracy: **0.364**

## Qualifier economics (supporting evidence)

- Baseline: 2-1-0, ROI +0.273, avg CLV +nan
- Challenger: 0-5-0, ROI -1.000, avg CLV +0.500

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, and individual disagreement games are secondary evidence and cannot alone justify a model promotion. Formal promotion decisions occur only at the predeclared review points and require a separate versioned decision.