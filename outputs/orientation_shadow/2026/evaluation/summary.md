# 2026 Baseline vs Orientation Shadow Evaluation

**Research only. This report cannot alter the operational weekly board or official prospective ledger.**

Protocol: `orientation-eval-2026.3`  
Challenger: `orientation-crosswind-hgb-v0.1`  
Graded paired orientation-ready games: **559**

## Primary paired model metric

- Baseline MAE: **12.723** points
- Challenger MAE: **12.538** points
- Challenger minus baseline MAE: **-0.185** points (negative is better)
- 95% paired bootstrap interval: **[-0.313, -0.052]**

## Decision-support diagnostics

- Status disagreements: **30**
- Qualifier disagreements: **15**
- Scorable status migrations: **30**
- Status-migration accuracy: **0.500**

## Qualifier economics (supporting evidence)

- Baseline: 4-6-0, ROI -0.236, avg CLV +0.250
- Challenger: 5-14-0, ROI -0.498, avg CLV +0.575

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, and individual disagreement games are secondary evidence and cannot alone justify a model promotion. Formal promotion decisions occur only at the predeclared review points and require a separate versioned decision.