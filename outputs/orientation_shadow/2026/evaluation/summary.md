# 2026 Baseline vs Orientation Shadow Evaluation

**Research only. This report cannot alter the operational weekly board or official prospective ledger.**

Protocol: `orientation-eval-2026.3`  
Challenger: `orientation-crosswind-hgb-v0.1`  
Graded paired orientation-ready games: **624**

## Primary paired model metric

- Baseline MAE: **12.545** points
- Challenger MAE: **12.370** points
- Challenger minus baseline MAE: **-0.175** points (negative is better)
- 95% paired bootstrap interval: **[-0.296, -0.049]**

## Decision-support diagnostics

- Status disagreements: **36**
- Qualifier disagreements: **18**
- Scorable status migrations: **36**
- Status-migration accuracy: **0.500**

## Qualifier economics (supporting evidence)

- Baseline: 4-7-0, ROI -0.306, avg CLV +0.250
- Challenger: 6-17-0, ROI -0.502, avg CLV +0.636

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, and individual disagreement games are secondary evidence and cannot alone justify a model promotion. Formal promotion decisions occur only at the predeclared review points and require a separate versioned decision.