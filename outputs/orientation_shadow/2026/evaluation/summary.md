# 2026 Baseline vs Orientation Shadow Evaluation

**Research only. This report cannot alter the operational weekly board or official prospective ledger.**

Protocol: `orientation-eval-2026.3`  
Challenger: `orientation-crosswind-hgb-v0.1`  
Graded paired orientation-ready games: **189**

## Primary paired model metric

- Baseline MAE: **11.980** points
- Challenger MAE: **11.995** points
- Challenger minus baseline MAE: **+0.015** points (negative is better)
- 95% paired bootstrap interval: **[-0.120, +0.151]**

## Decision-support diagnostics

- Status disagreements: **5**
- Qualifier disagreements: **4**
- Scorable status migrations: **5**
- Status-migration accuracy: **0.000**

## Qualifier economics (supporting evidence)

- Baseline: 0-0-0, ROI +nan, avg CLV +nan
- Challenger: 0-4-0, ROI -1.000, avg CLV +0.500

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, and individual disagreement games are secondary evidence and cannot alone justify a model promotion. Formal promotion decisions occur only at the predeclared review points and require a separate versioned decision.