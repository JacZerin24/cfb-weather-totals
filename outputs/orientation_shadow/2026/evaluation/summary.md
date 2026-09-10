# 2026 Baseline vs Orientation Shadow Evaluation

**Research only. This report cannot alter the operational weekly board or official prospective ledger.**

Protocol: `orientation-eval-2026.3`  
Challenger: `orientation-crosswind-hgb-v0.1`  
Graded paired orientation-ready games: **37**

## Primary paired model metric

- Baseline MAE: **11.850** points
- Challenger MAE: **11.795** points
- Challenger minus baseline MAE: **-0.055** points (negative is better)
- 95% paired bootstrap interval: **[-0.191, +0.064]**

## Decision-support diagnostics

- Status disagreements: **0**
- Qualifier disagreements: **0**
- Scorable status migrations: **0**
- Status-migration accuracy: **nan**

## Qualifier economics (supporting evidence)

- Baseline: 0-0-0, ROI +nan, avg CLV +nan
- Challenger: 0-0-0, ROI +nan, avg CLV +nan

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, and individual disagreement games are secondary evidence and cannot alone justify a model promotion. Formal promotion decisions occur only at the predeclared review points and require a separate versioned decision.