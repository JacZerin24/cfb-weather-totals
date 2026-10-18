# 2026 Baseline vs Orientation Shadow Evaluation

**Research only. This report cannot alter the operational weekly board or official prospective ledger.**

Protocol: `orientation-eval-2026.3`  
Challenger: `orientation-crosswind-hgb-v0.1`  
Graded paired orientation-ready games: **344**

## Primary paired model metric

- Baseline MAE: **12.310** points
- Challenger MAE: **12.209** points
- Challenger minus baseline MAE: **-0.101** points (negative is better)
- 95% paired bootstrap interval: **[-0.248, +0.049]**

## Decision-support diagnostics

- Status disagreements: **21**
- Qualifier disagreements: **9**
- Scorable status migrations: **21**
- Status-migration accuracy: **0.429**

## Qualifier economics (supporting evidence)

- Baseline: 3-2-0, ROI +0.145, avg CLV +0.750
- Challenger: 2-8-0, ROI -0.618, avg CLV +0.650

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, and individual disagreement games are secondary evidence and cannot alone justify a model promotion. Formal promotion decisions occur only at the predeclared review points and require a separate versioned decision.