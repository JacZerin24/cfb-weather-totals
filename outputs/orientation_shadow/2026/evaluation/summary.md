# 2026 Baseline vs Orientation Shadow Evaluation

**Research only. This report cannot alter the operational weekly board or official prospective ledger.**

Protocol: `orientation-eval-2026.3`  
Challenger: `orientation-crosswind-hgb-v0.1`  
Graded paired orientation-ready games: **394**

## Primary paired model metric

- Baseline MAE: **12.633** points
- Challenger MAE: **12.531** points
- Challenger minus baseline MAE: **-0.102** points (negative is better)
- 95% paired bootstrap interval: **[-0.242, +0.040]**

## Decision-support diagnostics

- Status disagreements: **23**
- Qualifier disagreements: **11**
- Scorable status migrations: **23**
- Status-migration accuracy: **0.435**

## Qualifier economics (supporting evidence)

- Baseline: 3-4-0, ROI -0.182, avg CLV +0.750
- Challenger: 2-10-0, ROI -0.682, avg CLV +0.875

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, and individual disagreement games are secondary evidence and cannot alone justify a model promotion. Formal promotion decisions occur only at the predeclared review points and require a separate versioned decision.