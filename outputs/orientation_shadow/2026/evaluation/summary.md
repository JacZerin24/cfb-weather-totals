# 2026 Baseline vs Orientation Shadow Evaluation

**Research only. This report cannot alter the operational weekly board or official prospective ledger.**

Protocol: `orientation-eval-2026.3`  
Challenger: `orientation-crosswind-hgb-v0.1`  
Graded paired orientation-ready games: **501**

## Primary paired model metric

- Baseline MAE: **12.659** points
- Challenger MAE: **12.515** points
- Challenger minus baseline MAE: **-0.144** points (negative is better)
- 95% paired bootstrap interval: **[-0.273, -0.018]**

## Decision-support diagnostics

- Status disagreements: **25**
- Qualifier disagreements: **12**
- Scorable status migrations: **25**
- Status-migration accuracy: **0.480**

## Qualifier economics (supporting evidence)

- Baseline: 4-6-0, ROI -0.236, avg CLV +0.250
- Challenger: 4-12-0, ROI -0.523, avg CLV +0.625

## Interpretation guardrail

The paired MAE comparison is primary. ROI, hit rate, CLV, and individual disagreement games are secondary evidence and cannot alone justify a model promotion. Formal promotion decisions occur only at the predeclared review points and require a separate versioned decision.