# NFL Model Tournament Phase 3 — Preregistered Protocol

**Branch:** `research/nfl-model-tournament-v3`  
**Parent research:** NFL Team-Context Phases 1-2  
**Protected baseline:** frozen `nfl_totals_paper_v1`  
**Goal:** test whether a substantially broader set of statistical, machine-learning, nonlinear, dynamic, and ensemble architectures can improve on the frozen v1 model using only information available before each game.

## Non-negotiable isolation

This phase is research-only.

It must not:
- change the frozen v1 protocol;
- change the live NFL board scoring;
- change the official prospective v1 ledger;
- change the current QUALIFIES or STRONG thresholds.

Any winner remains a challenger until separately frozen and prospectively tracked.

## Important holdout limitation

The 2018-2025 outcomes have already been inspected in earlier research. They are therefore not a pristine human-blind holdout.

To reduce researcher degrees of freedom:
1. this document fixes the tournament before Phase 3 results are generated;
2. model/feature selection is performed algorithmically inside chronological training data;
3. every outer test season is predicted only with information and selections available before that season;
4. the strongest evidence for promotion must ultimately be prospective 2026 performance.

## Target and market benchmark

Primary regression target:

`market_residual = actual_total_points - market_total`

Primary classification target:

`went_over = market_residual > 0`

The market-alone regression benchmark predicts residual = 0.

The frozen v1 model remains the betting benchmark.

## Data

### Market/game/weather
- existing forecast-native dataset;
- archived JMA GSM 24-hour weather forecasts;
- market total and game-context fields already used by v1.

### Conventional football context
Pregame team metrics already built in Phase 1:
- EPA/play;
- success rate;
- pass EPA;
- rush EPA;
- explosive-play rate;
- pass rate;
- sack rate;
- turnover rate;
- early-down EPA;
- corresponding defensive allowed metrics;
- matchup summaries.

All are based only on previously completed games.

### Dynamic football context
Phase 3 adds sequential state features built from the 2006-2025 game history, with no future information:
- exponentially weighted points scored and allowed;
- exponentially weighted game market residual;
- exponentially weighted scoring volatility;
- short, medium, and long memory spans;
- two fixed latent offense/defense state filters with different learning rates;
- direct dynamic projected total;
- direct dynamic market edge;
- home/away latent offense and defense states;
- agreement/disagreement between short- and long-memory states.

No current-game score, closing outcome, or future team information may enter a pregame row.

## Fixed feature sets

Four fixed feature sets are allowed:

1. **v1_weather**
   - frozen v1 market/context + 24h forecast-weather inputs.

2. **v1_plus_team**
   - v1_weather + all Phase-1 conventional football-context features.

3. **v1_plus_dynamic**
   - v1_weather + all Phase-3 dynamic features.

4. **all_context**
   - v1_weather + conventional football context + dynamic features.

No feature set may be added after seeing Phase 3 tournament results.

## Regression model families

The following regression families are fixed before results:

1. market-zero benchmark;
2. ordinary linear regression;
3. Ridge;
4. Elastic Net;
5. Bayesian Ridge;
6. Huber robust regression;
7. spline basis + Ridge (GAM-like);
8. Random Forest;
9. Extra Trees;
10. HistGradientBoosting;
11. GradientBoosting;
12. AdaBoost;
13. RBF SVR;
14. MLP neural network;
15. XGBoost;
16. LightGBM;
17. CatBoost;
18. direct dynamic latent-state projection.

Where a family exposes fixed variants below, the tournament may compare those variants only. No free-form hyperparameter search is permitted.

## Classification model families

The following probability models are fixed:

1. empirical base-rate benchmark;
2. logistic regression;
3. spline basis + logistic regression;
4. Random Forest;
5. Extra Trees;
6. HistGradientBoosting;
7. GradientBoosting;
8. RBF SVC with probability calibration enabled;
9. MLP neural network;
10. XGBoost classifier;
11. LightGBM classifier;
12. CatBoost classifier.

## Fixed model variants

To permit meaningful complexity without an unrestricted search:

- Ridge alpha: 10, 50
- Elastic Net alpha: 0.02, 0.08; l1_ratio fixed 0.15
- Huber epsilon: 1.35
- spline knots: 4; degree 3
- RF min leaf: 15, 25, 50
- Extra Trees min leaf: 15, 35
- HistGB min leaf: 20, 35, 60
- GradientBoosting depth: 2, 3
- AdaBoost estimators: 100
- RBF SVR C: 0.5, 2.0; gamma='scale'
- MLP: hidden layers (32,) and (32,16), alpha 0.01
- XGBoost: max_depth 2 or 3, learning_rate 0.03, n_estimators 250
- LightGBM: num_leaves 7 or 15, learning_rate 0.03, n_estimators 250
- CatBoost: depth 4 or 6, learning_rate 0.03, iterations 250

The same general complexity variants apply to classifiers where relevant.

## Chronological nested evaluation

Primary horizon: 24 hours.

Outer test seasons:
- 2021
- 2022
- 2023
- 2024
- 2025

For each outer season:
- training data contains only earlier seasons;
- an inner chronological evaluation among earlier seasons selects model/feature combinations;
- the outer season is not used for selection.

### Inner selection metric

Regression candidates:
- primary: mean inner-fold residual MAE;
- tie-breaker: median inner-fold MAE;
- second tie-breaker: lower model complexity according to a fixed complexity rank.

Classification candidates:
- primary: mean inner-fold Brier score;
- tie-breaker: mean log loss;
- second tie-breaker: lower complexity rank.

## Ensemble / stacking systems

Three ensemble challengers are fixed:

### selected_pair
For each outer season:
- use the inner-selected best regression candidate;
- use the inner-selected best classification candidate;
- apply frozen QUALIFIES / STRONG thresholds.

### diversified_average
For each outer season:
- select the best candidate from each of four broad regression families: linear/spline, tree bagging, boosting, nonlinear/neural;
- average available regression predictions;
- do the analogous family-level average for classification probabilities.

### stacked_meta
For each outer season:
- generate inner out-of-fold predictions using a fixed compact base library:
  - Ridge
  - spline-Ridge
  - Random Forest
  - Extra Trees
  - HistGradientBoosting
  - XGBoost
  - LightGBM
  - CatBoost
  - MLP
- fit a Ridge meta-regressor on OOF regression predictions;
- fit logistic regression on OOF classifier probabilities;
- refit base models on all outer-training data;
- apply the frozen meta-model to the outer test season.

If fewer than four complete inner folds exist, stacked_meta is skipped for that outer season rather than weakening the rule.

## Dynamic standalone system

The direct dynamic latent-state projection is also scored as its own regression signal.

For probability:
- estimate the standard deviation of historical dynamic residual errors using training data only;
- map the predicted residual to an Over probability using a zero-centered normal error assumption.

This is a deliberately simple probabilistic wrapper and must be labeled as such.

## Betting rules

No threshold tuning is allowed.

- QUALIFIES: predicted residual >= +3.0 AND P(Over) >= 0.60
- STRONG: predicted residual >= +4.0 AND P(Over) >= 0.60

Primary direction remains OVER to match the frozen v1 protocol.

## Metrics

Every outer-season system must report:

Prediction quality:
- residual MAE;
- RMSE;
- Brier score;
- log loss;
- calibration slope/intercept where estimable;
- mean predicted residual.

Betting:
- plays;
- wins/losses/pushes;
- hit rate;
- Wilson 95% interval;
- flat -110 ROI;
- maximum drawdown;
- season-by-season record.

Comparison:
- paired MAE/Brier differences versus v1;
- paired bootstrap 95% intervals;
- number of outer seasons improved;
- qualifier overlap and disagreement with v1.

## 2025 true decision-time test

After the nested forecast-native tournament is complete, all frozen tournament systems that can be reproduced at decision time are rerun using:
- archived 24-hour sportsbook total;
- archived 24-hour Over price;
- archived 24-hour JMA forecast;
- only pregame team/dynamic information.

Report:
- consensus-price ROI;
- best same-line price ROI;
- line/price availability;
- prediction quality versus the actual decision-time line.

## Multiple-comparison guardrail

The tournament is exploratory across many architectures. Therefore:

- no individual historical ROI row is sufficient for promotion;
- prediction-quality comparisons receive more weight than isolated betting ROI;
- Benjamini-Hochberg FDR-adjusted p-values are reported for the fixed-rule betting comparisons where applicable;
- a model that wins only one season is not considered robust.

## Advancement rule

A challenger may be labeled **Phase 3 prospective candidate** only if all of the following hold:

1. aggregate 2021-2025 MAE is no worse than v1 and Brier is no worse than v1;
2. at least 3 of 5 outer seasons improve either MAE or Brier without materially degrading the other;
3. QUALIFIES has at least 25 graded plays across 2021-2025;
4. QUALIFIES flat -110 ROI is positive;
5. no single season contributes more than 60% of total betting profit;
6. 2025 decision-time results are directionally consistent with the forecast-native result;
7. paired bootstrap probability of MAE improvement or Brier improvement is at least 0.80.

If no model passes, v1 remains the supported choice.

Even if one passes, it does not replace v1 without prospective 2026 tracking.
