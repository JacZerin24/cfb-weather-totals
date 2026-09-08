# Wind Gust Incremental-Value Research Findings

## Status

**Completed retrospective study. Result: NOT PROVEN / STOP.**

Historical kickoff-aligned ECMWF IFS gust information did **not** demonstrate incremental out-of-sample predictive value beyond same-source sustained wind for the GENERAL HistGradientBoosting totals model. Per the preregistered protocol, gust features should **not** be promoted into production, the live weekly model, picks, shadows, or prospective ledgers on the basis of this study.

This finding applies to the retrospective kickoff-reanalysis experiment described in `documentation/WIND_GUST_RESEARCH_PROTOCOL.md`.

## Reproducibility

- Research branch: `research/wind-gust-value`
- Frozen population: outdoor FBS-vs-FBS games, 2017–2025
- Complete IFS acquisition: **6,255 / 6,255 games (100%)**
- Unique paired venues: **167 / 167**
- Walk-forward scored games: **4,833**
- Test seasons: **7**
- Final acquisition recovery run: `34208786310`
- Frozen gust-value research run: `34208957124`
- Research workflow commit used for the model run: `396f3244852c4f6e3e631ca8f146bd322c3d14f9`

The completed recovery artifact independently passed the acquisition gate before modeling: 100% complete coverage, no missing venue coordinates, no final recovery request failures, no duplicate game/source rows, only the `ecmwf_ifs` source stack, no future-valid weather timestamps, and no wind or gust snapshots older than 120 minutes at kickoff.

The model integrity suite also passed all 8 preregistered tests before the real model ladder ran.

## Primary model results

All variants were scored on the exact same 4,833 walk-forward test games. Positive MAE delta means the challenger was worse than its reference; negative means better.

| Challenger | Reference | MAE delta | Game bootstrap 95% CI | Season-cluster 95% CI | Seasons improved | Status |
|---|---|---:|---:|---:|---:|---|
| IFS sustained control | Baseline | -0.0097 | [-0.0807, +0.0603] | [-0.0497, +0.0377] | 4 / 7 | NOT_PROVEN |
| Gust magnitude | IFS sustained control | +0.0145 | [-0.0603, +0.0858] | [-0.0228, +0.0500] | 3 / 7 | NOT_PROVEN |
| Gust spread | IFS sustained control | +0.0311 | [-0.0500, +0.1098] | [-0.0730, +0.1216] | 3 / 7 | NOT_PROVEN |
| Gust core | IFS sustained control | **+0.0899** | **[+0.0086, +0.1696]** | **[+0.0082, +0.1780]** | **1 / 7** | **NOT_PROVEN** |
| Gust interactions | Gust core | -0.0584 | [-0.1359, +0.0173] | [-0.1328, -0.0009] | 4 / 7 | NOT_PROVEN |

The preregistered incremental-value gate required all of the following for a gust challenger versus the same-source sustained-wind control:

1. Mean paired MAE delta below zero.
2. Game-level bootstrap 95% CI entirely below zero.
3. Season-cluster 95% CI entirely below zero.
4. Improvement in at least 70% of test seasons, which requires 5 of 7 seasons here.

No primary gust challenger satisfied the gate.

## Absolute model performance

| Model | MAE | Delta vs baseline | RMSE | Signed projection bias |
|---|---:|---:|---:|---:|
| Baseline | 13.0356 | 0.0000 | 16.4571 | -0.2358 |
| IFS sustained control | 13.0260 | -0.0097 | 16.4441 | -0.0957 |
| Gust magnitude | 13.0404 | +0.0048 | 16.4301 | -0.3412 |
| Gust spread | 13.0571 | +0.0215 | 16.4578 | -0.3128 |
| Gust core | 13.1159 | +0.0802 | 16.5102 | -0.2624 |
| Gust interactions | 13.0575 | +0.0219 | 16.4402 | -0.0209 |

## Interpretation

### 1. Same-source sustained wind itself was not proven better

Adding kickoff-aligned IFS sustained wind to the existing GENERAL HGB feature set improved MAE by only about 0.010 points. Both uncertainty intervals crossed zero and only 4 of 7 test seasons improved. This is too small and unstable to claim that simply replacing or supplementing the existing wind information with IFS sustained wind improves the model.

### 2. Gust magnitude did not add value beyond sustained wind

Adding IFS gust magnitude to the IFS sustained-wind control made MAE slightly worse, by about 0.014 points, and improved only 3 of 7 seasons. The result does not support gust magnitude as an incremental predictor.

### 3. Gust spread did not add value beyond sustained wind

Adding gust spread also worsened MAE, by about 0.031 points, and improved only 3 of 7 seasons. The uncertainty intervals were broad and crossed zero.

### 4. The combined gust core produced the clearest negative result

The combined gust magnitude + gust spread model worsened MAE by about **0.090 points** versus the same-source sustained-wind control. It improved only 1 of 7 seasons, and both the game-level and season-cluster 95% confidence intervals were entirely above zero.

Within this frozen experiment, that is evidence that the combined gust core was not merely unhelpful but was **reliably harmful to MAE** relative to using IFS sustained wind without the gust features.

### 5. Explicit interactions did not rescue the gust hypothesis

The preregistered interaction model recovered some of the loss created by `gust_core`, but its game-level confidence interval still crossed zero and it improved only 4 of 7 seasons versus `gust_core`. More importantly, its absolute MAE (13.0575) remained worse than both the baseline (13.0356) and IFS sustained control (13.0260).

Therefore the interaction result is not evidence that gusts add deployable value.

## Decision

**STOP. Do not add historical gust magnitude, gust spread, gust core, or the preregistered gust interactions to the production model.**

The protocol explicitly stated that if magnitude, spread, or core did not robustly beat the same-source sustained-wind control, the research should stop rather than searching additional thresholds, subgroups, or betting-return slices for a favorable result. That stopping rule is now satisfied.

Accordingly:

- No gust feature should be merged into `main`.
- No live weekly model or pick logic should be changed.
- No gust-based threshold should be selected from retrospective ROI or subgroup performance.
- No Stage 2 archived-forecast experiment is justified solely by these results.
- No prospective gust shadow should be started solely by these results.

The existing production model remains the correct default pending separate evidence from a future, independently motivated study.

## Research caveat

The weather source in this experiment is historical IFS/reanalysis-style archived data aligned to kickoff. Even a positive result here would not by itself establish deployable pregame value because the operational use case would require genuinely available pregame forecast data at a fixed lead time. Since the primary retrospective incremental-value gate failed, that additional deployment-validation stage was not reached.

## Final conclusion

For this population, model family, target, walk-forward design, and frozen feature ladder, **wind gust information does not add demonstrated predictive value beyond sustained wind**. The strongest combined gust specification actually degraded MAE with confidence intervals indicating a reproducible deterioration. The safest evidence-based action is to leave the live model unchanged.
