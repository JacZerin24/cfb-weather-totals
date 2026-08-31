# Stadium Wind Orientation — Final Bounded Play-by-Play Mechanism Study

**Date: 2026-08-31**  
**Status: RESEARCH ONLY — NO CHANGE TO THE WORKING MODEL, QUALIFIER/LEAN CLASSIFICATIONS, OR OFFICIAL 2026 PROSPECTIVE LEDGER.**

## Why this study was run

The earlier box-score mechanism study did not support a simple explanation in which cross-field wind lowers scoring mainly by degrading passing accuracy or passing efficiency. It did, however, leave aggregate kicking points as a potentially interesting channel that could not be separated cleanly into field goals versus extra points.

Before pulling play-by-play, this final historical extension was frozen to answer only two bounded questions:

1. Does cross-aligned wind reduce completed explosive passes of at least 20 yards per pass attempt?
2. Does cross-aligned wind reduce field-goal make probability after controlling for attempt distance?

No new betting thresholds, total ranges, or wind cutoffs were searched.

## Locked sample

The sample remained exactly the historical regime established before this study:

- FBS vs FBS
- outdoor games only
- raw wind >10 and <=15 mph
- cross alignment = 60–90 degrees from the field axis
- parallel alignment = 0–30 degrees
- 2014–2025 primary historical period
- 2021–2025 secondary recent-era check

Controls remain aligned with the previous mechanism work: raw wind speed, temperature, humidity, precipitation, dew point, pressure, closing total, neutral-site status, venue fixed effects, season fixed effects, and line-provider fixed effects. Field-goal models additionally control for attempt distance and distance squared.

## Data coverage and QA

The locked regime contained **1,232 target games**. CFBD play-by-play was available for **1,228 games (99.7%)**, producing **223,879 play rows**.

The predeclared parser identified:

- **80,061 pass attempts**
- **7,880 completed passes gaining at least 20 yards**
- **3,780 field-goal-related play rows**
- field-goal distance for **100%** of identified rows
- field-goal distance directly from play text for **99.63%** of rows

QA inspection showed two parser details worth testing explicitly:

- generic `Penalty` play rows can contain pass or field-goal descriptions even when the underlying play is nullified or otherwise ambiguous;
- a blocked field goal returned for a touchdown can have `scoring=true`, which must not be interpreted as a made field goal.

The original predeclared results were preserved. A separate parser-integrity sensitivity excluded generic `Penalty` rows and counted only explicit `Field Goal Good` play types as made field goals. This is a data-quality sensitivity, not a new hypothesis or threshold search.

For pass attempts, excluding generic Penalty rows also materially improved agreement with independent box-score pass-attempt totals: mean absolute game-level discrepancy fell from about **2.52 attempts** to about **0.69 attempts**, with the median discrepancy falling from **2 attempts to 0**. The substantive explosive-pass result did not change.

## Result 1 — Explosive passing mechanism

### Predeclared analysis

| Period | Games | Adjusted cross minus parallel explosive-pass rate | p-value |
|---|---:|---:|---:|
| 2014–2025 | 1,205 | +0.0009 | 0.764 |
| 2021–2025 | 480 | +0.0040 | 0.507 |

Cross-aligned wind does **not** reduce completed passes of at least 20 yards per pass attempt. The estimated effects are essentially zero and, if anything, slightly positive.

### Parser-integrity sensitivity

After excluding generic Penalty rows:

| Period | Adjusted cross minus parallel explosive-pass rate | p-value |
|---|---:|---:|
| 2014–2025 | +0.0009 | 0.770 |
| 2021–2025 | +0.0041 | 0.502 |

The result is unchanged.

### Passing verdict

**Explosive-passing mechanism supported: NO.**

Combined with the earlier null results for passing yards/attempt and completion percentage, there is now little evidence that the historical orientation signal is primarily a passing-efficiency or explosive-passing effect.

## Result 2 — Field-goal mechanism

### Predeclared analysis

After controlling for kick distance and the common game/venue/weather/market covariates:

| Period | FG attempts | Cross make rate | Parallel make rate | Adjusted cross minus parallel make probability | p-value |
|---|---:|---:|---:|---:|---:|
| 2014–2025 | 3,698 | 72.36% | 75.34% | **-4.35 percentage points** | **0.0078** |
| 2021–2025 | 1,478 | 73.09% | 77.94% | **-5.96 percentage points** | **0.0346** |

### Parser-integrity sensitivity

After excluding generic Penalty rows and requiring explicit `Field Goal Good` for a made kick:

| Period | FG attempts | Cross make rate | Parallel make rate | Adjusted cross minus parallel make probability | p-value |
|---|---:|---:|---:|---:|---:|
| 2014–2025 | 3,676 | 72.88% | 75.63% | **-4.08 percentage points** | **0.0129** |
| 2021–2025 | 1,472 | 73.92% | 78.15% | **-5.25 percentage points** | 0.0610 |

The full-era result survives the stricter parser treatment. The recent-era estimate remains similar in magnitude but its uncertainty is larger, so it should be described as supportive/suggestive rather than independently decisive.

### Predeclared descriptive distance bins after parser cleanup

| Distance | Cross | Parallel | Difference |
|---|---:|---:|---:|
| <40 yards | 82.08% (1,077 attempts) | 83.37% (1,317) | -1.29 pp |
| 40–49 yards | 60.98% (492) | 63.60% (577) | -2.63 pp |
| 50+ yards | 43.85% (130) | 54.27% (164) | -10.42 pp |

These bins were predeclared as descriptive checks, not separate model rules. The larger raw difference on longer kicks is physically plausible, but the sample is much smaller and no new distance-specific production threshold should be created from it.

### Kicking verdict

**Field-goal mechanism supported: YES, with moderate confidence rather than proof.**

The strongest mechanism evidence found in the entire orientation research sequence is now a lower distance-adjusted field-goal make probability under cross-aligned wind. This survives the full-era parser-integrity sensitivity and has a similar, though less certain, magnitude in 2021–2025.

The magnitude is meaningful but does not explain the entire historical scoring-residual signal by itself. With roughly 3.2 field-goal attempts per game in the controlled sample, a 4–5 percentage-point make-probability difference corresponds mechanically to only about **0.4–0.5 expected scoring points per game**, before accounting for behavioral changes such as coaches declining longer attempts. The broader market-residual signal therefore still cannot be attributed to field goals alone.

## Overall mechanism conclusion

### Simple passing mechanism: NOT SUPPORTED
### Explosive-passing mechanism: NOT SUPPORTED
### Field-goal mechanism: SUPPORTED
### Complete causal explanation for the orientation signal: NOT ESTABLISHED
### Production model change justified from historical mechanism evidence alone: NO

This is a better scientific position than either extreme. The stadium-orientation hypothesis is no longer merely an unexplained historical betting pattern: there is now a football outcome with a direct aerodynamic connection—field-goal success—that moves in the expected direction after distance, venue, weather, market, season, and provider controls.

At the same time, the evidence does not show a general degradation of passing, the recent-era kicking sensitivity is not independently decisive, and the field-goal effect can explain only part of the observed scoring difference. That is not sufficient to promote crosswind into production based on historical data alone.

## Stopping rule — historical research is now closed

This study completes the predeclared final historical mechanism extension.

**Do not open another stadium-orientation historical threshold-search or mechanism-mining cycle.**

The next major evidence must come from the frozen prospective challenger:

`orientation-crosswind-hgb-v0.1`

It will be evaluated under the separately frozen `orientation-eval-2026.1` protocol using paired projected-total MAE as the primary metric, with status migrations, qualifier economics, and CLV as supporting evidence.

Formal reviews are scheduled after Week 6, after Week 10, and after the 2026 regular season. Any promotion must be a separately documented versioned decision and may apply only prospectively from its declared effective time.
