# Archived Pregame Forecast Realism Findings

## Decision summary

The archived pregame forecast realism study is complete through the preregistered retrospective model stage.

**Decision: no production model change is justified by this study.**

The study found that archived National Blend of Models (NBM/NBS) weather forecasts become measurably more accurate as kickoff approaches, but the improvement from 24 hours to 12 hours to 6 hours does **not** translate into a material or statistically reliable improvement in GENERAL HGB total-prediction MAE.

The current production-style approach -- training on the long retrospective historical dataset and scoring with forecast weather -- shows a small aggregate MAE penalty relative to scoring the same model with retrospective weather, but the preregistered material-degradation gate does not pass. Retraining exclusively on the shorter 2019-2025 archived-forecast sample performs materially worse than the current long-history production-style training approach.

No live board, pick logic, prospective ledger, production configuration, or `main` branch behavior was changed.

## Frozen design

Primary population:

- regular-season games
- 2019-2025
- outdoor FBS-vs-FBS
- closing total and final score available
- valid kickoff and venue coordinates
- archived NBM station within 30 miles

Decision times were frozen at exactly:

- 24 hours before kickoff
- 12 hours before kickoff
- 6 hours before kickoff

A conservative NBM availability rule was frozen before model results were viewed:

`eligible availability time = archived NBM runtime + 2 hours`

The forecast was eligible only when that availability time was at or before the decision cutoff. Kickoff forecast valid time had to be within +/-90 minutes of kickoff.

Model comparisons used chronological walk-forward testing with prior seasons only. No random split was used.

## Source feasibility and coverage

The preferred source was Iowa State IEM's archive of NBM short-range text guidance (`NBS`). Source smoke testing succeeded across geographically separated cases from 2019 through 2025.

The initial station audit contained 4,833 otherwise eligible games. Of those:

- 4,829 had an eligible archived-guidance station within 30 miles
- station-mapping coverage = 99.92%
- median station distance = 4.16 miles
- 90th-percentile station distance = 8.92 miles

Only four games failed the frozen spatial gate.

### Full acquisition

GitHub Actions acquisition run: `34213947929`

Final source-QC result:

- eligible station-mapped games: 4,829
- complete 24h/12h/6h games: 4,829 / 4,829
- complete lead rows: 14,487
- overall coverage: 100%
- minimum single-season coverage: 100%
- core continuous forecast-field completeness: 100%
- availability-time leakage violations: 0
- kickoff valid-time offset violations: 0
- station-distance violations: 0
- duplicate game/lead rows: 0
- `ready_for_modeling=True`

There were 108 failed individual request attempts during acquisition, but candidate-station fallback/retry logic recovered all affected games; no final game was missing.

## Full-sample forecast realism

Forecast-versus-retrospective verification used 4,738 paired games for most continuous fields because some retrospective weather values were unavailable even though the archived forecast reconstruction itself was complete.

### Temperature

| Lead | Bias | MAE | Correlation |
| --- | ---: | ---: | ---: |
| 24h | -0.92 F | 3.12 F | 0.966 |
| 12h | -0.90 F | 3.00 F | 0.968 |
| 6h | -0.89 F | 2.95 F | 0.969 |

### Dew point

| Lead | Bias | MAE | Correlation |
| --- | ---: | ---: | ---: |
| 24h | +0.34 F | 2.57 F | 0.974 |
| 12h | +0.37 F | 2.42 F | 0.976 |
| 6h | +0.41 F | 2.38 F | 0.977 |

### Relative humidity

| Lead | Bias | MAE | Correlation |
| --- | ---: | ---: | ---: |
| 24h | +1.49 pp | 8.28 pp | 0.853 |
| 12h | +1.59 pp | 7.93 pp | 0.865 |
| 6h | +1.69 pp | 7.81 pp | 0.869 |

### Sustained wind

| Lead | Bias | MAE | Correlation |
| --- | ---: | ---: | ---: |
| 24h | -0.18 mph | 2.60 mph | 0.713 |
| 12h | -0.20 mph | 2.57 mph | 0.725 |
| 6h | -0.20 mph | 2.54 mph | 0.729 |

Circular wind-direction MAE improved from 33.0 degrees at 24h to 31.1 degrees at 12h and 30.9 degrees at 6h.

Precipitation-probability Brier score improved only modestly:

- 24h: 0.05895
- 12h: 0.05810
- 6h: 0.05629

The overall physical result is internally sensible: later NBM guidance is generally closer to the retrospective weather state, but the improvement from 24h to 6h is modest for the main deterministic variables.

These discrepancies are interpreted as **operational input differences**, not pure atmospheric forecast error, because NBM guidance is station based while the retrospective weather field represents the project's historical stadium-weather source.

## GENERAL HGB realism experiment

GitHub Actions model run: `34215779573`

The complete common model sample contained 4,829 games across seven test seasons (2019-2025).

### Variant MAE

| Variant | Games | Seasons | MAE |
| --- | ---: | ---: | ---: |
| Retrospective-weather control | 4,829 | 7 | 12.8887 |
| Production-style 24h NBM weather | 4,829 | 7 | 12.9269 |
| Production-style 12h NBM weather | 4,829 | 7 | 12.9201 |
| Production-style 6h NBM weather | 4,829 | 7 | 12.9186 |
| Core-continuous 24h sensitivity | 4,829 | 7 | 12.8765 |
| Core-continuous 12h sensitivity | 4,829 | 7 | 12.8801 |
| Core-continuous 6h sensitivity | 4,829 | 7 | 12.8818 |

The primary production-style variants replace temperature, dew point, relative humidity, sustained wind, precipitation and snowfall with archived forecast values, recompute wind/temperature categories, and prevent realized historical pressure from leaking into the forecast representation.

The core-continuous sensitivity substitutes only temperature/dew point/RH/wind, retains the retrospective precipitation/snow fields, and still removes realized pressure. It is diagnostic only because its retained period-weather fields are not operationally realistic.

## Does forecast weather materially degrade the current model?

The frozen degradation gate required all of:

1. positive mean paired MAE delta
2. game-bootstrap 95% CI entirely above zero
3. season-cluster 95% CI entirely above zero
4. worse in at least 70% of test seasons
5. degradation of at least 0.03 MAE points

### 24h versus retrospective

- mean MAE delta: **+0.03820**
- game-bootstrap 95% CI: **-0.00469 to +0.08145**
- season-cluster 95% CI: **+0.00827 to +0.07455**
- seasons worse: 6 / 7
- formal degradation gate: **FAIL / not established**

### 12h versus retrospective

- mean MAE delta: **+0.03135**
- game-bootstrap 95% CI: **-0.00964 to +0.07439**
- season-cluster 95% CI: **+0.01049 to +0.05507**
- seasons worse: 6 / 7
- formal degradation gate: **FAIL / not established**

### 6h versus retrospective

- mean MAE delta: **+0.02992**
- game-bootstrap 95% CI: **-0.01088 to +0.07099**
- season-cluster 95% CI: **+0.01509 to +0.04592**
- seasons worse: 7 / 7
- formal degradation gate: **FAIL / not established**

The direction is worth noting: forecast-weather scoring is slightly worse on average, and the 6h representation was worse in all seven seasons. However, the game-level bootstrap intervals include zero at every lead, and the 6h mean is just below the frozen 0.03 materiality floor. Therefore the preregistered evidence does **not** support declaring a material production distribution-shift failure.

## What part of the forecast representation appears responsible?

The diagnostic core-continuous sensitivities were slightly **better**, not worse, than the retrospective control:

- 24h core delta: -0.01223
- 12h core delta: -0.00858
- 6h core delta: -0.00692

None was statistically established as an improvement.

Because both the primary and core sensitivities remove realized pressure, while only the primary representation also substitutes NBM six-hour precipitation/snow fields, this pattern suggests that the small primary degradation is associated mainly with the period-precipitation/snow representation rather than forecast temperature, dew point, humidity or sustained wind themselves.

This should **not** be interpreted as proof that precipitation is harmful. The retrospective, NBM, and live NWS precipitation fields do not necessarily have identical temporal semantics. The result is better interpreted as evidence that weather-feature definitions should be made temporally consistent before using precipitation/snow in a future forecast-compatible historical training study.

## How late does weather need to be updated?

The preregistered material-update gate required a later forecast to improve MAE by at least 0.03 points, have both bootstrap CIs entirely below zero, and improve at least 70% of seasons.

### 12h versus 24h

- mean MAE delta: **-0.00685**
- game-bootstrap CI: -0.03197 to +0.01696
- season-cluster CI: -0.03425 to +0.02128
- seasons improved: 4 / 7
- gate: **FAIL**

### 6h versus 12h

- mean MAE delta: **-0.00143**
- game-bootstrap CI: -0.02196 to +0.01920
- season-cluster CI: -0.01614 to +0.01266
- seasons improved: 4 / 7
- gate: **FAIL**

### 6h versus 24h

- mean MAE delta: **-0.00828**
- game-bootstrap CI: -0.03363 to +0.01668
- season-cluster CI: -0.03966 to +0.02406
- seasons improved: 3 / 7
- gate: **FAIL**

The physical weather forecasts improve as kickoff approaches, but the model does not convert that improvement into a meaningful aggregate MAE gain. Under the present GENERAL HGB architecture, **there is no evidence that waiting from 24h to 12h or 6h before kickoff materially improves total prediction accuracy**.

This does not imply that the live system should stop refreshing weather. Updating current forecasts remains operationally sensible, especially for unusual or rapidly evolving events. It means only that the historical aggregate model-MAE evidence does not support a special later-update timing rule.

## Does training on archived forecasts solve the mismatch?

Forecast-trained models could first be tested in 2021 because the frozen rule required at least 1,000 prior common-sample forecast-training games. The forecast-trained evaluation therefore contains 3,626 games across 2021-2025.

### Forecast-trained MAE

| Lead | Forecast-trained MAE |
| --- | ---: |
| 24h | 12.9650 |
| 12h | 12.9659 |
| 6h | 12.9470 |

Against the production-style models on the same 2021-2025 games, forecast-trained models were materially worse:

| Lead | Production-style same-sample MAE | Forecast-trained MAE | Delta |
| --- | ---: | ---: | ---: |
| 24h | 12.6781 | 12.9650 | **+0.2869** |
| 12h | 12.6533 | 12.9659 | **+0.3126** |
| 6h | 12.6524 | 12.9470 | **+0.2946** |

All three game-level and season-cluster confidence intervals were entirely above zero. The forecast-trained variant was worse in all five test seasons at 24h and 12h, and four of five at 6h.

However, the production-style models have access to the much longer pre-2019 retrospective training history. A matched-population diagnostic therefore trained the control only on the same 2019-forward common-sample games.

Matched-population results:

- 24h: forecast-trained minus matched control = +0.05359, CIs include zero
- 12h: +0.03698, CIs include zero
- 6h: +0.01856, CIs include zero

Thus the large disadvantage versus the production-style model is primarily consistent with losing the longer historical training sample, not with a uniquely harmful forecast-training method. Once training population is matched, forecast-specific training still does **not** demonstrate an advantage.

**Decision: do not retrain GENERAL HGB only on the 2019-2025 NBM-era forecast sample.**

## Weather-effect persistence diagnostics

These diagnostics use frozen, previously defined screens and are not promotion tests.

### High total + high humidity

`closing total >=60 and RH >=80%` was notably stable across retrospective and forecast representations:

| Representation | Games | Avg market residual | Under rate |
| --- | ---: | ---: | ---: |
| Retrospective | 141 | -4.70 | 65.2% |
| 24h | 123 | -4.75 | 62.6% |
| 12h | 130 | -4.68 | 63.1% |
| 6h | 125 | -4.74 | 63.2% |

This is the clearest example in this study of a previously observed weather/context relationship retaining similar direction and magnitude under genuine pregame forecast inputs. It remains diagnostic; no production threshold is changed here.

### High total + moderate wind

`closing total >=60 and wind >=10 mph` also retained the same general direction:

- retrospective: -2.68 average residual, 57.4% under
- 24h: -2.32, 56.7% under
- 12h: -2.00, 56.0% under
- 6h: -2.06, 55.7% under

### Higher wind threshold with total >=58

The previously stronger retrospective screen `total >=58 and wind >=12 mph` did **not** transfer nearly as well:

- retrospective: -2.17 residual, 57.9% under
- 24h: -0.39, 51.6% under
- 12h: -0.22, 52.1% under
- 6h: +0.24, 50.0% under

This is a useful example of a retrospective weather effect that largely disappears when selection is based on information actually available before kickoff.

### Standalone wind thresholds

Standalone `wind >=15 mph` did not have a meaningful retrospective under effect in this sample (average residual approximately +0.03), reinforcing that wind should not be interpreted independently of market/context variables.

## Main conclusions

1. **Archived operationally realistic forecast reconstruction is feasible.** The NBM/NBS archive provided 100% three-lead coverage for all 4,829 station-mapped research games after fallback handling.
2. **24h, 12h and 6h forecasts are all reasonably close to retrospective weather for the main continuous fields.** Physical accuracy improves modestly as kickoff approaches.
3. **The current train-on-retrospective / score-on-forecast architecture does not trigger the frozen material-degradation warning.** There is a small consistent penalty, but uncertainty and the materiality floor prevent calling it established.
4. **There is no demonstrated model-value reason to wait from 24h to 12h or 6h.** Later weather is physically better, but GENERAL HGB MAE improves by less than 0.01 point in aggregate.
5. **Training only on archived NBM-era forecast weather is not an improvement.** It sacrifices valuable long historical training depth and performs materially worse than the current long-history production-style approach.
6. **Core continuous forecast weather itself is not the apparent problem.** The primary representation penalty appears concentrated in how period precipitation/snow fields are represented relative to retrospective features.
7. **Some retrospective weather/context relationships survive operational realism better than others.** High-total/high-humidity is notably stable; the total>=58/wind>=12 screen largely disappears when forecast rather than retrospective wind selects the games.

## Production decision

**No production change.**

Do not:

- replace the current GENERAL HGB with a forecast-trained 2019-2025 model
- impose a new 6-hour-only scoring/update rule
- promote any post-hoc lead time
- modify live pick thresholds based on this study alone

Continue using the latest available live forecast operationally, but do not expect the average model MAE to change materially solely because the weather forecast has moved from 24 hours to 6 hours before kickoff.

## Best follow-up research question

If another forecast-realism study is pursued, the cleanest next question is **forecast-compatible weather representation**, not another lead-time search.

A new preregistered study could test whether the long historical training sample can be retained while defining historical and live precipitation/snow/pressure inputs on genuinely comparable temporal semantics. The current core-continuous diagnostic suggests that temperature, dew point, humidity and sustained wind transfer reasonably well; the representation mismatch is more likely in the remaining weather fields.

That follow-up would require a new protocol and evidence gate before results are viewed. It is not implied by, or automatically promoted from, this completed study.
