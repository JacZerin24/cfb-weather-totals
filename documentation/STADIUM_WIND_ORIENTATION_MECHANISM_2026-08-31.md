# Stadium Wind Orientation Football-Mechanism Results — 2026-08-31

**Status: RESEARCH ONLY — NO CHANGE TO THE WORKING MODEL, QUALIFIER/LEAN CLASSIFICATIONS, OR PROSPECTIVE 2026 LEDGER.**

## Locked question

Following the orientation falsification phase, the mechanism study kept the previously identified regime fixed rather than searching new betting thresholds:

- FBS vs FBS
- outdoor games
- raw wind >10 and <=15 mph
- cross-aligned = 60–90 degrees from the field axis
- parallel-aligned = 0–30 degrees

Controlled models include venue, season, raw wind speed, temperature, humidity, precipitation, dew point, pressure, closing total, neutral-site status, and line provider, with uncertainty clustered by venue.

## Data

The CFBD game-team endpoint produced 610,289 long-format team-stat rows across the historical pull. The controlled mechanism sample contains 1,209 cross/parallel games for 2014–2025 and 480 games for 2021–2025.

Available categories support passing/rushing efficiency, touchdowns, turnovers, kicking points, and other aggregate team stats. They do not provide clean field-goal makes/attempts or punt-distance measures in this endpoint.

## Primary mechanism results — 2014–2025

| Outcome | Adjusted cross minus parallel | p (two-sided) | Interpretation |
|---|---:|---:|---|
| Passing yards/attempt | -0.068 | 0.507 | No meaningful passing-efficiency evidence |
| Completion percentage | +0.003 | 0.609 | No meaningful passing-accuracy evidence |
| Combined pass attempts | +2.18 | 0.024 | More pass volume in full-era sample |
| Pass-play share | +0.0136 | 0.014 | More pass-heavy in full-era sample |
| Passing TDs | +0.123 | 0.354 | No passing-TD reduction |
| Rush yards/attempt | -0.157 | 0.047 | Negative-control metric moves lower |
| Rushing TDs | -0.226 | 0.102 | Suggestive but not secure |
| Kicking points (FG+PAT aggregate) | -0.579 | 0.081 | Suggestive only; not a clean field-goal test |
| Turnovers | -0.042 | 0.711 | No meaningful difference |
| First-half points | -0.515 | 0.481 | No secure timing effect |
| Second-half points | -0.860 | 0.287 | No secure timing effect |
| Market residual | -0.999 pts | 0.383 | Same controlled scoring result as falsification phase |

## Higher-resolution wind-direction era — 2021–2025

The simple passing-volume signals do not replicate in the recent era:

- passing yards/attempt: -0.104, p=0.598
- completion percentage: -0.0007, p=0.940
- pass attempts: -1.17, p=0.485
- pass-play share: -0.0037, p=0.700
- passing TDs: +0.009, p=0.965

Other directional results are:

- rush yards/attempt: -0.249, p=0.082
- kicking points (FG+PAT aggregate): -1.062, p=0.100
- market residual: -2.592 points, p=0.189

## Interpretation

### Mechanism confirmed: NO
### Orientation hypothesis disproved: NO
### Continue prospective shadow evaluation: YES

This study weakens a simple aerodynamic explanation that cross-field wind lowers totals primarily by degrading passing accuracy or passing efficiency. The primary passing metrics are effectively unchanged after controls, while rushing efficiency — intended as a negative control — moves more clearly in the hypothesized scoring-lower direction.

Lower aggregate kicking points are potentially interesting, particularly in the recent era, but `kickingPoints` combines field goals and extra points and therefore cannot establish a field-goal-specific wind mechanism.

The results are consistent with a more complicated football/venue/weather process, or with residual confounding/noise. They do not justify designing a new production rule or adding more historically optimized thresholds.

## Decision

Do not promote stadium-relative wind into the working model from the mechanism evidence. Preserve the already-frozen `orientation-crosswind-hgb-v0.1` 2026 shadow challenger and let genuinely prospective games provide the next major evidence.

One bounded historical extension remains scientifically defensible: a play-by-play mechanism study using the same locked >10 to <=15 mph regime to test field goals and explosive passing directly. That study should not search betting thresholds. After that, historical orientation research should stop and the project should rely primarily on the prospective shadow evaluation.
