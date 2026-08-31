# Stadium Wind Orientation Falsification Results — 2026-08-31

**Status: RESEARCH ONLY — DO NOT USE TO CHANGE QUALIFIER/LEAN CLASSIFICATIONS OR THE PROSPECTIVE 2026 LEDGER.**

## Research question

The earlier exploratory phase identified the existing 10–15 mph wind bin as the most interesting field-orientation regime. This phase deliberately tries to falsify that result by controlling for venue and weather, randomizing field axes, shuffling angle labels within venues, testing axis-measurement uncertainty, and checking whether one venue or season drives the effect.

Because the 10–15 mph regime was selected after inspecting the 2014–2025 historical data, these are robustness tests rather than untouched confirmatory p-values.

## Dataset

- FBS outdoor common-support games with closing total, final score, measured stadium axis, wind speed, and wind direction: **7,823**
- Primary 10–15 mph controlled sample after pressure-quality completeness: **1,754**
- Cross-aligned games (60–90°): **539**
- Parallel-aligned games (0–30°): **670**

## Primary controlled result

After removing variation explained by venue, season, raw wind speed, temperature, humidity, precipitation, dew point, pressure, closing total, neutral-site status, and line provider:

- Adjusted cross-minus-parallel scoring residual: **-1.08 points**
- Standard venue fixed-effects coefficient: **-1.00 points**
- Cluster-robust SE: **1.15 points**
- 95% CI: **[-3.24, +1.25]**
- Two-sided p: **0.383**

The original raw effect therefore shrinks materially after venue/weather controls. That is evidence that some of the exploratory headline was confounded, and is a reason not to promote the feature yet.

Adding sine/cosine harmonics of the absolute compass wind direction strengthens rather than eliminates the relative-orientation effect:

- Direction-regime-controlled coefficient: **-2.70 points**
- Two-sided p: **0.085**

This argues against the signal being explained solely by broad compass-direction weather regimes, but it still does not constitute confirmation.

## Placebo and permutation tests

| Test | One-sided p | Two-sided p |
|---|---:|---:|
| Random 0–180° axis at every venue | 0.106 | 0.212 |
| Shuffle measured axes among venues | 0.071 | 0.142 |
| Shuffle observed angle labels within venue | 0.099 | 0.197 |
| Shuffle angle labels within venue-season | 0.084 | 0.169 |

The actual measured geometry beats most placebo configurations, but the full-era result is suggestive rather than decisive.

## Measurement uncertainty

Perturbing each field axis repeatedly using its estimated measurement uncertainty produced:

- Median adjusted effect: **-1.17 points**
- 95% jitter range: **[-1.59, -0.77]**
- Fraction of simulations retaining a negative effect: **100%**

The result is therefore not sensitive to the approximately 2° field-axis measurement uncertainty.

## Existing wind-bin falsification

| Wind | Games | Raw cross-parallel | Adjusted cross-parallel | FE coefficient | FE p | Random-axis one-sided p |
|---|---:|---:|---:|---:|---:|---:|
| 0–5 | 2,261 | +0.78 | +0.70 | +0.90 | 0.342 | 0.803 |
| 5–10 | 2,998 | +1.30 | +1.43 | +1.34 | 0.161 | 0.974 |
| **10–15** | **1,754** | **-2.05** | **-1.08** | **-1.00** | **0.383** | **0.110** |
| 15–20 | 412 | +0.88 | -1.21 | -2.18 | 0.563 | 0.165 |
| 20+ | 162 | -3.54 | -3.53 | -6.58 | 0.683 | 0.064 |

The relationship is not a simple monotonic “more crosswind always means lower scoring” effect. In particular, 5–10 mph points in the opposite direction. That is a major reason not to manufacture a hand-tuned crosswind betting rule from the historical sample.

## Within-venue replication

| Minimum games in each orientation group | Venues | Venues with cross lower | Fraction | Median venue difference | One-sided sign-test p |
|---|---:|---:|---:|---:|---:|
| 1 | 112 | 68 | 60.7% | -2.60 pts | 0.015 |
| 3 | 53 | 31 | 58.5% | -0.78 pts | 0.136 |
| 5 | 26 | 16 | 61.5% | -1.68 pts | 0.163 |

The broad sign is favorable, but once a reasonable minimum sample per stadium is required, statistical confidence falls. This is another caution against overclaiming the within-stadium evidence.

## Era sensitivity

| Period | Games | Adjusted cross-parallel | FE coefficient | FE p | Random-axis one-sided p |
|---|---:|---:|---:|---:|---:|
| 2014–2020 | 1,050 | -1.18 | -1.30 | 0.436 | 0.161 |
| **2021–2025** | **704** | **-2.02** | **-2.59** | **0.189** | **0.036** |
| 2023–2025 | 432 | -1.86 | -2.81 | 0.360 | 0.083 |

The recent-era adjusted signal is stronger. Historical wind-direction precision also improves materially: 2014–2017 have only 37 distinct direction values and essentially all are exact multiples of 10°, while 2025 has 202 distinct values and only about 67.5% are exact multiples of 10°.

This is encouraging, but 2021–2025 is not an untouched holdout because it was included in discovery.

## Influence analysis

- Leave-one-venue-out adjusted effect range: **-1.39 to -0.94 points**
- Leave-one-season-out adjusted effect range: **-1.36 to -0.50 points**
- No individual venue or season reverses the sign.

## Decision

### Continue research: YES
### Promote to working model now: NO

This phase weakens the simplest claim that the earlier raw UNDER rate proves orientation works, but it strengthens the deeper hypothesis that measured field-relative wind may contain real information not captured by raw wind speed alone.

The main reasons not to promote yet are:

1. Venue/weather controls reduce the original effect.
2. Full-era permutation tests are suggestive rather than decisive.
3. The 10–15 mph regime was discovered in the same historical sample now being tested.
4. The relationship across wind bins is nonlinear and not yet mechanistically explained.
5. The earlier simple HGB feature-addition test produced essentially no overall MAE improvement.

## Recommended next phase

**Stop historical threshold hunting here.** The next step should be to lock a minimal orientation challenger specification before looking at 2026 outcomes, begin research-only prospective 2026 shadow capture of wind direction/field axis/relative angle/crosswind/alongwind plus baseline and challenger predictions, and separately test a football mechanism such as passing efficiency, explosive passing, field-goal performance, punting, and scoring by period.

The operational Qualifier/Lean interface and current prospective protocol remain unchanged during that work. Only a repeatable, practically meaningful prospective improvement should trigger discussion of a versioned production promotion.
