# Stadium Orientation × Wind Research

## Status

**Research only. No operational model, weekly classification, prospective entry, or grading logic is changed by this work.**

This track tests whether wind direction relative to the football field adds repeatable information beyond the raw wind-speed feature already used by the CFB totals models. The current QUALIFIES / LEAN / NO PLAY presentation remains the intended operational interface if an orientation enhancement is ever promoted.

## Why this is worth testing

The historical CFBD dataset contains wind direction in addition to wind speed, while the current production GENERAL HGB uses raw wind speed but does not use wind direction or stadium orientation. Two games with the same 15 mph wind are therefore meteorologically different when one wind is nearly parallel to the field and the other is nearly perpendicular.

The orientation hypothesis is that the **field-relative wind vector** may be more informative than raw wind speed alone, particularly for passing and kicking conditions.

## Canonical orientation data

`data/reference/stadium_orientations.csv` is derived from `wind_orientation_measurement_workbook_v1_2.xlsx`.

The workbook remains the detailed measurement record with goal-line endpoint coordinates and QA. The research CSV stores the measured field axes needed by the code and joins by CFBD `venue_id`; it does not use fuzzy venue-name matching.

The field is represented as an **undirected 0–180° axis**. This is intentional: teams switch ends during a football game, so a whole-game totals model should not assign a permanent signed headwind or tailwind to one direction of play.

## Derived variables

Let:

- `V` = wind speed in mph
- `θ` = smallest angle from the field axis to the wind axis, in the range 0–90°

The research variables are:

- `wind_field_angle_deg = θ`
- `crosswind_mph = V × sin(θ)`
- `alongwind_mph = V × cos(θ)`

Historical wind direction is meteorological direction **from** which the wind blows. Because this research uses an undirected field axis and absolute vector-component magnitudes, converting from-direction to toward-direction does not change these magnitudes.

## Research design

### 1. Data-quality gate

Before any model comparison, report orientation and wind-direction coverage by season and by division track. FCS is not allowed to borrow an FBS conclusion. The live system already uses a dedicated FCS model, so any FCS orientation enhancement must ultimately be evaluated in that dedicated framework after FCS venue coverage is adequate.

### 2. Descriptive physical-signal screen

Use the repository's existing raw wind bins:

- 0–5 mph
- 5–10 mph
- 10–15 mph
- 15–20 mph
- 20+ mph

Within each raw-wind bin, divide the 0–90° field-relative angle into equal-width, non-optimized bins:

- parallel: 0–30°
- oblique: 30–60°
- cross: 60–90°

These tables are **exploratory/descriptive only**. A profitable bucket cannot promote a feature by itself.

### 3. Common-support GENERAL HGB test

For FBS-vs-FBS games that have wind speed, wind direction, and a measured field axis, compare the current GENERAL HGB against two physically motivated variants:

1. `baseline`: current exact feature list
2. `crosswind`: baseline + `crosswind_mph`
3. `components`: baseline + `crosswind_mph` + `alongwind_mph`

All variants use the same games, current HGB hyperparameters, and walk-forward season structure. Each test season is predicted using only prior seasons.

This **common-support test isolates incremental feature value**. It is not a claim about full production-universe performance because games without measured orientations are excluded from all variants.

### 4. Economic screen without threshold tuning

Evaluate the existing GENERAL HGB production qualifier unchanged:

- UNDER only
- predicted market residual ≤ -3.5 points
- market total ≥ 56
- -110 settlement

No orientation-specific betting threshold is optimized in this stage.

## Preliminary findings from the June 14, 2026 historical-research artifact

These are **preliminary common-support results, not a production-model recommendation**.

- Games with closing totals and scores: 11,757 (2014–2025).
- FBS-vs-FBS orientation research coverage: **7,823 / 8,605 = 90.9%**.
- FCS-vs-FCS orientation research coverage: **339 / 2,345 = 14.5%**. FCS is currently under-covered for a mature orientation-model conclusion.
- The most notable descriptive pattern is in the existing 10–15 mph raw-wind bin. For FBS games:
  - parallel 0–30°: 677 games, average market residual +0.92, under rate 51.7%;
  - cross 60–90°: 555 games, average market residual -0.88, under rate 57.3%.
- The cross-minus-parallel residual difference in that 10–15 mph FBS bin is negative in 9 of 12 seasons. The direction is also present in 2023–2025, although the effect size is smaller than in the earlier sample.
- A continuous crosswind feature has **not** shown a decisive overall prediction-error improvement. Aggregate MAE differences are extremely small and season-to-season results are mixed.
- Some orientation variants improve the unchanged 3.5+/56+ UNDER qualifier economics on the common-support sample, but the gains are not yet uniform enough to justify promotion. Recent-season behavior is mixed, so the result remains exploratory.

## Promotion standard

An orientation feature may be considered for a new production model version only if the total body of evidence supports all of the following:

1. **Physical plausibility:** the behavior makes meteorological and football sense.
2. **Incremental value:** it adds information beyond raw wind speed and the existing model features.
3. **Out-of-sample value:** improvement survives walk-forward testing rather than only an in-sample rule table.
4. **Season stability:** results are not carried by one or two seasons.
5. **Sensitivity robustness:** reasonable changes in market provider, conference, weather-quality filters, and total ranges do not destroy the effect.
6. **Adequate coverage:** the relevant operational track has enough trustworthy venue orientation data.
7. **No leakage:** only information available at the prediction timestamp is used.
8. **Prospective confirmation:** after a candidate is frozen, it is shadow-scored on genuinely new 2026 games.
9. **Practical value:** any statistical gain is large enough to matter for decision support and qualifier quality.

If those gates are met strongly enough to justify a mid-2026 enhancement, the change must receive a **new documented model/protocol version and effective date**. Earlier 2026 entries remain permanently attributed to the model version that generated them.

## Running the research

After the historical modeling dataset has been built:

```powershell
python -m src.stadium_wind_orientation_research
```

Outputs are written only under:

`outputs/orientation_research/`

The script does not import or modify the prospective ledger and does not write `outputs/weekly_board.csv`.
