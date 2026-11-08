# Phase 1: historical decision-time realism audit

**Audit date:** 2026-11-08  
**Scope:** historical CFB totals (2014–2025), GENERAL HGB screening, historical inputs, and future replay feasibility.  
**Change policy:** research-only. This audit makes no modifications to operational scoring, scheduling, website generation, production thresholds, or immutable 2026 prospective records.

## Verdict

**Historical +11.64% ROI is NOT VERIFIED AS EXECUTABLE.** The published 2016–2025 HGB UNDER 4.0 / total 56+ result is 651 historical game selections, 643 graded bets, 376 wins, 267 losses, 8 pushes, and +11.64% simulated return using a flat -110 settlement convention. That is a retrospective walk-forward model result, **not** a decision-time odds-and-forecast backtest. No replacement return is estimated here.

### Evidence reviewed on main

1. `src/pull_historical_lines.py` calls CFBD `/lines` and stores provider, `overUnder`, spread, and moneylines, but no line quote timestamp or OVER/UNDER price. `src/build_dataset.py::pick_total` chooses the first provider by preference. The resulting `closing_total` is a research target/feature, not a verified pregame executable price.
2. `src/pull_historical_weather.py` calls CFBD `/games/weather`; `src/build_dataset.py` joins temperature/wind/precipitation by game ID without a weather forecast issuance or availability timestamp. An API record of game weather is not proof that a particular value was forecast when the bet was placed.
3. `src/model_bakeoff.py` and `src/hgb_regime_audit.py` walk forward by **season**. This avoids training on later seasons, but does not address within-game feature provenance or the market quote as of a specified pregame time.
4. `src/predict_week.py` uses actual NWS grid forecasts for current live games, but historical CFBD weather values do not reconstruct those previously issued forecasts.
5. `documentation/HGB_TRAINING_REGIME_AUDIT.md` documents `early_stopping=False` for the frozen HGB candidate, a historical result and a deliberate threshold search. The 2025 season was included when choosing the threshold, so it is not an entirely untouched strategy-selection holdout.
6. The source-of-truth 2026 prospective ledger is distinct and includes immutable scheduled snapshots. It is not modified by this audit.

## Existing historical data coverage (not decision-time coverage)

From committed `outputs/data_quality_summary.csv`:

| Period | Games with selected historical total | Games with any recorded game weather | Recorded game-weather coverage | Verified dated forecast vintages | Verified timestamped bookmaker totals/prices |
|---|---:|---:|---:|---:|---:|
| 2014–2015 | 1,445 | 1,430 | 99.0% | Not present in committed historical dataset | Not present in committed historical dataset |
| 2016–2025 | 10,312 | 10,102 | 98.0% | Not present in committed historical dataset | Not present in committed historical dataset |
| **2014–2025** | **11,757** | **11,532** | **98.1%** | **Not verifiable from committed historical dataset** | **Not verifiable from committed historical dataset** |

The `with_weather` count means **a temperature or wind observation/value exists** in the historical source. It does not mean an archived pregame weather **forecast** is present, and this coverage percentage must not be used as a forecast-vintage coverage percentage. These 11,757 rows are the subset with historical market totals/outcomes; they are not all 26,843 games in the larger underlying game universe.

The processed `modeling_dataset.csv` and raw pull CSVs are **not committed** to main; live workflows restore them from GitHub Actions artifacts. The November 7, 2026 live-training artifact (run `42380170001`) was recovered and its `modeling_dataset.csv` SHA-256 is `a2f5a61248afe895e39bde58c5d3f8e7d10ed490095418e45d69ff0bbc7b3678`. An unpinned local reproduction (sklearn 1.8, rather than the repository-pinned 1.9) produced 701 game selections, 404 wins, 287 losses and 10 pushes, so it cannot certify exact parity with the published 651-game result. The independent **pinned-environment** GitHub Actions check on November 8, 2026 (scikit-learn 1.9.0) **reproduced the published baseline exactly**: 651 game selections, 643 graded, 376–267 with 8 pushes, +74.8181818 net units, +11.6358% simulated ROI. Its input dataset SHA matched the November 7 archive. These two separate local/CI results demonstrate that the historical simulation is software-version sensitive, but reproducible with its pinned environment. The read-only `research/phase1/reproduce_baseline.py` is provided for that artifact; it reports the input SHA-256 and compares the result to published counts without reclassifying the historical number as a realized ROI.

## Completed Phase 1 evidence-gate execution (November 8, 2026)

Running the audit with the recovered `modeling_dataset.csv`, **without fabricating any odds, forecast, or prediction-vintage files**, produced:

- **26,835** completed-score games available for an audit of source provenance across the full game universe (the broader universe includes games without CFBD historical totals);
- **107,340** separate game/horizon rows (48h, 24h, 6h and 2h); these are alternative scenarios, not independent bets;
- **0** fully substantiated source/quote/forecast/prediction triples from the **supplied historical training artifact**;
- **0** execution-verified qualifying bets and ROI `null`, as required by the hard stop.

This does **not** say zero historical forecast or betting quotes existed elsewhere. It means their point-in-time evidence is not present in the input dataset used by the current historical backtest. A paid odds archive and genuine issued forecast vintages would be needed before an executable historical ROI can be estimated.

Pinned results are committed for auditability in `research/phase1/verified_results/`. They are the output of the research-only GitHub check, not changes to live/production data.

## Required evidence and fail-closed design

`research/phase1/decision_time_audit.py` evaluates predeclared 48h, 24h, 6h, and 2h lead horizons and requires **all** of the following to be available before the selected decision time:

- A bookmaker quote captured at or before the decision, no more than 12h old, with exact game ID, bookmaker, total, UNDER price, quote ID, and the kickoff known at that observation time (within 45m of the final event time). Untimestamped CFBD totals do **not** count.
- A forecast **vintage** with issue, availability, validity, weather values, and source identifier. The vintage must be available at the decision time, no more than 18h old, and valid within 90m of kickoff. Observed/game-weather rows are rejected.
- An archived pregame model prediction made by the decision time with a training cutoff strictly earlier than the decision and exact matching quote/vintage identifiers. Newly generated, hindsight-derived 2026 predictions cannot masquerade as archived pregame decisions.
- A scored game, a fixed UNDER edge >= 4.0, a price total >= 56.0, and the actual recorded bookmaker price for settlement.

When any source is absent, the script reports an exclusion reason and **ROI `null`**, never zero ROI, an inferred price, a substituted observed weather value, or an artificially reconstructed vintage. Different lead horizons represent *alternative* policies and must not be combined as independent bets. Date/time matching is strict; missing or rescheduled kickoff evidence is excluded.

The audit is a provenance/coverage gate rather than a new model: it does not attempt to train HGB itself on odds/forecast vintages. Once enough historical archive data is assembled, a separately frozen chronological replay model can be built. Its new synthetic retrospectively generated predictions must be labeled reconstruction research, not passed as *archived pregame predictions*.

## Accessible archive paths for a later full reconstruction

- The Odds API publishes paid timestamped NCAAF historical market snapshots (NCAAF since 2020; a separate FCS key from August 2026). Snapshots contain bookmaker-specific totals and prices and an event time. Coverage by game/bookmaker at each desired decision horizon must be measured; no API calls or paid credits were used during this audit. https://the-odds-api.com/historical-odds-data/
- NOAA HRRR archives in public cloud storage contain **forecast run vintages** beginning in 2014, with practical coverage, forecast lengths and formats varying by model era. Forecast availability must be reconstructed conservatively (run initialization alone is not proof that all fields were published). https://rapidrefresh.noaa.gov/hrrr/ and https://registry.opendata.aws/noaa-hrrr-pds/
- The CFBD game-weather and line endpoints are useful game-matching/reference sources but lack the exact quote/forecast-vintage metadata required for strict replay. https://api.collegefootballdata.com/api/games and https://api.collegefootballdata.com/data-availability

## Run locally (no network, no live writes)

With the original historical modeling artifact downloaded locally:

```bash
python -m research.phase1.reproduce_baseline --games data/processed/modeling_dataset.csv --out research/phase1/local_results
python -m research.phase1.decision_time_audit --games data/processed/modeling_dataset.csv --out research/phase1/local_results
python -m unittest discover -s research/phase1 -p 'test_*.py' -v
```

The second command intentionally returns a blocked coverage report when no timestamped odds, forecast vintages and archived pregame predictions are provided. To test historical replay inputs, pass `--odds path/to/quotes.csv --weather path/to/vintages.csv --predictions path/to/archived_predictions.csv`.

**Important:** baseline reproduction is historical source-code consistency only; it cannot prove executable 2016–2025 returns until the old observations/lines are replaced with genuinely available pregame inputs and the model retrained under a new, predeclared experimental protocol.

## Operational isolation checklist

- No edits to `src/`, `config/`, existing operational GitHub workflows, `docs/`, `outputs/` or `data/`. A new PR-scoped test-only workflow can retrieve a fixed training artifact without writing to production.
- No new scheduled jobs, API billing, bookmaker requests, or new data-provider network dependencies. The independent test-only GitHub Actions job may download the existing stored training artifact and install the already-pinned Python requirements.
- No alteration to 2026 prospective immutable snapshots or their grading.
- New code is additive, under `research/phase1/` plus this human-written audit document.
- Test fixture uses invented synthetic games; passing tests validate exclusion logic, **not** predictive profitability.
