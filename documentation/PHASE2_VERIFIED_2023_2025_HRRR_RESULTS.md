# Phase 2 Verified NOAA HRRR Forecast Vintage Pilot — 2023–2025

**Reproduced 2026-11-08, source GitHub Actions run [42460132132](https://github.com/JacZerin24/cfb-weather-totals/actions/runs/42460132132).** This is a real NOAA-data retrieval and nearest-gridpoint engineering verification, **not a verified betting backtest or evidence of a profitable strategy**. The draft Phase 2 PR is [#43](https://github.com/JacZerin24/cfb-weather-totals/pull/43), stacked on Phase 1 draft PR #42, with no modifications to the live pipeline.

## Results

- **12 distinct FBS games**, one predetermined first home FBS opponent game per venue and year (2023, 2024, 2025), across **LSU Tiger Stadium, Wisconsin Camp Randall, Colorado Folsom Field, and Washington Husky Stadium**. All four venue coordinates were independently checked against public stadium records. The fixed game sample was committed before fetching the expanded sample's forecasts.
- **36 of 36 nominally feasible** 24h, 6h, and 2h archived HRRR forecasts successfully retrieved from NOAA AWS and decoded via the original GRIB2 message with its internal cycle and forecast-valid time. **0 archive fetch failures**.
- **12 of 12 48-hour horizons explicitly unavailable**, as dictated by the conservative 2h dissemination-lag assumption and HRRR 48h maximum forecast lead; not data download failures.
- **Nearest-gridpoint distances 0.57–1.55 km** across all extracted messages.
- **Mean absolute 24h-to-2h forecast revision across these 12 games:** **3.62°F** temperature and **2.74 mph** 10-meter wind; maximum absolute revisions **8.0°F** and **9.4 mph**, respectively. These numbers describe only the fixed engineering sample. Do not estimate betting performance from them.
- **0 verified timestamped historical sportsbook quotes**, **0 archived same-time model predictions**, so *actual historical ROI is unknown/null*, **not 0%**. The original CFBD closing totals are not substitutes for executable prices.

## Individual game revision examples (24h versus 2h pre-kickoff)

| Season | Game | T24 (°F) | T2 (°F) | Change (°F) | Wind24 (mph) | Wind2 (mph) | Change (mph) |
|---|---|---:|---:|---:|---:|---:|---:|
| 2023 | Nebraska at Colorado | 77.1 | 73.8 | -3.3 | 1.2 | 2.8 | +1.6 |
| 2023 | Arkansas at LSU | 85.7 | 93.3 | +7.6 | 9.5 | 6.1 | -3.4 |
| 2023 | Boise State at Washington | 86.7 | 78.7 | -8.0 | 4.2 | 4.0 | -0.2 |
| 2023 | Buffalo at Wisconsin | 91.2 | 88.5 | -2.7 | 10.2 | 10.9 | +0.7 |
| 2024 | Baylor at Colorado | 61.6 | 65.9 | +4.2 | 13.0 | 3.6 | -9.4 |
| 2024 | UCLA at LSU | 89.0 | 90.2 | +1.2 | 9.5 | 6.9 | -2.7 |
| 2024 | Eastern Michigan at Washington | 84.1 | 79.5 | -4.7 | 5.1 | 3.4 | -1.7 |
| 2024 | Western Michigan at Wisconsin | 69.2 | 69.4 | +0.2 | 7.5 | 7.4 | -0.0 |
| 2025 | Georgia Tech at Colorado | 75.4 | 78.8 | +3.4 | 10.8 | 4.1 | -6.7 |
| 2025 | Louisiana Tech at LSU | 87.9 | 89.4 | +1.5 | 4.6 | 9.2 | +4.6 |
| 2025 | Colorado State at Washington | 69.9 | 64.5 | -5.3 | 3.4 | 2.6 | -0.8 |
| 2025 | Miami (OH) at Wisconsin | 67.5 | 66.1 | -1.4 | 6.8 | 5.7 | -1.1 |

The largest temperature revision was **Washington vs Boise State, 2023** (8.0°F cooler at 2h than at 24h). The largest wind revision was **Colorado vs Baylor, 2024** (9.4 mph lower at 2h than at 24h). These examples justify testing weather forecast evolution as a prospective research feature, not automatically applying a betting rule.

## Provenance and limitations

The source weather was decoded directly from archive GRIB2 fields `TMP:2 m`, `DPT:2 m`, `UGRD:10 m`, `VGRD:10 m`, `GUST:surface`, at each stadium point; GRIB internal initialization and valid timestamps were explicitly checked; NOAA object URL, `.idx` SHA-256, individual message SHA-256 digests, gridpoint distance and actual **2026 archive retrieval time** were retained in the research CSV. The CSV SHA-256 is `3080c7c9f2822d4da42ff32d5179c6bde9d3086660090dee6cdee6bccc61e79b`.

For nominal weather availability, the reconstruction uses **initialization + assumed 2 hours ≤ betting decision**. Actual historical publication time at which these GRIB messages were accessible has **not independently been proven**, nor does the stored CFBD kickoff represent a verified schedule version as known to every bookmaker at the decision time. Thus forecast reconstruction is strong evidence of historical model **run content**, not proven as-of publication availability. Forecast-valid times are within 30 minutes of kickoff. The model samples only kickoff weather and not rain/wind changes over the entire game.

This is a **small, nonrandom, early-season four-venue sample**. It does not evaluate a sufficient number of windy, cold, snow, or tropical-weather matches, and it cannot estimate a robust climate-specific or ROI effect. It only demonstrates an auditable method that can scale.

A full 2023–2025 retrospective strategy replay still needs (1) an authentic pregame sportsbook snapshot for each game with the actual UNDER price and timestamp, (2) a decision-time forecast-availability publication criterion, (3) a reproducible pregame kickoff/venue map, and (4) an independently chronological model retrained on matching forecast features. A paid historical odds feed such as https://the-odds-api.com/historical-odds-data/ is an option but was NOT accessed, subscribed to, or billed during this work.

## Files

- `research_data/phase2/noaa_2023_2025_stadium_forecast_vintages.csv` — 48 full game-horizon rows, including 36 reconstructed weather points and archive GRIB metadata; no betting price or odds inferred.
- `research_data/phase2/forecast_revision_24h_to_2h.csv` — 12-game paired forecast changes.
- `research_data/phase2/pilot_summary_20261108.json` — raw GitHub workflow summary.
- `research_data/phase2/verified_metrics_20261108.json` — summarized checked results and source hash.
- `research/phase2/hrrr_archive_pilot.py` — read-only NOAA weather retrieval, decoding, and provenance checks, with local offline tests.
- `research/phase2/odds_snapshot_import.py` — offline-only price/vintage parser; does not request paid data.

**Operational isolation:** All code changes are additive on a separate unmerged draft research PR. The sole new workflow is `pull_request` triggered, has contents read-only permission, no secrets, no schedule, and writes only standalone research artifacts. Production workflows, live model, thresholds, website, 2026 prospective ledger, and all actual wagers remain unchanged.
