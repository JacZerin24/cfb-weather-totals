# Phase 2: 2023–2025 archived weather and odds feasibility pilot

**Status:** Isolated R&D. No modification to live scores, schedules, betting thresholds, config, model, `data/`, site, or the frozen 2026 prospective ledger. **Not an executable ROI backtest.**

## Goal and provenance

Reconstruct actual historic HRRR GRIB forecast **run vintages** for a fixed sample of college-football games. Then prepare a separate, conservative importer for bookmaker total-and-UNDER-price **snapshots at a historical decision time**. This project refuses to claim a verified return until both sources and a separately trained/replayed model are present. Distinguish `historical CFBD closing_total` from timestamped bookmaker quotes. Never use postgame CFBD weather observations as forecast input.

### Predeclared pilot sample

The `sample_games_2023_2025.csv` sample was derived *before inspecting archived HRRR* from the November 7 2026 stored CFBD modeling artifact. It selects the first two chronological LSU home games against FBS teams **per season**, with the same named outdoor venue and a historical CFBD line, for seasons 2023–2025. This is a **single-stadium engineering sample, not a representative strategy validation**. Pilot CI tests only the first game of each season: Arkansas at LSU (2023), UCLA at LSU (2024), Louisiana Tech at LSU (2025). Six cases can be expanded via `--per-season 2`, but expanding to multiple stadiums is required for science conclusions.

LSU Tiger Stadium is CFBD venue 3958 and is fixed at 30.4120° N, 91.1838° W from https://cfbtrack.com/stadiums/3958-tiger-stadium-la . The location is manually audited; the original 164-column modeling data does *not* include venue lat/lon, so large-scale expansion needs a trustworthy historic venue-coordinate mapping. No observations or results were used to select cases.

## Weather side: original archived forecast cycles

Public NOAA HRRR object path `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfNN.grib2`. The companion `.idx` indexes GRIB message bytes; use HTTP Range for only `TMP:2 m`, `DPT:2 m`, `UGRD:10 m`, `VGRD:10 m`, `GUST:surface`. Decode values at the nearest model point at stadium coordinates with ecCodes. Record exact archive object URL, cycle, fXX, valid time, message hashes, gridpoint distance and retrieval time. Derive temperature/dewpoint F, RH%, 10 m wind mph, gust mph, and meteorological direction. **No actual game-weather fields are involved.**

- Fixed candidate decisions: kickoff minus 48h, 24h, 6h, and 2h.
- A conservative 2-hour *assumed* model-publication lag requires initialization + 2h ≤ decision time. **The model run timestamp alone does not prove it was available by then**; retrospective quote-availability validation remains necessary.
- HRRR long runs (00/06/12/18 UTC) reach 48h; other cycles generally 18h. Valid time must be within 30 min of listed kickoff. If no available HRRR lead exists, **leave the 48h row unavailable**, never replace it with an analysis or a forecast initialized after the decision.
- HRRR is not equivalent to the NWS official forecast used operationally, and it is not a true full-game weather forecast. A rigorous strategy test will need forecast-window features and a revised, re-trained model rather than applying the historic model to a different weather-input distribution without retraining.
- The stored kickoff may reflect a later schedule correction; an executable backtest additionally needs kickoff as known at decision time. This pilot only tests data-access and forecast-vintage reconstruction feasibility.

Source: https://registry.opendata.aws/noaa-hrrr-pds/ and https://herbie.readthedocs.io/en/stable/user_guide/background/grib2.html

## Odds side: no credentials and no inferred prices

`odds_snapshot_import.py` is an **offline-only** parser for supplied historical JSON from https://the-odds-api.com/historical-odds-data/ . It never sends an API request. Rows retain sports-event ID, bookmaker, historical snapshot timestamp, market/book update time, quoted UNDER total and actual American price, kickoff, and unique quote provenance. Conservative quote availability is **snapshot timestamp**, not the earlier book update time. Matching to CFBD games requires an explicit reviewed event-to-CFBD mapping and exact case-insensitive team names (or manually reviewed aliases), kickoff difference ≤45m, and quote snapshot no older than 12h. Missing prices/mappings are excluded. It does not manufacture an opening or closing price from CFBD records.

Available public GitHub collectors reviewed were founded too late for this 2023–25 study (e.g. 2026 `vladk-III/College-Football`). Published historical books with opening/closing totals, but *without archived timestamped actionable sportsbook quotes*, cannot satisfy the strict replay standard. The Odds API historical NCAAF feed offers this data commercially from 2020 onward; no paid API calls, subscriptions, or keys were used.

## How to run without touching production

From repo root:

```bash
python -m unittest discover -s research/phase2 -p 'test_*.py' -v
python -m research.phase2.hrrr_archive_pilot --dry-run --per-season 1 --out research/phase2/dry_run
python -m research.phase2.hrrr_archive_pilot --per-season 1 --out research/phase2/pilot_results
```

The last command downloads *only selected public GRIB fields* and requires `pip install requests eccodes` and network access. All writes are under `research/phase2/`. The new GitHub workflow has **pull_request only**, no schedule, no secrets, contents read-only, no production checkout writes, and uploads an artifact, not committed operational outputs. No sportsbook transactions or paid credits.

To parse already-obtained authorized historical snapshots **offline**:

```bash
python -m research.phase2.odds_snapshot_import \
  --snapshots /path/to/snapshots.json \
  --event-map /path/to/reviewed_event_to_cfbd_mapping.csv \
  --schedule research/phase2/pilot_results/forecast_vintage_pilot.csv \
  --out research/phase2/pilot_results/eligible_odds_quotes.csv
```

## Required before profit conclusions

1. Verify forecast data and *availability* at the exact decision horizon; quantify S3 archive field coverage and individual forecast QC across many stadiums.
2. Secure real archived book-specific UNDER quotes plus timestamped prices for that same game/time. Reconstruct event IDs, kickoff known as-of, data vendor coverage and quote staleness.
3. Train/replay model out-of-sample using *equivalent pregame features* with chronological training cutoff, no hindsight weather, no postgame lines; record each season's coverage.
4. Compute season-level ROI only on actual quotes using a frozen candidate strategy, include bootstrapped confidence bands and variation by season. **No 50% ROI claim without a demonstrably independent and sufficiently large prospective test.**

The retrospective reconstructed weather is an engineering feasibility demonstration, not proof that the legacy historical selection was placeable or that its edge generalizes.
