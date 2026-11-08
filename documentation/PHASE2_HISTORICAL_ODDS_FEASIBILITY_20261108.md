# Phase 2B — Historical Odds Coverage and Cost Audit (2026-11-08)

Research only. No purchases, API credentials, historical sportsbook data downloads, or production changes. Companion files and full runnable snapshot planner are provided in the user-facing 2023–2025 odds-feasibility package.

## Ranked data candidates

1. **The Odds API, https://the-odds-api.com/** (with hyphens). Its NCAAF historical featured-market archive starts in 2020; five-minute snapshots since September 2022. Historical requires payment. November 2026 advertised plans: 20,000 credits/$30 per month, 100,000/$59, 5M/$119. Historical requests cost 10 credits * region * market. A `regions=us,markets=totals` request costs up to 10 credits and returns the available upcoming NCAAF events at the snapshot time. Most promising documented 2023–25 source. https://the-odds-api.com/historical-odds-data/ and https://the-odds-api.com/liveapi/guides/v4/
2. **Oddize, https://oddize.com/odds-api**. The published *NCAAF-specific* archive starts **August 2024**, not January 2024 (overall archive). Free developer tier offers 1,500 credits/month and historical event movement; advertised event odds history costs five credits. Actual 2024–25 event coverage, total/Under markets, book selection, and historical event-ID discovery on the free plan are **unverified** because no account/API key was provided. Try the predeclared 2024–25 cases free before purchasing anything.
3. **Scottfree Analytics**: $99 one-time advertised NCAAF opening/closing CSV; optional benchmark only. Opening/closing lines cannot stand in for 24h/6h/2h bookmaker quotes. https://shop.scottfreellc.com/shop/p/ncaaf-historical-odds-data

Unsuitable for 2023–25: OddsLink's *NCAAF-specific* coverage begins Sep 2026 (its general archive is older); The Odds Gap snapshots start Aug 2026; and https://theoddsapi.com/ (no hyphens, different provider) records historical odds from May 2026. Similar vendor names are not the same service.

## Empirical request budget — frozen project dataset

Calculated from the Nov 7, 2026 `cfb-live-training-data` modeling_dataset.csv artifact (SHA256 `a2f5a61248afe895e39bde58c5d3f8e7d10ed490095418e45d69ff0bbc7b3678`), selecting unique FBS-vs-FBS games in 2023–25. **2,265 games**: 750 in 2023, 753 in 2024, 762 in 2025. This is the universe of possible game lookups, **not proof any sportsbook carries all games**. One query snapshot may serve multiple game/horizon decisions. All requested times are floored to the specified UTC bucket, never pushed forward.

| Coverage / query bucket | Distinct requests | Max credits | Smallest single-month advertised plan |
| --- | ---: | ---: | ---: |
| Fixed 12-game pilot, 24h/6h/2h, 5m | 33 | 330 | $30 |
| All 2,265 games, 24h only, 5m | 866 | 8,660 | $30 |
| All 2,265 games, 24h/6h/2h, 5m | 2,259 | 22,590 | $59 |
| All 2,265 games, 24h/6h/2h, 30m | 2,076 | 20,760 | $59 |
| All 2,265 games, 24h/6h/2h, 60m | 1,536 | 15,360 | $30 |
| All 2,265 games, 48h/24h/6h/2h, 5m | 2,934 | 29,340 | $59 |

The 60-minute alternative can sample **up to 59 minutes before** the actual target and is not an equal-fidelity substitute; bookmaker quotes may be older still. Counts assume all nonempty calls cost 10 credits, without retries or provider coverage filtering. A $30 plan is a subscription, not a per-request charge. Don't buy before checking sample coverage.

## Weather expansion

The previous fixed 12-game HRRR pilot reconstructed 36/36 nominally feasible 24h/6h/2h forecast vintages; HRRR fails its conservative 48h cutoff by design. NOAA's **GFS quarter-degree archive from June 2019 onward explicitly includes 2023–25**. This is a plausible no-data-fee source for a separate 48h forecast-reconstruction test, but no GFS GRIB sample has yet been decoded for this study. GFS is a different model, so retain model source, initialization, lead, valid-time offset, and archive integrity hashes rather than passing it off as HRRR. For a full-game feature, sample forecast evolution over four or more hours, not only kickoff.

NOAA: https://www.ready.noaa.gov/archives.php ; GFS archive: https://gdex.k8s.ucar.edu/datasets/d084001/

## Next **smallest-cost** experimental gate

1. Use the previously fixed 12-game IDs (three seasons, LSU/Wisconsin/Colorado/Washington). For Oddize, first test eight games in 2024–25 at $0 if a free API account is made available; require actual quote observations (not just advertised coverage).
2. If not adequate, make **at most 33** historical snapshot requests for the fixed 12-game 24h/6h/2h schedule from The Odds API, only after explicit user authorization of its $30 monthly subscription.
3. Accept a quote only if the source snapshot is at/before the cutoff, its bookmaker-market timestamp is plausible and sufficiently fresh, the actual UNDER odds and total are present, both teams and kickoff match the verified CFBD game, and raw provenance is stored. Reject unmatched or stale quotes.
4. After an observed-coverage report, reconstruct a forecast-trained chronological challenger. **Do not apply postgame observed-weather HGB to new pregame forecast inputs and call the old historical ROI executable**.
5. Keep the immutable 2026 prospective ledger, production workflows, thresholds and website unmodified. No execution-verified historical ROI is available yet.

No operational file changes. This report concerns data feasibility and acquisition only.