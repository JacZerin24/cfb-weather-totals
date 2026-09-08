# Wind Gust Historical Source Feasibility

**Research branch:** `research/wind-gust-value`

**Status:** source/QC feasibility only. No model fitting and no production effect.

## Purpose

This note records the real-source smoke and geographic-pilot results from the wind-gust acquisition phase so that source decisions are based on observed API behavior rather than only documentation.

## Safety boundary

All tests in this note:

- ran only from the research branch;
- restored the existing historical modeling artifact without modifying it in the repository;
- wrote gust data/QC only inside the workflow workspace;
- uploaded workflow artifacts for inspection;
- did not commit data outputs;
- did not call `run_live_week`;
- did not fit any scoring model;
- did not change production features, thresholds, weekly boards, picks, shadows, or prospective ledgers.

## Attempt 1: ERA5 sustained wind + ERA5-Land gust

GitHub Actions run:

```text
34201984211
```

The first real-source smoke queried five venue-season groups at venue 36 for 2014-2018.

All ten Open-Meteo HTTP requests succeeded:

- five ERA5 sustained-wind/direction requests;
- five ERA5-Land gust requests.

However, the ERA5-Land responses produced **zero usable `wind_gusts_10m` values** after parsing, even though the requests returned HTTP 200 and hourly timestamps.

Observed result:

```text
27 rows with sustained wind
0 rows with gust
0 complete wind+gust rows
0 HTTP request failures
```

### Decision

Do **not** scale the `era5_era5_land` stack or use it for modeling in its present form.

An HTTP-success response is not sufficient source validation. The long-history mixed-source route is paused unless a separately verified source or API representation provides actual historical gust values.

## Attempt 2: same-model ECMWF IFS smoke

GitHub Actions run:

```text
34202217584
```

The workflow was changed to require actual non-null wind and gust values rather than accepting HTTP success alone.

Five venue-season groups at venue 36 were tested across 2017, 2018, 2019, 2021, and 2022.

Result:

```text
27 / 27 fetched game rows complete
0 request failures
0 negative gust spreads in this small smoke
0 component timestamp mismatches
0 component grid-point separations > 20 km
```

On the six smoke games with both CFBD and IFS sustained wind:

```text
Pearson r = 0.7903
mean IFS - CFBD = -2.45 mph
median IFS - CFBD = -1.00 mph
mean absolute difference = 2.68 mph
median absolute difference = 1.25 mph
```

### Decision

ECMWF IFS was viable enough to justify a geographically distributed pilot.

## Attempt 3: geographically distributed 2022 IFS pilot

GitHub Actions run:

```text
34202428519
```

A deterministic 5 x 6 latitude/longitude quantile grid was used to select geographically distributed outdoor FBS venues from the 2022 historical sample.

Pilot sample:

```text
148 games
29 venues
latitude range: 25.75 to 53.34
longitude range: -123.07 to -6.23
```

The wide range intentionally tests source behavior beyond one U.S. region and includes historical neutral/international venue coverage when present in the modeling artifact.

### Coverage

```text
148 / 148 games with sustained wind
148 / 148 games with gust
148 / 148 complete wind+gust rows
100.0% complete coverage
29 / 29 venue requests successful
0 request failures
0 missing coordinates in the pilot
0 partial rows
0 missing weather rows
```

### Basic QC

```text
negative wind rows: 0
negative gust rows: 0
negative gust-spread rows: 2 / 148 (1.35%)
gust > 100 mph: 0
gust spread > 50 mph: 0
component timestamp mismatches: 0
component grid separation > 20 km: 0
wind age > 90 minutes: 0
gust age > 90 minutes: 0
```

### IFS sustained wind vs CFBD wind

All 148 pilot games had both values available.

```text
Pearson r = 0.7398
mean IFS - CFBD = +0.08 mph
median IFS - CFBD = -0.10 mph
mean absolute difference = 2.76 mph
median absolute difference = 2.15 mph
```

This is sufficiently coherent for source feasibility, while still showing meaningful game-level differences that must remain controlled in any later model comparison.

### Pilot distributions

IFS sustained wind:

```text
mean = 7.62 mph
median = 6.50 mph
95th percentile = 16.06 mph
99th percentile = 20.93 mph
max = 27.50 mph
```

IFS gust:

```text
mean = 16.58 mph
median = 15.45 mph
95th percentile = 32.20 mph
99th percentile = 40.62 mph
max = 53.00 mph
```

Gust minus sustained wind:

```text
mean = 8.95 mph
median = 9.15 mph
95th percentile = 16.67 mph
99th percentile = 19.71 mph
max = 25.50 mph
min = -5.70 mph
```

## Negative gust-spread inspection

The two negative-spread rows were retained exactly as returned.

### FIU Stadium

```text
game_id = 401426584
kickoff = 2022-10-15 00:00 UTC
IFS sustained wind = 5.3 mph
IFS gust = 4.7 mph
gust spread = -0.6 mph
CFBD wind = 4.3 mph
```

### Aggie Memorial Stadium

```text
game_id = 401409237
kickoff = 2022-09-25 00:00 UTC
IFS sustained wind = 16.7 mph
IFS gust = 11.0 mph
gust spread = -5.7 mph
CFBD wind = 10.6 mph
```

Wind and gust used the same IFS grid point and the same returned timestamp in both cases.

These rows are not automatically treated as corrupt. Open-Meteo's ECMWF IFS documentation defines:

- 10 m wind speed as an instantaneous field;
- 10 m gust as the maximum 3-second wind over the preceding three hours.

Therefore `gust_mph - wind_mph` compares fields with different temporal support and can be negative when the instantaneous wind has increased relative to the preceding gust window representation.

### Consequence for feature semantics

`gust_spread_mph` should **not** be described as a literal same-moment gust factor.

If retained in later research, it should be described as:

```text
IFS preceding-3-hour gust maximum minus instantaneous 10-m wind at the aligned hour
```

Negative values must remain unaltered for the preregistered continuous-feature test. `gust_factor` remains diagnostic only.

## Source decision after QC phase

### Feasible source

For a first retrospective gust-value experiment, **ECMWF IFS from 2017 onward** is currently the defensible source because it provides:

- sustained wind;
- wind direction;
- gust;
- one explicit model source;
- common grid metadata;
- complete coverage in the geographic pilot.

### Blocked source

The proposed ERA5 + ERA5-Land long-history stack is **blocked** until its missing gust values are resolved or replaced by another verified historical source.

### Important limitation

IFS historical/reanalysis-style data still does not represent a forecast available at a fixed pregame issuance time. A positive retrospective experiment would show physical/statistical information value, not deployable betting value. Forecast-archive and prospective-shadow gates remain mandatory before any production discussion.

## Next safe engineering step

Do not launch the full 2017-2025 archive acquisition with the current one-venue-season-per-request strategy yet.

Open-Meteo's free API rate limits and call accounting make season-length requests expensive because date ranges longer than two weeks count as multiple call units. The next engineering step should therefore optimize acquisition by batching multiple venue coordinates and/or shorter reusable windows, then re-run coverage/QC before model fitting.

No gust model should be fit until that acquisition strategy is both source-safe and rate-limit-safe.
