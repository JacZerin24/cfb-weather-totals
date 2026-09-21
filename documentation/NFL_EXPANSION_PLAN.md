# NFL Weather Totals Expansion Plan

## Non-negotiable isolation rule

The existing college-football pipeline is the protected production baseline.

NFL work must remain additive until it is independently validated:

- do not rename, move, or repurpose existing CFB source files;
- do not change the frozen CFB prospective protocol or historical outputs;
- do not make the CFB weekly workflow depend on NFL code;
- do not replace `docs/index.html` or the CFB dashboard generator during the research phase;
- use `data/nfl/`, `outputs/nfl/`, `src/nfl/`, and NFL-specific configuration/workflows for the new track.

The first branch for this work is `feature/nfl-weather-totals`, based on the current CFB production commit at the time the expansion began.

## Research target

Use the same defensible target as the CFB project:

```text
market_residual = actual_total_points - closing_total
```

This matters because raw statements such as "windy NFL games score fewer points" do not establish a betting edge. The closing market may already price that effect. The research question is whether weather and weather/game interactions explain a repeatable residual after the closing total is known.

## Phase 1: clean historical baseline

Primary source: nflverse schedule/game data maintained through the nflverse ecosystem. The schedule dataset provides final scores, closing total line, roof status, temperature, wind, stadium, and game context.

Initial research window: 2006-2025 NFL regular seasons.

Why 2006:

- it gives a long modern-era sample;
- the documented schedule dataset includes the needed market and weather fields in this era;
- it avoids mixing the first baseline with much older structural eras before we establish robustness.

Phase-1 outputs:

1. Data-quality counts by season.
2. Weather-bin residual summaries.
3. Simple under/over rule screens treated only as hypothesis generation.
4. Chronological expanding-window model bake-off.
5. Direct comparison of context-only models versus context-plus-weather models.

## Model families in the first bake-off

The first pass deliberately uses only dependencies already present in the repository:

- Ridge regression;
- Elastic Net;
- Random Forest;
- Extra Trees;
- HistGradientBoosting.

No model is presumed to be the NFL winner because HGB was useful in the CFB project. NFL has fewer games per season, more indoor games, and a highly efficient betting market, so regularization and chronological stability matter heavily.

Model selection should prioritize:

1. walk-forward MAE versus the zero-residual market baseline;
2. incremental value of weather features versus a context-only version;
3. season-to-season directional stability;
4. robustness across nearby edge thresholds;
5. adequate sample size;
6. later prospective/forecast-realistic confirmation.

## Phase 2: better historical weather

The nflverse schedule fields are an excellent discovery baseline but do not contain the full weather vector needed for the final system.

Add stadium-coordinate history and enrich outdoor/open-roof games with consistent hourly historical weather, including at minimum:

- sustained wind;
- wind gust;
- wind direction;
- temperature;
- dew point / humidity;
- precipitation amount/type;
- pressure;
- snow where applicable.

Open-Meteo's archive API can provide a consistent reanalysis path (ERA5/ERA5-Land), while forecast-archive products can be used for a later forecast-realistic validation period.

Important distinction: observed/reanalysis weather is appropriate for discovering physical effects. It is not sufficient evidence for a live betting model because a bettor only has a forecast before kickoff.

## Phase 3: football context controls

Once the weather-only residual question is understood, add pregame football context without leaking postgame information. Candidate sources/features include nflverse play-by-play-derived team efficiency known before the game:

- offensive/defensive EPA per play;
- success rate;
- pass rate / PROE;
- pace / seconds per play;
- explosive-play rates;
- sack/pressure proxies;
- field-goal tendency and kicker context;
- rest differential;
- travel/time-zone context if it proves useful.

Because the closing total already embeds substantial team information, context features must demonstrate incremental out-of-sample value rather than simply making the model more complex.

## Phase 4: forecast-realistic backtest

Before any NFL qualifier can be enabled, rebuild a recent-period test using weather information that would actually have been available at a fixed lead time (for example, Monday early look, Thursday refresh, and game-day snapshot).

This phase answers whether the model survives forecast error, especially for wind and precipitation.

## Phase 5: live NFL board

Live data plan:

- schedule/results: nflverse and/or a second live schedule source for redundancy;
- totals: The Odds API using sport key `americanfootball_nfl`, preserving bookmaker count/min/max/median/range similarly to CFB;
- weather: reuse the existing NWS kickoff-grid logic through an NFL-specific cache/output path;
- venue/roof: NFL stadium reference with explicit roof handling;
- scoring: selected NFL residual model only after research is frozen.

The live NFL board should write only NFL paths such as:

```text
outputs/nfl/weekly_board.csv
outputs/nfl/weekly_snapshot.json
outputs/nfl/prospective/<season>/...
docs/nfl/...
```

## Phase 6: one website, two modes

Do not alter the current CFB page until the NFL page independently renders and passes its own smoke tests.

Target presentation architecture:

```text
CFB/NCAA mode  -> existing CFB board and all current controls/features
NFL mode       -> NFL board with equivalent interaction patterns
```

The eventual selector should be a thin presentation-layer addition, not a rewrite of the CFB dashboard. The safest migration is:

1. build NFL page independently under `docs/nfl/`;
2. add NFL-specific site tests;
3. snapshot the existing CFB generated HTML behavior;
4. introduce a small sport selector/header shared by both pages;
5. verify the CFB generated board is otherwise byte/behavior compatible where practical;
6. only then merge the dual-mode presentation.

The NFL board can intentionally mirror useful CFB functionality: status filters, full slate, interactive map, weather details, model edge/projected total, research explanations, and historical research pages. NFL thresholds/status labels must remain NFL-specific rather than inheriting CFB values.

## Production gate

NFL production stays disabled until all of the following are true:

- historical data audit passes;
- chronological model bake-off is complete;
- any weather edge is robust to reasonable alternate bins/thresholds;
- context-only versus weather-added comparison supports the weather contribution;
- forecast-realistic validation is complete for a meaningful recent sample;
- a frozen NFL protocol is written before prospective tracking;
- NFL site/workflow tests pass without changing CFB outputs or behavior.
