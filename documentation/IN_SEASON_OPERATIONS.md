# 2026 In-Season Operations

This document describes the live 2026 workflow from a scheduled board refresh through prospective grading. The prospective ledger is designed so an already-recorded entry cannot be replaced after kickoff based on what happened in the game.

## 1. Historical training state

The weekly workflow does not rebuild the full 2014-2025 historical dataset from scratch every time. It restores the latest non-expired `cfb-live-training-data` GitHub Actions artifact. If that artifact does not yet exist, the workflow seeds it from the latest successful manual historical-research artifact.

The primary training file is `data/processed/modeling_dataset.csv`. Team prior-feature data are restored when available for the general model. The dedicated FCS model uses its own market/weather/game-context feature set.

## 2. Scheduled live-board refresh

`.github/workflows/weekly-cfb-weather.yml` builds the live board on the frozen prospective schedules:

- Thursday 14Z
- Friday 14Z
- Saturday 13Z

Push/manual-style refreshes may also rebuild the website, but only the declared scheduled snapshots can become official prospective entries.

`src/run_live_week.py` determines the live season/week slate and keeps games visible even when they do not yet have a market total.

## 3. Market totals

CFBD is the primary live line source. The live pipeline preserves provider context when available, including provider count, minimum total, maximum total, median total, range, and the selected total relative to the market median.

For FCS-vs-FCS games without a usable CFBD total:

1. OddsPapi is the primary fallback.
2. The Odds API is the secondary fallback.

The OddsPapi production source list is intentionally constrained so alternate ladders do not dominate the consensus. Each book contributes one vote to the consensus center.

## 4. Kickoff weather

`src/nws_forecast.py` uses the National Weather Service point/grid forecast for the game venue and kickoff time.

For outdoor games the board can include temperature, dew point, relative humidity, wind, gusts, precipitation probability, quantitative precipitation, snowfall, and a short weather summary. Indoor/dome games are marked indoor rather than being assigned synthetic outdoor weather.

NWS point/grid responses are cached in `outputs/nws_grid_cache.json` to reduce unnecessary repeat calls.

## 5. Model scoring

The project predicts a market residual rather than a score from scratch:

```text
predicted market residual = predicted actual total - current market total
```

A negative value points UNDER; a positive value points OVER.

### General track

The general live workflow uses the HGB residual model and the frozen production research screen:

- UNDER direction only
- edge at least 3.5 points
- market total at least 56
- usable NWS forecast or indoor designation
- known kickoff time

### FCS track

FCS-vs-FCS games use the separate FCS-only HGB model. The frozen research screen is:

- UNDER direction only
- edge at least 7.5 points
- market total at least 56
- usable NWS forecast or indoor designation
- known kickoff time

FCS is labeled `FCS RESEARCH QUALIFIES` on the public site to distinguish its higher validation uncertainty. The underlying prospective status remains `QUALIFIES` so the frozen ledger logic is unchanged.

There is no operational OVER strategy. A large positive model residual can still appear as useful research context, but it remains `NO PLAY` under the current method.

## 6. Live status labels

The board can show:

- `QUALIFIES` / `FCS RESEARCH QUALIFIES`: meets the frozen straight-under research rule.
- `LEAN`: meaningful model-under signal that misses the full qualifying rule.
- `WATCH`: forecast or kickoff-time limitation prevents a clean decision.
- `NO PLAY`: does not meet the current under rule or points to an unsupported OVER.
- `NO LINE`: no current market total is available.

The website is a presentation layer. The prospective ledger uses the archived board data, not whatever the current website happens to show later.

## 7. Dashboard and staking helper

The weekly build generates the live dashboard in `docs/index.html` and related output files. `src/bankroll_helper.py` adds an optional flat-risk planning layer without changing model decisions.

The default prospective-conservative percentages are:

- 0.50% general straight
- 0.25% FCS research straight
- 0.25% validated two-leg general card

FCS can be set to 0% for paper-only tracking. The two-leg card remains general-track only and appears only when two eligible general qualifiers exist.

## 8. Immutable prospective board snapshots

After the scheduled board is built, `src/prospective_ledger.py snapshot-board` writes the full board to:

`outputs/prospective/2026/snapshots/`

The snapshot records the protocol version/hash, GitHub event, exact schedule string, run ID, run attempt, source commit SHA, game data, line state, model state, and weather state.

The filename contains the first 12 characters of the SHA-256 hash of the CSV contents. The manifest later recalculates the hash and refuses silently altered immutable files.

## 9. Selecting the official entry

When the ledger rebuilds, each game is evaluated across all archived board snapshots.

The official entry is the latest eligible scheduled snapshot that is at least 120 minutes before kickoff. For repeated attempts of the same scheduled GitHub run, the first successful attempt wins. Push/manual refreshes are not official entries.

This selection is performed from the immutable history, so a later website refresh does not replace an earlier official entry after the fact.

## 10. Near-kickoff market capture and CLV

`.github/workflows/prospective-close-capture.yml` records market totals separately from model-entry snapshots.

Protocol 2026.1 attempted this once per hour. After Week 1 exposed misses caused by GitHub schedule delays just outside the 90-minute window, protocol 2026.2 changed only capture reliability:

- attempts occur twice per hour at minute 15 and minute 45;
- the official capture window remains 90 minutes before kickoff;
- the workflow remains schedule-only;
- only GitHub run attempt 1 is allowed to capture;
- no postgame/manual backfill is allowed.

The ledger independently re-checks the capture window, scheduled-event status, and first-run-attempt rule before selecting a CLV benchmark, so those integrity rules do not rely only on the workflow definition.

The script writes a close-capture file only when at least one game starts within the next 90 minutes. Multiple pregame captures of the same game are allowed; the ledger selects the latest valid pre-kickoff benchmark. The preferred benchmark is the median across available books, with the selected current total as fallback when a median is unavailable.

For an UNDER:

```text
CLV points = entry total - benchmark close total
```

Positive CLV means the entry captured a higher total than the later benchmark.

Missing CLV is kept missing. It is not reconstructed after the game.

## 11. Postgame grading

`.github/workflows/prospective-grade.yml` rebuilds grades on safe postgame mornings at 12Z Sunday, Monday, and Tuesday.

The grader fetches final scores, combines them with the immutable official entry, and determines win/loss/push against the entry total. It also joins the selected near-kickoff benchmark when one exists and calculates CLV.

At -110 paper pricing:

- win: +0.9090909091 units
- loss: -1 unit
- push: 0 units

Grading is intentionally not performed continuously during live games.

## 12. Derived ledger products

`outputs/prospective/2026/` contains:

- `snapshots/`: immutable full-board captures
- `close_captures/`: immutable near-kickoff market captures
- `manifest.csv`: hashes and sizes of immutable files
- `official_entries.csv`: one selected official entry per eligible game
- `graded_entries.csv`: official entries plus final scores, result, units, and CLV
- `summary.csv`: aggregate prospective results
- `prospective_summary.md`: human-readable current summary

The immutable snapshot/capture files are the source of truth. The derived CSV/Markdown files can be rebuilt.

## 13. What can change during 2026

The model thresholds, official-entry timing, and research rules are frozen through the declared review point. A documented data-integrity correction can receive a new protocol version and apply prospectively.

Protocol 2026.2 is such an operational correction: it improves the chance that GitHub Actions records a valid close benchmark but does not change which games qualify, when official model entries are selected, how games are graded, or how the two-leg card is built.

Research branches can continue testing alternatives, but those experiments should not silently alter the frozen live rules.
