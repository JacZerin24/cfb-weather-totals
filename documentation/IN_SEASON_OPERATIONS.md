# 2026 In-Season Operations

This document describes the live 2026 workflow from a scheduled board refresh through prospective grading. The prospective system is designed so an already-recorded entry cannot be replaced after kickoff based on what happened in the game.

## 1. Historical training state

The weekly workflow does not rebuild the full 2014-2025 historical dataset from scratch every time. It restores the latest non-expired `cfb-live-training-data` GitHub Actions artifact. If that artifact does not yet exist, the workflow seeds it from the latest successful manual historical-research artifact.

The primary training file is `data/processed/modeling_dataset.csv`. Team prior-feature data are restored when available for the general model. The dedicated FCS model uses its own market/weather/game-context feature set.

## 2. Scheduled live-board refresh

`.github/workflows/weekly-cfb-weather.yml` builds the live board on these Central-time schedules:

- Monday 5:17 AM: operational early look only
- Thursday 5:17 AM: safety snapshot
- Thursday 8:17 AM: freshness snapshot
- Friday 5:17 AM: safety snapshot
- Friday 8:17 AM: freshness snapshot
- Saturday 1:17 AM: the single normal Saturday full build

Paired UTC cron expressions plus an `America/Chicago` offset gate preserve these local clock times across CDT and CST. Monday remains outside the official-entry eligible cron list.

Push/manual refreshes may rebuild the website, but they are not eligible to create official prospective entries. The watchdog checks Monday, Thursday, and Friday 30 minutes after the 5:17 AM safety build. On Saturday it checks at 6:47 AM Central that the 1:17 AM full build succeeded. If no qualifying full build exists, it dispatches one operational backup. A watchdog backup refreshes operations but remains ineligible for official entry.

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

There is no operational OVER strategy. A large positive model residual can appear as research context but remains `NO PLAY` under the current method.

## 6. Live status labels

The board can show:

- `QUALIFIES` / `FCS RESEARCH QUALIFIES`: meets the frozen straight-under research rule.
- `LEAN`: meaningful model-under signal that misses the full qualifying rule.
- `WATCH`: forecast or kickoff-time limitation prevents a clean decision.
- `NO PLAY`: does not meet the current under rule or points to an unsupported OVER.
- `NO LINE`: no current market total is available.
- `POSTPONED / STALE`: website-only status used when the preserved prospective snapshot no longer represents the current event context.

The website is a presentation layer. The prospective ledger uses archived board data, not whatever the current website happens to show later.

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

The filename contains the first 12 characters of the SHA-256 hash of the CSV contents. The manifest recalculates the hash and rejects silently altered immutable files.

## 9. Selecting the official entry

For each game, the derived ledger normally uses the latest eligible scheduled snapshot at least 120 minutes before kickoff. For repeated attempts of one scheduled GitHub run, the first successful attempt wins. Push, manual, and watchdog-dispatched refreshes do not become official entries.

Protocol 2026.4 adds one narrowly scoped integrity correction. The Sep. 5, 2026 Saturday run `33961120017` was the intentionally deployed 1:17 AM CDT production build, but its immutable snapshot was stamped `official_eligible=False` because the protocol configuration still listed the superseded Saturday schedules. The immutable file is not changed. `src/prospective_integrity_rebuild.py` applies the documented run-level metadata correction in memory and then performs normal latest-eligible selection across the entire preserved run. This prevents cherry-picking an individual game after the result.

Future Saturday snapshots are stamped correctly because the active protocol now exactly mirrors the production workflow's 1:17 AM Central CDT/CST crons.

## 10. Near-kickoff market capture and CLV

`.github/workflows/prospective-close-capture.yml` records market totals separately from model-entry snapshots.

Protocol 2026.2 increased capture reliability to twice per hour while keeping the official 90-minute pre-kickoff window, schedule-only rule, and first-run-attempt requirement. The ledger independently re-checks those restrictions before selecting a CLV benchmark.

The preferred benchmark is the median across available books, with the selected current total as fallback when a median is unavailable. Missing CLV remains missing and is never reconstructed after the game.

For an UNDER:

```text
CLV points = entry total - benchmark close total
```

Positive CLV means the entry captured a higher total than the later benchmark.

## 11. Postgame grading

`.github/workflows/prospective-grade.yml` rebuilds derived grades at 12Z Sunday, Monday, and Tuesday. It may also be manually dispatched because a rebuild changes only derived products from preserved immutable inputs.

The canonical automated rebuild is `src/prospective_integrity_rebuild.py`.

Two protections now apply before an entry can affect performance statistics:

1. CFBD must explicitly mark the game `completed=true`; an incomplete/postponed 0-0 placeholder cannot be settled.
2. A documented event-level integrity exclusion can preserve the original prospective entry while excluding it from wins, losses, units, ROI, and CLV.

The Western Carolina at Campbell game (`401866625`) is the first such exclusion. Its original Saturday qualifier remains auditable, but the game was postponed to a different kickoff/weather context and is labeled `POSTPONED / STALE` for grading rather than being counted as a model result.

At -110 paper pricing:

- win: +0.9090909091 units
- loss: -1 unit
- push: 0 units
- excluded/stale: no units and no performance CLV

## 12. Derived ledger products

`outputs/prospective/2026/` contains:

- `snapshots/`: immutable full-board captures
- `close_captures/`: immutable near-kickoff market captures
- `manifest.csv`: hashes and sizes of immutable files
- `official_entries.csv`: one selected official entry per eligible game
- `graded_entries.csv`: official entries plus final scores, result, units, CLV, and grading disposition
- `summary.csv`: aggregate prospective results including settled/excluded/pending qualifier counts
- `prospective_summary.md`: human-readable current summary

The immutable snapshot/capture files are the source of truth. The derived CSV/Markdown files can be rebuilt.

## 13. Research-only shadow evaluations

Orientation/crosswind and joint-core challengers remain isolated from production. Their evaluation schedules mirror the production official-entry timing. The same preserved Sep. 5 run-level eligibility correction is applied by `src/shadow_integrity_rebuild.py` so the research evaluators do not silently use an older Friday snapshot when the real Saturday production run existed.

Their graders already require completed games. Shadow failures remain `continue-on-error` and cannot block the official production ledger.

## 14. Validation and change control

`.github/workflows/prospective-ledger-tests.yml` runs on relevant pull requests and main-branch changes. The self-test checks:

- frozen production thresholds still match code;
- official cron lists match the weekly workflow;
- the Sep. 5 run-level correction is narrow and does not make manual snapshots eligible;
- close captures remain schedule-only and first-attempt-only;
- incomplete 0-0 games cannot grade;
- postponed/stale exclusions do not contribute units/ROI/CLV;
- shadow evaluations receive the same documented schedule correction;
- immutable writer behavior remains append-only.

Protocol 2026.4 does not change model science. Production thresholds, direction rules, minimum lead time, pricing assumptions, and historical research conclusions remain frozen through the declared review point unless a separate evidence-backed, versioned decision is made.
