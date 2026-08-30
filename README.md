# CFB Weather Totals Research

Research and prospective paper-tracking project for testing whether college-football weather and game context can identify **systematic market residuals** in totals.

The target is:

```text
market_residual = actual_total_points - closing_total
```

A negative predicted residual points toward an UNDER; a positive residual points toward an OVER. The project is not trying to predict scores from scratch. It starts with the sportsbook total and asks whether the game may finish materially above or below that market expectation.

Live site: https://jaczerin24.github.io/cfb-weather-totals/

## Current status

The project is in active **2026 in-season prospective validation**.

- Historical research covers 2014-2025 regular seasons.
- The general live model uses the existing HGB residual workflow.
- FCS-vs-FCS games use a separate FCS-only HGB research model.
- Current production research rules are UNDER-only; no operational OVER rule has validated.
- General QUALIFIES require a model under edge of at least 3.5 points and a market total of at least 56.
- FCS RESEARCH QUALIFIES require a model under edge of at least 7.5 points and a market total of at least 56.
- The legacy two-leg card uses only the two strongest eligible general qualifiers and never forces a card.
- Every official 2026 entry is selected from immutable scheduled snapshots before kickoff and later graded against the final score.
- Near-kickoff market captures are stored separately for CLV tracking.

The 2026 prospective protocol is versioned in `config/prospective_protocol_2026.yml`. Protocol 2026.2 changes only close-capture reliability; the frozen model thresholds and official-entry rules are unchanged.

### 2026 validation snapshot

As of **August 30, 2026**, the prospective ledger contains:

- 6 immutable official-board snapshots;
- 4 immutable near-kickoff close captures;
- 376 official game entries selected from eligible scheduled snapshots;
- 1 qualifying entry, from the FCS-only track;
- 1 settled qualifier, graded 1-0;
- 0 general-track qualifying entries so far;
- 0 settled qualifiers with a valid CLV benchmark so far.

This is far too little prospective evidence to evaluate the frozen strategy. The dated counts above are only a project-status snapshot. The current source of truth is [`outputs/prospective/2026/prospective_summary.md`](outputs/prospective/2026/prospective_summary.md).

## Research status

The historical work supports **continued prospective testing**, not a claim that the model broadly predicts totals better than the market.

- The useful historical profile is selective: stronger HGB UNDER signals, especially in higher-total games, have performed better than the broad model population.
- The general residual model does not beat the zero-residual market baseline on unconditional MAE, so the research question is whether it can identify useful conditional subsets rather than forecast every game's residual more accurately than the market.
- Historical threshold and filter exploration creates post-selection uncertainty. Attractive backtest rows are treated as hypotheses requiring chronological, robustness, and prospective validation rather than as confirmed edges.
- The FCS track has substantially less evidence than the general track. Its current historical screen is promising enough to paper-track, but the sample is small and the FCS model's unconditional MAE is worse than the zero-residual baseline in each available walk-forward test season.
- No OVER strategy is operational. Positive predicted residuals can be retained as research context, but they remain `NO PLAY` under the frozen 2026 protocol.
- Experimental work performed during the 2026 season is kept separate from the official 2026 prospective validation and cannot retroactively change recorded entries.

Useful research summaries on the current main branch include:

- [`outputs/deep_research_summary.md`](outputs/deep_research_summary.md) - historical data quality, weather groups, simple rules, and baseline walk-forward diagnostics.
- [`outputs/model_bakeoff_summary.md`](outputs/model_bakeoff_summary.md) - chronological model-family comparison.
- [`outputs/model_edge_validation_summary.md`](outputs/model_edge_validation_summary.md) - edge stability by season, recent period, provider, total range, and weather regime.
- [`outputs/edge_refinement_summary.md`](outputs/edge_refinement_summary.md) - stability-first screening of HGB UNDER refinements and interaction hypotheses.
- [`outputs/fcs_research_summary.md`](outputs/fcs_research_summary.md) - dedicated FCS-vs-FCS model results, threshold sensitivity, and uncertainty guardrails.
- [`outputs/prospective/2026/prospective_summary.md`](outputs/prospective/2026/prospective_summary.md) - current official 2026 prospective results and CLV summary.

Research code or diagnostics in open branches/pull requests should be treated as experimental until reviewed and intentionally merged. They do not alter the frozen production protocol merely by existing.

## In-season automation

The main automated pieces are:

- `.github/workflows/weekly-cfb-weather.yml` - builds the live board and creates official-entry snapshots on the frozen Thursday, Friday, and Saturday schedules.
- `.github/workflows/prospective-close-capture.yml` - captures near-kickoff market totals for CLV without allowing manual backfill.
- `.github/workflows/prospective-grade.yml` - rebuilds the ledger and grades completed games on safe postgame mornings.
- `.github/workflows/deploy-pages.yml` - publishes the generated site from `docs/`.
- `.github/workflows/prospective-ledger-tests.yml` - validates the frozen protocol, selection behavior, cadence, and immutable-file rules.

See [`documentation/IN_SEASON_OPERATIONS.md`](documentation/IN_SEASON_OPERATIONS.md) for the full lifecycle of a game from live-board generation through grading.

## Repository map

- `src/` - model, data, live-board, odds, NWS, research, and prospective-ledger code.
- `config/` - model/settings configuration and the frozen 2026 prospective protocol.
- `data/raw/` - locally/generated raw historical inputs; large data are not committed.
- `data/processed/` - generated model-training inputs; live training data are restored through GitHub Actions artifacts.
- `outputs/` - generated research, live-board, cache, and prospective-ledger products.
- `outputs/prospective/2026/` - immutable board snapshots, immutable close captures, manifest, official entries, grades, and prospective summaries.
- `docs/` - generated GitHub Pages site. Do not treat this as hand-maintained source code.
- `documentation/` - human-maintained project documentation.
- `.github/workflows/` - automation, validation, grading, and deployment.

See [`documentation/REPOSITORY_MAP.md`](documentation/REPOSITORY_MAP.md) for a more detailed guide.

## Core data sources

- CollegeFootballData (CFBD) - games, historical/live totals, team/venue context, and historical source data.
- National Weather Service - live kickoff weather for outdoor games.
- OddsPapi - primary FCS fallback when CFBD does not have a usable FCS total.
- The Odds API - secondary FCS fallback.

The live market pipeline preserves provider count, minimum, maximum, median, and range when available so line-source uncertainty can be audited.

## API keys

GitHub Actions uses repository secrets as needed:

- `CFBD_API_KEY`
- `ODDSPAPI_API_KEY`
- `ODDS_API_KEY`
- `NOAA_API_TOKEN`

For local development, copy `.env.example` to `.env` and add the keys you intend to use.

## Local setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Historical research workflow

A typical full historical build uses the scripts in `src/` to pull source data, build the modeling dataset, run walk-forward research, validate model edges, and generate research dashboards. The dedicated GitHub Actions historical workflow is `.github/workflows/manual-historical-research.yml`.

Historical outputs are intentionally kept separate from the frozen 2026 prospective ledger. Research changes must not rewrite an already-recorded official entry.

## 2026 prospective integrity rules

The prospective system is designed to make hindsight difficult:

- official entries come only from the declared scheduled board snapshots;
- the selected snapshot must be at least 120 minutes before kickoff;
- reruns of the same scheduled model run cannot opportunistically replace the first successful attempt;
- push/manual website refreshes can be archived but cannot become official entries;
- near-kickoff close captures are schedule-only and first-attempt-only;
- missing close captures stay missing rather than being reconstructed after the result is known;
- immutable CSV filenames contain a SHA-256 content-hash prefix and are verified when the ledger rebuilds;
- postgame grades are derived from the immutable entry plus final score, not from the current live board;
- production thresholds are not retuned from 2026 outcomes before the declared review point except for documented data-integrity fixes.

## Interpretation

This remains a research and paper-tracking project. Historical threshold selection contains post-selection uncertainty, the broad model does not beat the market total on unconditional MAE, and the FCS track has substantially more validation uncertainty than the general track. A qualifier is therefore a rule-based research signal, not proof of a causal weather effect or a guaranteed profitable wager.
