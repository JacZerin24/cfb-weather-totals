# CFB Weather Totals Research

Research project for testing whether weather conditions influence college football totals **after accounting for the betting market**.

The main research question is not simply whether weather affects scoring. The main question is:

> Do weather conditions explain **actual total points minus the closing total** enough to create a repeatable edge?

This repository is designed to start as a research and paper-tracking project before any real-money decisions are considered.

## Current status

Phase 1 starter repo.

- Pull historical college football games from CollegeFootballData.
- Pull historical betting totals/lines from CollegeFootballData.
- Pull historical game weather from CollegeFootballData.
- Build a game-level modeling dataset.
- Research weather impacts on market residuals.
- Backtest simple totals strategies.
- Generate a weekly paper-tracking report.
- Run weekly through GitHub Actions once API secrets are configured.

## Important responsible-use note

This project is for research and paper tracking. It does not guarantee profit, and early model output should not be treated as betting advice. Sports betting has real financial risk. Start with historical validation and paper tracking before risking money.

## API keys

Create these as GitHub Actions repository secrets:

- `CFBD_API_KEY` — required for CollegeFootballData.
- `ODDS_API_KEY` — optional later if using live sportsbook odds.
- `NOAA_API_TOKEN` — optional later if using NOAA/NCEI historical data.

For local development, copy `.env.example` to `.env` and fill in your key.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and add your CollegeFootballData key.

## Basic workflow

### 1. Pull historical data

```bash
python -m src.pull_historical_games
python -m src.pull_historical_lines
python -m src.pull_historical_weather
```

### 2. Build the modeling dataset

```bash
python -m src.build_dataset
```

### 3. Research weather signals

```bash
python -m src.research_weather_edges
```

### 4. Run simple backtests

```bash
python -m src.backtest_totals
```

### 5. Generate weekly report skeleton

```bash
python -m src.predict_week
python -m src.report
```

## Core target variable

The most important column is:

```text
market_residual = actual_total_points - closing_total
```

Negative residuals mean the game went under market expectation. Positive residuals mean the game went over market expectation.

## Suggested preseason timeline

With about 11 weeks before Week 1, a realistic path is:

1. Weeks 1-2: Confirm historical data pulls and clean joins.
2. Weeks 3-4: Research wind, precipitation, cold, heat, and dome/outdoor splits.
3. Weeks 5-6: Build and validate simple betting-rule backtests.
4. Weeks 7-8: Add forecast-weather workflow and weekly report.
5. Weeks 9-10: Paper-test against preseason/Week 0 odds.
6. Week 11: Finalize Week 1 run and monitor outputs.

## Outputs

Generated files are written to `outputs/`.

- `outputs/research_summary.md`
- `outputs/backtest_summary.csv`
- `outputs/weekly_report.md`
- `outputs/weekly_picks.csv`

## Model philosophy

Start simple and interpretable:

- Compare actual total points to closing totals.
- Test weather bins and thresholds.
- Avoid overfitting.
- Use walk-forward validation by season.
- Track closing-line value and ROI separately.
- Treat parlays as experimental only.

## Straight bets vs parlays

The initial model focuses on straight totals. Parlays should only be considered after straight-bet edges are validated out-of-sample and correlation between legs is understood.
