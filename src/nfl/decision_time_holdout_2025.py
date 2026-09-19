from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ..utils import ensure_dir, read_df, write_df
from .model_bakeoff import BREAKEVEN, _prep
from .forecast_native_bakeoff import (
    DATA_PATH,
    FIXED_CANDIDATES,
    _feature_columns,
    _weather_features,
)
from .roi_search import _deep_models


MARKET_DB = Path(os.getenv('NFL_MARKET_DB', '/tmp/nfl_odds.duckdb'))
LEADS = (24, 48, 72)
MAX_SNAPSHOT_STALENESS_HOURS = 14.0

TEAM_NAMES = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs',
    'LA': 'Los Angeles Rams',
    'LAC': 'Los Angeles Chargers',
    'LV': 'Las Vegas Raiders',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NO': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SEA': 'Seattle Seahawks',
    'SF': 'San Francisco 49ers',
    'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders',
}


def _american_profit(odds: float) -> float:
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def _load_market_rows() -> pd.DataFrame:
    if not MARKET_DB.exists():
        raise FileNotFoundError(f'Market database not found: {MARKET_DB}')
    conn = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "select table_name from information_schema.tables "
                "where table_schema='main'"
            ).fetchall()
        }
        if 'raw_odds' not in tables:
            raise RuntimeError(
                f'Expected raw_odds table; found {sorted(tables)[:20]}'
            )
        query = """
            select
                captured_at,
                commence_time as game_start_time,
                event_id,
                home_team,
                away_team,
                bookmaker_title as sportsbook,
                outcome_name as outcome,
                outcome_price as price,
                outcome_point as line
            from raw_odds
            where market_key = 'totals'
              and captured_at < commence_time
              and commence_time >= timestamp '2025-09-01'
              and commence_time < timestamp '2026-02-15'
        """
        rows = conn.execute(query).df()
    finally:
        conn.close()

    rows['captured_at'] = pd.to_datetime(
        rows['captured_at'], utc=True, errors='coerce'
    )
    rows['game_start_time'] = pd.to_datetime(
        rows['game_start_time'], utc=True, errors='coerce'
    )
    rows['line'] = pd.to_numeric(rows['line'], errors='coerce')
    rows['price'] = pd.to_numeric(rows['price'], errors='coerce')
    return rows.dropna(
        subset=['captured_at', 'game_start_time', 'line', 'price']
    )


def _consensus_snapshot(
    event_rows: pd.DataFrame,
    target: pd.Timestamp,
) -> dict | None:
    eligible = event_rows[event_rows['captured_at'] <= target].copy()
    if eligible.empty:
        return None
    captured = eligible['captured_at'].max()
    staleness = (target - captured).total_seconds() / 3600.0
    if staleness > MAX_SNAPSHOT_STALENESS_HOURS:
        return None

    snap = eligible[eligible['captured_at'].eq(captured)].copy()
    over = snap[snap['outcome'].astype(str).str.lower().eq('over')].copy()
    if over.empty:
        return None

    counts = over['line'].value_counts(dropna=True)
    if counts.empty:
        return None
    max_count = counts.max()
    candidates = [float(v) for v in counts[counts.eq(max_count)].index]
    market_median = float(over['line'].median())
    selected_line = min(
        candidates,
        key=lambda value: (abs(value - market_median), value),
    )
    at_line = over[np.isclose(over['line'], selected_line)].copy()
    if at_line.empty:
        return None

    price_median = float(at_line['price'].median())
    idx = (at_line['price'] - price_median).abs().idxmin()
    consensus_price = float(at_line.loc[idx, 'price'])
    best_price = float(at_line['price'].max())

    return {
        'decision_capture_at': captured,
        'decision_target_at': target,
        'decision_staleness_hours': float(staleness),
        'decision_total': float(selected_line),
        'consensus_over_price': consensus_price,
        'best_over_price_same_line': best_price,
        'sportsbooks_at_selected_line': int(at_line['sportsbook'].nunique()),
        'sportsbooks_in_snapshot': int(over['sportsbook'].nunique()),
        'market_total_min': float(over['line'].min()),
        'market_total_max': float(over['line'].max()),
        'market_total_median': market_median,
    }


def _build_market_snapshots(
    games: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    event_groups = {
        (home, away): group
        for (home, away), group in market.groupby(
            ['home_team', 'away_team'], sort=False
        )
    }
    rows: list[dict] = []
    for _, game in games.iterrows():
        home_name = TEAM_NAMES.get(str(game['home_team']))
        away_name = TEAM_NAMES.get(str(game['away_team']))
        if not home_name or not away_name:
            continue
        event = event_groups.get((home_name, away_name))
        if event is None or event.empty:
            continue

        expected_kickoff = pd.to_datetime(
            game.get('kickoff_utc'), utc=True, errors='coerce'
        )
        event = event.copy()
        if pd.notna(expected_kickoff):
            event['kickoff_distance_hours'] = (
                event['game_start_time'] - expected_kickoff
            ).abs().dt.total_seconds() / 3600.0
            nearest_start = (
                event.sort_values('kickoff_distance_hours')
                ['game_start_time']
                .iloc[0]
            )
            event = event[event['game_start_time'].eq(nearest_start)].copy()
            if event['kickoff_distance_hours'].min() > 6:
                continue

        starts = event['game_start_time'].dropna()
        if starts.empty:
            continue
        kickoff = starts.mode().iloc[0]

        for lead in LEADS:
            target = kickoff - pd.Timedelta(hours=lead)
            snapshot = _consensus_snapshot(event, target)
            if snapshot is None:
                continue
            rows.append({
                'game_id': game['game_id'],
                'season': int(game['season']),
                'week': int(game['week']),
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'lead_hours': lead,
                'external_game_start_time': kickoff,
                **snapshot,
            })
    return pd.DataFrame(rows)


def _price_roi(frame: pd.DataFrame, column: str) -> tuple[int, float]:
    residual = pd.to_numeric(frame['decision_market_residual'], errors='coerce')
    prices = pd.to_numeric(frame[column], errors='coerce')
    valid = residual.notna() & prices.notna() & residual.ne(0)
    units = 0.0
    for idx in frame.index[valid]:
        units += (
            _american_profit(float(prices.loc[idx]))
            if residual.loc[idx] > 0
            else -1.0
        )
    graded = int(valid.sum())
    return graded, units / graded if graded else np.nan


def _grade(frame: pd.DataFrame) -> dict:
    residual = pd.to_numeric(
        frame['decision_market_residual'], errors='coerce'
    )
    wins = int((residual > 0).sum())
    losses = int((residual < 0).sum())
    pushes = int((residual == 0).sum())
    graded = wins + losses
    flat_units = wins * (100 / 110) - losses
    consensus_n, consensus_roi = _price_roi(
        frame, 'consensus_over_price'
    )
    best_n, best_roi = _price_roi(
        frame, 'best_over_price_same_line'
    )
    return {
        'games': int(len(frame)),
        'graded': graded,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / graded if graded else np.nan,
        'flat_roi_minus110': flat_units / graded if graded else np.nan,
        'consensus_price_graded': consensus_n,
        'consensus_price_roi': consensus_roi,
        'best_price_graded': best_n,
        'best_price_roi': best_roi,
        'p_value_vs_minus110': (
            binomtest(
                wins,
                graded,
                p=BREAKEVEN,
                alternative='greater',
            ).pvalue
            if graded else np.nan
        ),
        'avg_snapshot_staleness_hours': float(
            frame['decision_staleness_hours'].mean()
        ) if len(frame) else np.nan,
    }


def _predict_holdout(
    historical: pd.DataFrame,
    snapshots: pd.DataFrame,
    lead: int,
) -> pd.DataFrame:
    work = _weather_features(historical, lead)
    work = work[work[f'forecast_complete_{lead}h'].fillna(False)].copy()

    train = work[work['season'] < 2025].copy()
    test = work[work['season'].eq(2025)].copy()
    snap = snapshots[snapshots['lead_hours'].eq(lead)].copy()
    test = test.merge(
        snap.drop(
            columns=[
                'season', 'week', 'home_team', 'away_team',
            ],
            errors='ignore',
        ),
        on='game_id',
        how='inner',
    )
    if test.empty:
        return test

    test['original_closing_total'] = test['closing_total']
    test['closing_total'] = pd.to_numeric(
        test['decision_total'], errors='coerce'
    )
    test['decision_market_residual'] = (
        pd.to_numeric(test['actual_total_points'], errors='coerce')
        - test['closing_total']
    )
    test['total_bin'] = pd.cut(
        test['closing_total'],
        bins=[0, 38, 42, 46, 50, 54, 100],
        labels=['<=38', '38-42', '42-46', '46-50', '50-54', '54+'],
    )

    combined = pd.concat([train, test], ignore_index=True, sort=False)
    nums, cats = _feature_columns(combined, lead, True)
    train_work = _prep(train, nums, cats)
    test_work = _prep(test, nums, cats)

    reg = _deep_models(nums, cats)['reg_rf_leaf25'][1]
    cls = _deep_models(nums, cats)['cls_rf_leaf25'][1]
    reg.fit(train_work[nums + cats], train_work['market_residual'])

    cls_train = train_work[train_work['market_residual'] != 0].copy()
    cls.fit(
        cls_train[nums + cats],
        (cls_train['market_residual'] > 0).astype(int),
    )

    out = test.copy()
    out['reg_signal'] = reg.predict(test_work[nums + cats])
    out['over_probability'] = cls.predict_proba(
        test_work[nums + cats]
    )[:, 1]
    return out


def main() -> None:
    historical = read_df(DATA_PATH).copy()
    historical['season'] = pd.to_numeric(
        historical['season'], errors='coerce'
    )
    games_2025 = historical[historical['season'].eq(2025)].copy()

    market = _load_market_rows()
    snapshots = _build_market_snapshots(games_2025, market)
    write_df(
        snapshots,
        'outputs/nfl/decision_time_market_snapshots_2025.csv',
    )

    rows: list[dict] = []
    play_parts: list[pd.DataFrame] = []

    for lead in LEADS:
        predicted = _predict_holdout(historical, snapshots, lead)
        if predicted.empty:
            continue
        for name, (reg_threshold, prob_threshold) in FIXED_CANDIDATES.items():
            plays = predicted[
                (predicted['reg_signal'] >= reg_threshold)
                & (predicted['over_probability'] >= prob_threshold)
            ].copy()
            rows.append({
                'lead_hours': lead,
                'candidate': name,
                'reg_threshold': reg_threshold,
                'classifier_probability': prob_threshold,
                **_grade(plays),
            })
            if not plays.empty:
                plays['candidate'] = name
                plays['lead_hours'] = lead
                play_parts.append(plays)

    summary = pd.DataFrame(rows)
    write_df(summary, 'outputs/nfl/decision_time_holdout_2025.csv')

    if play_parts:
        plays = pd.concat(play_parts, ignore_index=True)
        keep = [
            c for c in [
                'game_id', 'week', 'gameday', 'home_team', 'away_team',
                'lead_hours', 'candidate',
                'decision_capture_at', 'decision_target_at',
                'decision_staleness_hours', 'decision_total',
                'consensus_over_price', 'best_over_price_same_line',
                'sportsbooks_at_selected_line', 'sportsbooks_in_snapshot',
                'market_total_min', 'market_total_max',
                'original_closing_total', 'actual_total_points',
                'decision_market_residual',
                'reg_signal', 'over_probability',
                'decision_roof',
            ]
            if c in plays.columns
        ]
        write_df(
            plays[keep],
            'outputs/nfl/decision_time_holdout_2025_plays.csv',
        )

    coverage = (
        snapshots.groupby('lead_hours', as_index=False)
        .agg(
            matched_games=('game_id', 'nunique'),
            avg_staleness_hours=('decision_staleness_hours', 'mean'),
            median_sportsbooks=('sportsbooks_in_snapshot', 'median'),
        )
    )
    write_df(
        coverage,
        'outputs/nfl/decision_time_holdout_2025_coverage.csv',
    )

    lines = [
        '# NFL 2025 Decision-Time Holdout',
        '',
        'This is the first test in the project where both the weather input and the market total are taken from information captured before kickoff.',
        '',
        '## Market source',
        '',
        '- Public nfl-market-movement-tracker DuckDB release built from The Odds API.',
        '- The source project captured NFL markets four times per day across 30+ operators during the 2025 season.',
        '- For each 24h/48h/72h target, this test uses the latest available capture at or before the target and rejects captures more than 14 hours stale.',
        '- Consensus total is an actually offered line: the most common line across books, with ties resolved nearest the market median.',
        '- ROI is shown at flat -110, a representative consensus Over price at that line, and the best available Over price at the same line.',
        '',
        '## Frozen model',
        '',
        '- Training seasons: 2018-2024 only.',
        '- Weather features: archived JMA GSM forecasts available at the matching lead time.',
        '- Model architecture: RF leaf-25 residual regression + RF leaf-25 over classifier.',
        '- Rules: >=3.0 residual points + >=60% Over probability, and >=4.0 + >=60%.',
        '- No 2025 result is used to select the model or threshold.',
        '',
        '## Results',
        '',
        summary.to_markdown(index=False) if not summary.empty else '_No qualifying results._',
        '',
        '## Snapshot coverage',
        '',
        coverage.to_markdown(index=False) if not coverage.empty else '_No market snapshots matched._',
        '',
        '## Remaining limitation',
        '',
        '- Historical training targets are residuals versus the final closing total because multi-season pre-kickoff market snapshots are not currently available in this repository. The 2025 deployment test feeds the true decision-time total to a model trained on historical closing-market residuals. That is substantially more realistic than the earlier tests, but not perfectly symmetric training.',
        '- A future paid historical-odds archive or additional public seasons would allow the entire training history to use time-matched market lines.',
    ]
    out = ensure_dir('outputs/nfl') / 'decision_time_holdout_2025.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote 2025 decision-time holdout outputs under {out.parent}')


if __name__ == '__main__':
    main()
