from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .build_dataset import pick_total
from .cfbd_client import CFBDClient
from .predict_week import (
    add_live_categories,
    as_bool,
    classify_row,
    fetch_nws_weather,
    fit_and_score,
    line_market_context,
    merge_prior_team_features,
    normalize_games,
    normalize_lines,
    normalize_venues,
    research_tags,
    write_outputs,
)
from .utils import ROOT, ensure_dir, get_settings, write_df


def write_no_market_board(board: pd.DataFrame, season: int, week: int | None) -> None:
    out = board.copy()
    out['closing_total'] = np.nan
    out['status'] = 'NO LINE'
    out['decision_reason'] = 'No current market total is available.'
    out['research_tags'] = ''
    out['abs_pred_edge'] = np.nan
    out['pred_market_residual'] = np.nan
    out['model_projected_total'] = np.nan
    out['model_side'] = ''
    out['status_rank'] = 4
    keep = [c for c in [
        'status', 'decision_reason', 'research_tags', 'season', 'week', 'game_id', 'start_date', 'start_time_tbd',
        'away_team', 'home_team', 'venue', 'home_conference', 'away_conference',
        'home_classification', 'away_classification', 'closing_total', 'model_projected_total',
        'pred_market_residual', 'abs_pred_edge', 'model_side', 'status_rank',
    ] if c in out.columns]
    out = out[keep].head(100)
    write_df(out, 'outputs/weekly_board.csv')
    write_df(pd.DataFrame([{
        'status': 'no_current_lines',
        'note': 'The upcoming week does not yet have usable market totals. No play is forced.',
    }]), 'outputs/weekly_picks.csv')
    write_df(pd.DataFrame([{
        'card_status': 'NO_CARD',
        'note': 'No current market totals are available.',
    }]), 'outputs/weekly_card.csv')
    write_df(pd.DataFrame(), 'outputs/nws_forecasts.csv')
    snapshot = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'season': season,
        'week': week,
        'games_scanned': int(len(out)),
        'games_with_lines': 0,
        'qualifying_targets': 0,
        'leans': 0,
        'nws_ready_outdoor_games': 0,
        'board': [],
        'top_two_card_game_ids': [],
    }
    ensure_dir('outputs')
    (ROOT / 'outputs/weekly_snapshot.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')


def main() -> None:
    settings = get_settings()
    season_type = settings['data'].get('season_type', 'regular')
    season = datetime.now(timezone.utc).year
    now = pd.Timestamp.now(tz='UTC')

    client = CFBDClient()
    print(f'Pulling {season} {season_type} games...')
    games = normalize_games(client.get('/games', {'year': season, 'seasonType': season_type}), season)
    if games.empty:
        raise RuntimeError('CFBD returned no games for the current season.')

    future = games[games['start_date'].notna() & (games['start_date'] >= now)].sort_values('start_date').copy()
    if future.empty:
        raise RuntimeError('No upcoming games remain in the current regular season.')

    target_week = int(future.iloc[0]['week']) if pd.notna(future.iloc[0]['week']) else None
    board = future[future['week'].eq(target_week)].copy() if target_week is not None else future.head(100).copy()
    print(f'Upcoming season {season}, week {target_week}: {len(board)} scheduled games before market screening')

    line_records = client.get('/lines', {'year': season, 'week': target_week, 'seasonType': season_type}) if target_week is not None else []
    lines = normalize_lines(line_records)
    selected = pick_total(lines, settings['cfbd']['preferred_line_providers'])
    board = board.merge(selected, on='game_id', how='left')

    with_lines = board[board['closing_total'].notna()].copy()
    print(f'Games with a usable current total: {len(with_lines)}')
    if with_lines.empty:
        write_no_market_board(board, season, target_week)
        return
    board = with_lines

    context = line_market_context(lines)
    if not context.empty:
        board = board.merge(context, on='game_id', how='left')
        board['selected_vs_market_median'] = board['closing_total'] - board['line_total_median']
    else:
        board['line_provider_count'] = np.nan
        board['line_total_range'] = np.nan
        board['line_total_median'] = np.nan
        board['selected_vs_market_median'] = np.nan

    venues = normalize_venues(client.get('/venues'))
    if not venues.empty and 'venue_id' in board.columns:
        board = board.merge(venues, on='venue_id', how='left')
    if 'venue_name' not in board.columns and 'venue' in board.columns:
        board['venue_name'] = board['venue']
    if 'venue_dome' not in board.columns:
        board['venue_dome'] = False
    board['game_indoors'] = board['venue_dome'].map(as_bool)

    board = merge_prior_team_features(board)
    weather = fetch_nws_weather(board)
    if not weather.empty:
        board = board.merge(weather, on='game_id', how='left')
    board = add_live_categories(board)
    board = fit_and_score(board)

    statuses = board.apply(classify_row, axis=1)
    board['status'] = [s[0] for s in statuses]
    board['decision_reason'] = [s[1] for s in statuses]
    board['research_tags'] = board.apply(research_tags, axis=1)

    display_cols = [c for c in [
        'status', 'decision_reason', 'research_tags', 'season', 'week', 'game_id', 'start_date', 'start_time_tbd',
        'away_team', 'home_team', 'venue_name', 'venue_city', 'venue_state', 'game_indoors',
        'closing_total', 'line_provider', 'line_provider_count', 'line_total_range', 'line_total_median', 'selected_vs_market_median',
        'model_projected_total', 'pred_market_residual', 'abs_pred_edge', 'model_side',
        'temperature_f', 'dewpoint_f', 'humidity', 'wind_mph', 'wind_gust_mph', 'precip_probability_pct',
        'precipitation', 'snowfall', 'weather_summary', 'nws_status', 'nws_office',
        'home_conference', 'away_conference', 'home_classification', 'away_classification', 'fbs_vs_fbs',
    ] if c in board.columns]
    board = board[display_cols].copy()
    board['status_rank'] = board['status'].map({'QUALIFIES': 0, 'LEAN': 1, 'WATCH': 2, 'NO PLAY': 3, 'NO LINE': 4}).fillna(5)

    write_outputs(board, season, target_week)
    print(f"Wrote live weekly outputs with {int(board['status'].eq('QUALIFIES').sum())} qualifying target(s).")


if __name__ == '__main__':
    main()
