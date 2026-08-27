from __future__ import annotations

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
from .utils import get_settings


DISPLAY_COLUMNS = [
    'status', 'decision_reason', 'research_tags', 'season', 'week', 'game_id', 'start_date', 'start_time_tbd',
    'away_team', 'home_team', 'venue_name', 'venue_city', 'venue_state', 'venue_latitude', 'venue_longitude',
    'game_indoors', 'closing_total', 'line_provider', 'line_provider_count', 'line_total_range',
    'line_total_median', 'selected_vs_market_median', 'model_projected_total', 'pred_market_residual',
    'abs_pred_edge', 'model_side', 'temperature_f', 'dewpoint_f', 'humidity', 'wind_mph', 'wind_gust_mph',
    'precip_probability_pct', 'precipitation', 'snowfall', 'weather_summary', 'nws_status', 'nws_office',
    'home_conference', 'away_conference', 'home_classification', 'away_classification', 'fbs_vs_fbs',
]

STATUS_RANK = {'QUALIFIES': 0, 'LEAN': 1, 'WATCH': 2, 'NO PLAY': 3, 'NO LINE': 4}


def finalize_board(board: pd.DataFrame) -> pd.DataFrame:
    out = board[[c for c in DISPLAY_COLUMNS if c in board.columns]].copy()
    out['status_rank'] = out['status'].map(STATUS_RANK).fillna(5)
    return out


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
    print(f'Upcoming season {season}, week {target_week}: {len(board)} scheduled games')

    line_records = client.get('/lines', {'year': season, 'week': target_week, 'seasonType': season_type}) if target_week is not None else []
    lines = normalize_lines(line_records)
    selected = pick_total(lines, settings['cfbd']['preferred_line_providers'])
    board = board.merge(selected, on='game_id', how='left')

    games_with_lines = int(board['closing_total'].notna().sum()) if 'closing_total' in board.columns else 0
    print(f'Games with a usable current total: {games_with_lines}')

    context = line_market_context(lines)
    if not context.empty:
        board = board.merge(context, on='game_id', how='left')
        board['selected_vs_market_median'] = board['closing_total'] - board['line_total_median']
    else:
        board['line_provider_count'] = np.nan
        board['line_total_range'] = np.nan
        board['line_total_median'] = np.nan
        board['selected_vs_market_median'] = np.nan

    # Venue coordinates are retained in the production board so the website can
    # map every scheduled game, including games that do not yet have a total.
    venues = normalize_venues(client.get('/venues'))
    if not venues.empty and 'venue_id' in board.columns:
        board = board.merge(venues, on='venue_id', how='left')
    if 'venue_name' not in board.columns and 'venue' in board.columns:
        board['venue_name'] = board['venue']
    if 'venue_dome' not in board.columns:
        board['venue_dome'] = False
    board['game_indoors'] = board['venue_dome'].map(as_bool)

    # When the market has not posted any totals yet, keep the complete slate as
    # NO LINE instead of dropping games or spending NWS/model work unnecessarily.
    if games_with_lines == 0:
        board['status'] = 'NO LINE'
        board['decision_reason'] = 'No current market total is available.'
        board['research_tags'] = ''
        board['model_projected_total'] = np.nan
        board['pred_market_residual'] = np.nan
        board['abs_pred_edge'] = np.nan
        board['model_side'] = ''
        board['nws_status'] = ''
        board = finalize_board(board)
        write_outputs(board, season, target_week)
        print('Wrote the full weekly slate with no current market totals.')
        return

    board = merge_prior_team_features(board)

    # NWS requests are only needed for games the market can currently score.
    # NO LINE games still remain on the site/map with their venue coordinates.
    weather_input = board[board['closing_total'].notna()].copy()
    weather = fetch_nws_weather(weather_input)
    if not weather.empty:
        board = board.merge(weather, on='game_id', how='left')

    board = add_live_categories(board)
    board = fit_and_score(board)

    statuses = board.apply(classify_row, axis=1)
    board['status'] = [s[0] for s in statuses]
    board['decision_reason'] = [s[1] for s in statuses]
    board['research_tags'] = board.apply(research_tags, axis=1)
    board = finalize_board(board)

    write_outputs(board, season, target_week)
    print(
        f"Wrote {len(board)} weekly games with {games_with_lines} current totals and "
        f"{int(board['status'].eq('QUALIFIES').sum())} qualifying target(s)."
    )


if __name__ == '__main__':
    main()
