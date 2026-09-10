from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .build_dataset import pick_total
from .cfbd_client import CFBDClient
from .fcs_model import (
    classify_division_row,
    division_track,
    fcs_research_tags,
    score_fcs_rows,
)
from .market_line_cache import apply_market_line_cache, cached_market_reason
from .odds_api_fallback import apply_fbs_odds_fallback, apply_fcs_odds_fallback
from .oddspapi_fallback import apply_fcs_oddspapi_fallback
from .predict_week import (
    add_live_categories,
    as_bool,
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


CT = ZoneInfo('America/Chicago')

DISPLAY_COLUMNS = [
    'status', 'decision_reason', 'research_tags', 'division_track', 'model_track',
    'season', 'week', 'game_id', 'start_date', 'start_time_tbd',
    'away_team', 'home_team', 'venue_name', 'venue_city', 'venue_state', 'venue_latitude', 'venue_longitude',
    'game_indoors', 'closing_total', 'line_provider', 'line_source', 'line_provider_count', 'line_total_range',
    'line_total_median', 'selected_vs_market_median', 'odds_match_confidence',
    'line_cache_status', 'line_is_cached', 'line_last_seen_utc', 'line_age_hours',
    'model_projected_total', 'pred_market_residual', 'abs_pred_edge', 'model_side',
    'temperature_f', 'dewpoint_f', 'humidity', 'wind_mph', 'wind_gust_mph',
    'precip_probability_pct', 'precipitation', 'snowfall', 'weather_summary', 'nws_status', 'nws_office',
    'home_conference', 'away_conference', 'home_classification', 'away_classification', 'fbs_vs_fbs',
]

STATUS_RANK = {'QUALIFIES': 0, 'LEAN': 1, 'WATCH': 2, 'NO PLAY': 3, 'NO LINE': 4}


def _utc_timestamp(value: object | None = None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz='UTC')
    ts = pd.to_datetime(value, utc=True, errors='coerce')
    if pd.isna(ts):
        raise ValueError(f'Could not parse timestamp: {value!r}')
    return ts


def active_cfb_season(now: object | None = None) -> int:
    """Map January/February postseason dates back to the prior CFB season."""
    ts = _utc_timestamp(now)
    return int(ts.year - 1 if ts.month <= 2 else ts.year)


def calendar_slate_window(future_games: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the Central-time Monday-Sunday window containing the next game.

    This avoids a single postponed game that retains an old CFBD week number
    pinning the entire site to the wrong football week. Games from multiple
    CFBD week labels can coexist if they are genuinely scheduled in the same
    calendar slate.
    """
    if future_games.empty:
        raise ValueError('future_games must not be empty')
    first = pd.to_datetime(future_games['start_date'], utc=True, errors='coerce').dropna().min()
    if pd.isna(first):
        raise ValueError('future_games has no usable kickoff time')
    first_local = first.tz_convert(CT)
    start_local = first_local.normalize() - pd.Timedelta(days=int(first_local.weekday()))
    end_local = start_local + pd.Timedelta(days=7)
    return start_local.tz_convert('UTC'), end_local.tz_convert('UTC')


def load_live_games(
    client: CFBDClient,
    season: int,
    configured_season_type: str,
    now: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    """Load regular season first, then roll to postseason only when needed."""
    season_types = [configured_season_type]
    if configured_season_type == 'regular':
        season_types.append('postseason')

    last = pd.DataFrame()
    for season_type in season_types:
        records = client.get('/games', {'year': season, 'seasonType': season_type})
        games = normalize_games(records, season)
        last = games
        if games.empty:
            continue
        future = games[games['start_date'].notna() & games['start_date'].ge(now)]
        if not future.empty:
            return games, season_type

    return last, season_types[-1]


def finalize_board(board: pd.DataFrame) -> pd.DataFrame:
    out = board[[c for c in DISPLAY_COLUMNS if c in board.columns]].copy()
    out['status_rank'] = out['status'].map(STATUS_RANK).fillna(5)
    return out


def combined_research_tags(row: pd.Series) -> str:
    base = str(research_tags(row) or '').strip()
    fcs = str(fcs_research_tags(row) or '').strip()
    tags = [tag for tag in [base, fcs] if tag]
    return '; '.join(tags)


def main() -> None:
    settings = get_settings()
    configured_season_type = settings['data'].get('season_type', 'regular')
    now = pd.Timestamp.now(tz='UTC')
    season = active_cfb_season(now)

    client = CFBDClient()
    print(f'Pulling active CFB season {season}...')
    games, season_type = load_live_games(client, season, configured_season_type, now)
    if games.empty:
        raise RuntimeError(f'CFBD returned no games for active season {season}.')

    future = games[games['start_date'].notna() & games['start_date'].ge(now)].sort_values('start_date').copy()
    if future.empty:
        raise RuntimeError(f'No upcoming {season} {season_type} games remain.')

    slate_start, slate_end = calendar_slate_window(future)
    slate = future[future['start_date'].ge(slate_start) & future['start_date'].lt(slate_end)].copy()
    if slate.empty:
        raise RuntimeError('Could not construct the next Central-time football slate.')

    # The live site is scoped to games involving an FBS or FCS team. Pure
    # lower-division matchups do not have a researched model/market use case.
    home_classification = slate.get(
        'home_classification', pd.Series('', index=slate.index)
    ).astype(str).str.lower()
    away_classification = slate.get(
        'away_classification', pd.Series('', index=slate.index)
    ).astype(str).str.lower()
    live_scope = home_classification.isin({'fbs', 'fcs'}) | away_classification.isin({'fbs', 'fcs'})
    removed_lower_division = int((~live_scope).sum())
    board = slate[live_scope].copy()
    if board.empty:
        raise RuntimeError('No FBS/FCS-involved games remain in the next calendar slate.')

    board['division_track'] = division_track(board)
    week_values = pd.to_numeric(board.get('week'), errors='coerce').dropna().astype(int)
    target_week = int(week_values.mode().iloc[0]) if not week_values.empty else None
    week_labels = sorted(set(week_values.tolist()))
    start_label = slate_start.tz_convert(CT).strftime('%b %-d')
    end_label = (slate_end - pd.Timedelta(seconds=1)).tz_convert(CT).strftime('%b %-d')
    print(
        f'Upcoming season {season} {season_type}, calendar slate {start_label}-{end_label} CT: '
        f'{len(board)} live-site games ({int(board["division_track"].eq("FBS").sum())} FBS-vs-FBS, '
        f'{int(board["division_track"].eq("FCS").sum())} FCS-vs-FCS, '
        f'{removed_lower_division} pure lower-division game(s) omitted); CFBD week label(s)={week_labels or ["unknown"]}.'
    )

    # A rescheduled game can retain a different CFBD week number while sharing
    # the same actual calendar slate. Pull lines for every represented week.
    line_parts: list[pd.DataFrame] = []
    for week in week_labels:
        line_records = client.get('/lines', {'year': season, 'week': week, 'seasonType': season_type})
        part = normalize_lines(line_records)
        if not part.empty:
            line_parts.append(part)
    lines = pd.concat(line_parts, ignore_index=True) if line_parts else pd.DataFrame()
    selected = pick_total(lines, settings['cfbd']['preferred_line_providers'])
    board = board.merge(selected, on='game_id', how='left')

    context = line_market_context(lines)
    if not context.empty:
        board = board.merge(context, on='game_id', how='left')
        board['selected_vs_market_median'] = board['closing_total'] - board['line_total_median']
    else:
        board['line_provider_count'] = np.nan
        board['line_total_range'] = np.nan
        board['line_total_median'] = np.nan
        board['selected_vs_market_median'] = np.nan

    preferred = settings['cfbd'].get('preferred_line_providers', [])
    board, oddspapi_stats = apply_fcs_oddspapi_fallback(board, preferred)
    board, odds_api_stats = apply_fcs_odds_fallback(board, preferred)
    # CFBD remains primary for all FBS-involved games. Only if a current line is
    # still missing do we spend one normal NCAAF Odds API call to fill the gap.
    board, fbs_odds_api_stats = apply_fbs_odds_fallback(board, preferred)

    fcs_mask = board['division_track'].eq('FCS')
    fcs_games = int(fcs_mask.sum())
    fresh_games_with_lines = int(board['closing_total'].notna().sum()) if 'closing_total' in board.columns else 0
    fresh_fcs_with_lines = int(board.loc[fcs_mask, 'closing_total'].notna().sum()) if fcs_games else 0

    # Preserve the most recently observed total for every current live-site
    # game. Cache-restored rows stay visible and scoreable for orientation, but
    # are forced to WATCH below until a live source confirms the market again.
    board, cache_stats = apply_market_line_cache(board, now=now)
    games_with_lines = int(board['closing_total'].notna().sum()) if 'closing_total' in board.columns else 0
    fcs_with_lines = int(board.loc[fcs_mask, 'closing_total'].notna().sum()) if fcs_games else 0
    cached_fcs = int((fcs_mask & board['line_cache_status'].eq('CACHED')).sum()) if fcs_games else 0

    print(
        f"Market line cache: {fresh_games_with_lines} fresh total(s), "
        f"{cache_stats.get('restored_lines', 0)} restored last-known total(s), "
        f"{games_with_lines} usable total(s), {cache_stats.get('cached_entries', 0)} cached game(s), "
        f"{cache_stats.get('skipped_rescheduled', 0)} stale line(s) rejected after kickoff change."
    )
    print(
        f"OddsPapi: status={oddspapi_stats.get('oddspapi_status', 'unknown')}; "
        f"filled={oddspapi_stats.get('fcs_oddspapi_filled', 0)} FCS game(s); "
        f"matched={oddspapi_stats.get('fcs_oddspapi_matched_fixtures', 0)}; "
        f"matched hasOdds={oddspapi_stats.get('fcs_oddspapi_matched_with_any_odds', 0)}; "
        f"fixtures={oddspapi_stats.get('oddspapi_fixtures', 0)} "
        f"({oddspapi_stats.get('oddspapi_fixtures_with_any_odds', 0)} with any odds); "
        f"bulk books={oddspapi_stats.get('oddspapi_bulk_bookmaker_count', 0)}; "
        f"odds fixtures={oddspapi_stats.get('oddspapi_odds_fixtures', 0)}; "
        f"account requests={oddspapi_stats.get('oddspapi_request_count_before', '—')}→"
        f"{oddspapi_stats.get('oddspapi_request_count', '—')}/"
        f"{oddspapi_stats.get('oddspapi_request_limit', '—')}."
    )
    if oddspapi_stats.get('oddspapi_error_detail'):
        print(f"OddsPapi response detail: {oddspapi_stats['oddspapi_error_detail']}")
    print(
        f"FCS line coverage: {fresh_fcs_with_lines}/{fcs_games} fresh; "
        f"{cached_fcs} restored from cache; {fcs_with_lines}/{fcs_games} usable. "
        f"Secondary Odds API filled {odds_api_stats.get('fcs_fallback_filled', 0)} game(s) "
        f"(status={odds_api_stats.get('odds_api_status', 'unknown')})."
    )
    print(
        f"Missing-only NCAAF Odds API fallback filled "
        f"{fbs_odds_api_stats.get('fbs_fallback_filled', 0)} FBS-involved game(s) "
        f"(status={fbs_odds_api_stats.get('fbs_odds_api_status', 'unknown')})."
    )

    venues = normalize_venues(client.get('/venues'))
    if not venues.empty and 'venue_id' in board.columns:
        board = board.merge(venues, on='venue_id', how='left')
    if 'venue_name' not in board.columns and 'venue' in board.columns:
        board['venue_name'] = board['venue']
    if 'venue_dome' not in board.columns:
        board['venue_dome'] = False
    board['game_indoors'] = board['venue_dome'].map(as_bool)

    if games_with_lines == 0:
        board['status'] = 'NO LINE'
        board['decision_reason'] = np.where(
            board['division_track'].eq('FCS'),
            'No current or last-known FCS market total is available from CFBD, OddsPapi, the secondary odds fallback, or the market-line cache.',
            'No current or last-known market total is available.',
        )
        board['research_tags'] = ''
        board['model_track'] = np.where(board['division_track'].eq('FCS'), 'FCS-only HGB', 'GENERAL HGB')
        board['model_projected_total'] = np.nan
        board['pred_market_residual'] = np.nan
        board['abs_pred_edge'] = np.nan
        board['model_side'] = ''
        board['nws_status'] = ''
        board = finalize_board(board)
        write_outputs(board, season, target_week)
        print('Wrote the full weekly slate with no fresh or cached market totals.')
        return

    board = merge_prior_team_features(board)

    weather_input = board[board['closing_total'].notna()].copy()
    weather = fetch_nws_weather(weather_input)
    if not weather.empty:
        board = board.merge(weather, on='game_id', how='left')

    board = add_live_categories(board)
    board = fit_and_score(board)
    board['model_track'] = 'GENERAL HGB'
    board = score_fcs_rows(board)

    statuses: list[tuple[str, str]] = []
    for _, row in board.iterrows():
        cache_reason = cached_market_reason(row)
        if cache_reason:
            statuses.append(('WATCH', cache_reason))
        else:
            statuses.append(classify_division_row(row))
    board['status'] = [s[0] for s in statuses]
    board['decision_reason'] = [s[1] for s in statuses]
    board['research_tags'] = board.apply(combined_research_tags, axis=1)
    board = finalize_board(board)

    write_outputs(board, season, target_week)
    print(
        f"Wrote {len(board)} weekly games with {games_with_lines} usable totals "
        f"({fresh_games_with_lines} fresh, {cache_stats.get('restored_lines', 0)} cached), "
        f"{fcs_games} FCS-vs-FCS games ({fcs_with_lines} with usable totals), and "
        f"{int(board['status'].eq('QUALIFIES').sum())} qualifying target(s)."
    )


if __name__ == '__main__':
    main()
