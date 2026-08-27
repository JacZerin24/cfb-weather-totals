from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .build_dataset import pick_total
from .cfbd_client import CFBDClient
from .deep_research import prep
from .model_bakeoff import feature_lists, prep_features, reg_models
from .nws_forecast import NWSClient
from .pull_historical_lines import flatten_lines
from .utils import ROOT, ensure_dir, get_settings, read_df, write_df


def _provider_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get('name') or value.get('title') or value.get('provider') or 'unknown')
    return str(value) if value is not None else 'unknown'


def normalize_games(records: list[dict[str, Any]], season: int) -> pd.DataFrame:
    games = pd.json_normalize(records)
    if games.empty:
        return games
    games = games.rename(columns={
        'id': 'game_id',
        'seasonType': 'season_type',
        'startDate': 'start_date',
        'startTimeTBD': 'start_time_tbd',
        'homeTeam': 'home_team',
        'awayTeam': 'away_team',
        'venueId': 'venue_id',
        'conferenceGame': 'conference_game',
        'neutralSite': 'neutral_site',
        'homeClassification': 'home_classification',
        'awayClassification': 'away_classification',
        'homeConference': 'home_conference',
        'awayConference': 'away_conference',
        'homePregameElo': 'home_pregame_elo',
        'awayPregameElo': 'away_pregame_elo',
    })
    games['season'] = season
    games['start_date'] = pd.to_datetime(games.get('start_date'), utc=True, errors='coerce')
    games['week'] = pd.to_numeric(games.get('week'), errors='coerce').astype('Int64')
    if 'start_time_tbd' not in games.columns:
        games['start_time_tbd'] = False
    return games


def normalize_venues(records: list[dict[str, Any]]) -> pd.DataFrame:
    venues = pd.json_normalize(records)
    if venues.empty:
        return venues
    venues = venues.rename(columns={
        'id': 'venue_id',
        'name': 'venue_name',
        'dome': 'venue_dome',
        'latitude': 'venue_latitude',
        'longitude': 'venue_longitude',
        'timezone': 'venue_timezone',
        'city': 'venue_city',
        'state': 'venue_state',
    })
    keep = [c for c in [
        'venue_id', 'venue_name', 'venue_dome', 'venue_latitude', 'venue_longitude',
        'venue_timezone', 'venue_city', 'venue_state',
    ] if c in venues.columns]
    return venues[keep].drop_duplicates('venue_id') if 'venue_id' in keep else venues[keep]


def normalize_lines(records: list[dict[str, Any]]) -> pd.DataFrame:
    lines = flatten_lines(records)
    if lines.empty:
        return lines
    if 'provider' in lines.columns:
        lines['provider'] = lines['provider'].map(_provider_name)
    lines['over_under'] = pd.to_numeric(lines.get('over_under'), errors='coerce')
    return lines


def line_market_context(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty or 'game_id' not in lines.columns:
        return pd.DataFrame(columns=[
            'game_id', 'line_provider_count', 'line_total_min', 'line_total_max',
            'line_total_median', 'line_total_range',
        ])
    work = lines.dropna(subset=['game_id', 'over_under']).copy()
    if work.empty:
        return pd.DataFrame()
    grouped = work.groupby('game_id', as_index=False).agg(
        line_provider_count=('provider', 'nunique'),
        line_total_min=('over_under', 'min'),
        line_total_max=('over_under', 'max'),
        line_total_median=('over_under', 'median'),
    )
    grouped['line_total_range'] = grouped['line_total_max'] - grouped['line_total_min']
    return grouped


def merge_prior_team_features(dataset: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / 'data/processed/team_prior_features.csv'
    if not path.exists():
        return dataset
    feats = read_df(path)
    if feats.empty or 'feature_team' not in feats.columns:
        return dataset
    stat_cols = [c for c in feats.columns if c not in {'season', 'feature_team'}]
    home = feats.rename(columns={'feature_team': 'home_team', **{c: f'home_prior_{c}' for c in stat_cols}})
    away = feats.rename(columns={'feature_team': 'away_team', **{c: f'away_prior_{c}' for c in stat_cols}})
    out = dataset.merge(home, on=['season', 'home_team'], how='left')
    out = out.merge(away, on=['season', 'away_team'], how='left')
    return out


def add_live_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['closing_total'] = pd.to_numeric(out.get('closing_total'), errors='coerce')
    for col in ['wind_mph', 'temperature_f', 'humidity', 'precipitation', 'snowfall', 'dewpoint_f', 'pressure']:
        out[col] = pd.to_numeric(out.get(col), errors='coerce') if col in out.columns else np.nan
    out['game_indoors_bool'] = out.get('game_indoors', False).fillna(False).astype(bool)
    out['wind_bin'] = pd.cut(out['wind_mph'], bins=[-1, 5, 10, 15, 20, 200], labels=['0-5', '5-10', '10-15', '15-20', '20+'])
    out['temp_bin'] = pd.cut(out['temperature_f'], bins=[-100, 35, 50, 70, 85, 200], labels=['<=35', '35-50', '50-70', '70-85', '85+'])
    out['total_bin'] = pd.cut(out['closing_total'], [0, 42, 49, 56, 63, 100], labels=['<=42', '42-49', '49-56', '56-63', '63+'])
    if {'home_classification', 'away_classification'} <= set(out.columns):
        out['fbs_vs_fbs'] = (
            out['home_classification'].astype(str).str.lower().eq('fbs')
            & out['away_classification'].astype(str).str.lower().eq('fbs')
        )
    else:
        out['fbs_vs_fbs'] = False
    return out


def fetch_nws_weather(board: pd.DataFrame) -> pd.DataFrame:
    nws = NWSClient()
    rows: list[dict[str, Any]] = []
    for _, game in board.iterrows():
        game_id = game.get('game_id')
        indoors = bool(game.get('game_indoors', False))
        if indoors:
            rows.append({
                'game_id': game_id,
                'nws_status': 'indoor',
                'weather_summary': 'Indoor/dome game',
            })
            continue
        kickoff = game.get('start_date')
        latitude = pd.to_numeric(pd.Series([game.get('venue_latitude')]), errors='coerce').iloc[0]
        longitude = pd.to_numeric(pd.Series([game.get('venue_longitude')]), errors='coerce').iloc[0]
        if pd.isna(kickoff):
            rows.append({'game_id': game_id, 'nws_status': 'missing_kickoff', 'weather_summary': 'Kickoff time unavailable'})
            continue
        if pd.isna(latitude) or pd.isna(longitude):
            rows.append({'game_id': game_id, 'nws_status': 'missing_coordinates', 'weather_summary': 'Venue coordinates unavailable'})
            continue
        try:
            forecast = nws.kickoff_forecast(float(latitude), float(longitude), kickoff.to_pydatetime())
            forecast['game_id'] = game_id
            rows.append(forecast)
        except Exception as exc:
            rows.append({
                'game_id': game_id,
                'nws_status': 'error',
                'weather_summary': f'NWS forecast unavailable: {type(exc).__name__}',
            })
    nws.save_cache()
    return pd.DataFrame(rows)


def fit_and_score(live: pd.DataFrame) -> pd.DataFrame:
    historical = prep(read_df('data/processed/modeling_dataset.csv'))
    nums, cats = feature_lists(historical)
    historical = prep_features(historical, cats)
    model = reg_models(nums, cats)['hist_gradient_boosting']
    model.fit(historical[nums + cats], historical['market_residual'])

    scored = live.copy()
    for col in nums:
        if col not in scored.columns:
            scored[col] = np.nan
        scored[col] = pd.to_numeric(scored[col], errors='coerce')
    for col in cats:
        if col not in scored.columns:
            scored[col] = 'missing'
        scored[col] = scored[col].astype(str).fillna('missing')

    has_line = scored['closing_total'].notna()
    scored['pred_market_residual'] = np.nan
    if has_line.any():
        scored.loc[has_line, 'pred_market_residual'] = model.predict(scored.loc[has_line, nums + cats])
    scored['model_projected_total'] = scored['closing_total'] + scored['pred_market_residual']
    scored['model_side'] = np.where(scored['pred_market_residual'] < 0, 'under', 'over')
    scored['abs_pred_edge'] = scored['pred_market_residual'].abs()
    return scored


def research_tags(row: pd.Series) -> str:
    tags: list[str] = []
    edge = row.get('abs_pred_edge')
    total = row.get('closing_total')
    wind = row.get('wind_mph')
    humidity = row.get('humidity')
    providers = row.get('line_provider_count')
    selected_vs_median = row.get('selected_vs_market_median')

    if pd.notna(edge) and edge >= 5:
        tags.append('5.0+ model edge')
    if pd.notna(total) and pd.notna(wind) and total >= 60 and wind >= 10:
        tags.append('refinement: total≥60 + wind≥10')
    elif pd.notna(total) and pd.notna(wind) and total >= 58 and wind >= 12:
        tags.append('refinement: total≥58 + wind≥12')
    if pd.notna(total) and pd.notna(humidity) and total >= 60 and humidity >= 80:
        tags.append('refinement: total≥60 + RH≥80%')
    if pd.notna(providers) and pd.notna(total) and providers >= 3 and total >= 56:
        tags.append('3+ line providers + high total')
    if pd.notna(selected_vs_median) and selected_vs_median >= 0.5:
        tags.append('selected total above market median')
    return '; '.join(tags[:4])


def classify_row(row: pd.Series) -> tuple[str, str]:
    if pd.isna(row.get('closing_total')):
        return 'NO LINE', 'No current market total is available.'
    if pd.isna(row.get('pred_market_residual')):
        return 'NO PLAY', 'The model could not score this game.'

    forecast_ready = bool(row.get('game_indoors_bool')) or row.get('nws_status') == 'ok'
    if not forecast_ready:
        return 'WATCH', 'Outdoor game does not yet have a usable NWS kickoff forecast.'
    if bool(row.get('start_time_tbd', False)):
        return 'WATCH', 'Kickoff time is still TBD, so the weather match is not stable yet.'

    pred = float(row['pred_market_residual'])
    total = float(row['closing_total'])
    if pred <= -3.5 and total >= 56:
        return 'QUALIFIES', 'Validated HGB under edge ≥3.5 with the preferred high-total screen.'
    if pred <= -3.5:
        return 'LEAN', 'HGB under edge ≥3.5, but the market total is below the preferred 56+ weekly screen.'
    if pred >= 3.5:
        return 'NO PLAY', 'Model points over, but overs did not validate as the production direction.'
    return 'NO PLAY', 'Model edge is below the 3.5-point production threshold.'


def clean_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.copy()
    clean = clean.where(pd.notnull(clean), None)
    records = clean.to_dict(orient='records')
    for row in records:
        for key, value in list(row.items()):
            if isinstance(value, pd.Timestamp):
                row[key] = value.isoformat()
            elif isinstance(value, np.generic):
                row[key] = value.item()
    return records


def write_outputs(board: pd.DataFrame, season: int, week: int | None) -> None:
    board = board.sort_values(['status_rank', 'abs_pred_edge'], ascending=[True, False], na_position='last').reset_index(drop=True)
    write_df(board, 'outputs/weekly_board.csv')

    targets = board[board['status'].eq('QUALIFIES')].copy()
    if targets.empty:
        picks = pd.DataFrame([{
            'status': 'no_qualifying_plays',
            'note': 'No game currently satisfies the HGB under 3.5+ edge, 56+ total, and forecast-readiness screen.',
        }])
    else:
        picks = targets.copy()
    write_df(picks, 'outputs/weekly_picks.csv')

    card = targets.sort_values('abs_pred_edge', ascending=False).head(2).copy()
    if len(card) == 2:
        card['card_leg'] = [1, 2]
        card['card_status'] = 'TOP_2_LEG_RESEARCH_CARD'
    else:
        card = pd.DataFrame([{
            'card_status': 'NO_CARD',
            'note': 'Fewer than two qualifying high-total HGB under targets are available; no 2-leg card is forced.',
        }])
    write_df(card, 'outputs/weekly_card.csv')

    weather_cols = [c for c in [
        'game_id', 'away_team', 'home_team', 'start_date', 'venue_name', 'game_indoors',
        'nws_status', 'nws_office', 'temperature_f', 'dewpoint_f', 'humidity', 'wind_mph',
        'wind_gust_mph', 'precip_probability_pct', 'precipitation', 'snowfall', 'weather_summary',
    ] if c in board.columns]
    write_df(board[weather_cols], 'outputs/nws_forecasts.csv')

    snapshot = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'season': season,
        'week': week,
        'games_scanned': int(len(board)),
        'games_with_lines': int(board['closing_total'].notna().sum()),
        'qualifying_targets': int(board['status'].eq('QUALIFIES').sum()),
        'leans': int(board['status'].eq('LEAN').sum()),
        'nws_ready_outdoor_games': int(board['nws_status'].eq('ok').sum()) if 'nws_status' in board.columns else 0,
        'board': clean_json_records(board),
        'top_two_card_game_ids': [str(v) for v in targets.sort_values('abs_pred_edge', ascending=False).head(2).get('game_id', pd.Series(dtype=str)).tolist()] if len(targets) >= 2 else [],
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
    game_records = client.get('/games', {'year': season, 'seasonType': season_type})
    games = normalize_games(game_records, season)
    if games.empty:
        raise RuntimeError('CFBD returned no games for the current season.')

    future = games[games['start_date'].notna() & (games['start_date'] >= now)].sort_values('start_date').copy()
    if future.empty:
        raise RuntimeError('No upcoming games remain in the current regular season.')
    target_week = int(future.iloc[0]['week']) if pd.notna(future.iloc[0]['week']) else None
    board = future[future['week'].eq(target_week)].copy() if target_week is not None else future.head(80).copy()
    print(f'Building live board for season {season}, week {target_week}: {len(board)} games')

    venues = normalize_venues(client.get('/venues'))
    if not venues.empty and 'venue_id' in board.columns:
        board = board.merge(venues, on='venue_id', how='left')
    if 'venue_name' not in board.columns and 'venue' in board.columns:
        board['venue_name'] = board['venue']
    if 'venue_dome' not in board.columns:
        board['venue_dome'] = False
    board['game_indoors'] = board['venue_dome'].fillna(False).astype(bool)

    line_records: list[dict[str, Any]] = []
    if target_week is not None:
        line_records = client.get('/lines', {'year': season, 'week': target_week, 'seasonType': season_type})
    lines = normalize_lines(line_records)
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
    board['status_rank'] = board['status'].map({'QUALIFIES': 0, 'LEAN': 1, 'WATCH': 2, 'NO PLAY': 3, 'NO LINE': 4}).fillna(5)

    display_cols = [c for c in [
        'status', 'decision_reason', 'research_tags', 'season', 'week', 'game_id', 'start_date', 'start_time_tbd',
        'away_team', 'home_team', 'venue_name', 'venue_city', 'venue_state', 'game_indoors',
        'closing_total', 'line_provider', 'line_provider_count', 'line_total_range', 'line_total_median', 'selected_vs_market_median',
        'model_projected_total', 'pred_market_residual', 'abs_pred_edge', 'model_side',
        'temperature_f', 'dewpoint_f', 'humidity', 'wind_mph', 'wind_gust_mph', 'precip_probability_pct',
        'precipitation', 'snowfall', 'weather_summary', 'nws_status', 'nws_office',
        'home_conference', 'away_conference', 'home_classification', 'away_classification', 'fbs_vs_fbs',
    ] if c in board.columns]
    board = board[display_cols + [c for c in board.columns if c not in display_cols and c.startswith(('home_prior_', 'away_prior_'))]]
    board['status_rank'] = board['status'].map({'QUALIFIES': 0, 'LEAN': 1, 'WATCH': 2, 'NO PLAY': 3, 'NO LINE': 4}).fillna(5)

    write_outputs(board, season, target_week)
    print(f"Wrote live weekly outputs with {int(board['status'].eq('QUALIFIES').sum())} qualifying target(s).")


if __name__ == '__main__':
    main()
