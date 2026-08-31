from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .cfbd_client import CFBDClient
from .deep_research import prep
from .model_bakeoff import feature_lists, prep_features, reg_models
from .nws_forecast import NWSClient
from .predict_week import add_live_categories, merge_prior_team_features
from .utils import ROOT, ensure_dir, read_df

ORIENTATION_PATH = ROOT / 'data/reference/stadium_orientations.csv'
SHADOW_DIR = ROOT / 'outputs/orientation_shadow/2026'
CHALLENGER_VERSION = 'orientation-crosswind-hgb-v0.1'


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def field_angle(wind_direction: np.ndarray, field_axis: np.ndarray) -> np.ndarray:
    wind_axis = np.mod(wind_direction, 180.0)
    axis = np.mod(field_axis, 180.0)
    delta = np.abs(wind_axis - axis)
    return np.minimum(delta, 180.0 - delta)


def orientation_table() -> pd.DataFrame:
    orientation = pd.read_csv(ORIENTATION_PATH)
    keep = [c for c in ['venue_id', 'field_axis_deg', 'axis_uncertainty_deg', 'roof_behavior'] if c in orientation.columns]
    orientation = orientation[keep].copy()
    for col in ['venue_id', 'field_axis_deg', 'axis_uncertainty_deg']:
        if col in orientation.columns:
            orientation[col] = pd.to_numeric(orientation[col], errors='coerce')
    return orientation.drop_duplicates('venue_id')


def add_historical_crosswind(raw: pd.DataFrame) -> pd.DataFrame:
    historical = raw.copy()
    historical['venue_id'] = pd.to_numeric(historical.get('venue_id'), errors='coerce')
    historical['wind_mph'] = pd.to_numeric(historical.get('wind_mph'), errors='coerce')
    historical['wind_direction_degrees'] = pd.to_numeric(historical.get('wind_direction_degrees'), errors='coerce')
    historical = historical.merge(orientation_table()[['venue_id', 'field_axis_deg']], on='venue_id', how='left')
    historical['crosswind_mph'] = np.nan
    valid = historical[['wind_mph', 'wind_direction_degrees', 'field_axis_deg']].notna().all(axis=1)
    if valid.any():
        angle = field_angle(
            historical.loc[valid, 'wind_direction_degrees'].to_numpy(float),
            historical.loc[valid, 'field_axis_deg'].to_numpy(float),
        )
        historical.loc[valid, 'crosswind_mph'] = historical.loc[valid, 'wind_mph'].to_numpy(float) * np.sin(np.deg2rad(angle))
    return historical


def live_game_context(season: int) -> pd.DataFrame:
    client = CFBDClient()
    games = pd.json_normalize(client.get('/games', {'year': season, 'seasonType': 'regular'}))
    if games.empty:
        return pd.DataFrame(columns=['game_id', 'venue_id', 'neutral_site', 'conference_game'])
    games = games.rename(columns={
        'id': 'game_id',
        'venueId': 'venue_id',
        'neutralSite': 'neutral_site',
        'conferenceGame': 'conference_game',
    })
    for col in ['game_id', 'venue_id']:
        games[col] = pd.to_numeric(games.get(col), errors='coerce')
    keep = [c for c in ['game_id', 'venue_id', 'neutral_site', 'conference_game'] if c in games.columns]
    return games[keep].drop_duplicates('game_id')


def fetch_shadow_direction(board: pd.DataFrame) -> pd.DataFrame:
    nws = NWSClient()
    rows: list[dict] = []
    for _, row in board.iterrows():
        game_id = row.get('game_id')
        if str(row.get('division_track', '')).upper() != 'FBS' or as_bool(row.get('game_indoors')):
            rows.append({'game_id': game_id, 'shadow_wind_direction_degrees': np.nan, 'shadow_weather_status': 'not_applicable'})
            continue
        kickoff = pd.to_datetime(row.get('start_date'), utc=True, errors='coerce')
        latitude = pd.to_numeric(pd.Series([row.get('venue_latitude')]), errors='coerce').iloc[0]
        longitude = pd.to_numeric(pd.Series([row.get('venue_longitude')]), errors='coerce').iloc[0]
        if pd.isna(kickoff) or pd.isna(latitude) or pd.isna(longitude):
            rows.append({'game_id': game_id, 'shadow_wind_direction_degrees': np.nan, 'shadow_weather_status': 'missing_location_or_time'})
            continue
        try:
            forecast = nws.kickoff_forecast(float(latitude), float(longitude), kickoff.to_pydatetime())
            rows.append({
                'game_id': game_id,
                'shadow_wind_direction_degrees': forecast.get('wind_direction_degrees'),
                'shadow_weather_status': forecast.get('nws_status', 'ok'),
            })
        except Exception as exc:
            rows.append({
                'game_id': game_id,
                'shadow_wind_direction_degrees': np.nan,
                'shadow_weather_status': f'error:{type(exc).__name__}',
            })
    nws.save_cache()
    return pd.DataFrame(rows)


def challenger_status(prediction: float, total: float, weather_status: object, start_time_tbd: object) -> str:
    if not np.isfinite(total):
        return 'NO LINE'
    if not np.isfinite(prediction):
        return 'NO PLAY'
    if str(weather_status or '').lower() not in {'ok', 'indoor'}:
        return 'WATCH'
    if as_bool(start_time_tbd):
        return 'WATCH'
    if prediction <= -3.5 and total >= 56:
        return 'QUALIFIES'
    if prediction <= -3.5:
        return 'LEAN'
    return 'NO PLAY'


def score_challenger(board: pd.DataFrame) -> pd.DataFrame:
    historical = prep(add_historical_crosswind(read_df('data/processed/modeling_dataset.csv')))
    nums, cats = feature_lists(historical)
    nums = nums + ['crosswind_mph']
    historical = prep_features(historical, cats)
    model = reg_models(nums, cats)['hist_gradient_boosting']
    model.fit(historical[nums + cats], historical['market_residual'])

    live = add_live_categories(merge_prior_team_features(board.copy()))
    for col in nums:
        if col not in live.columns:
            live[col] = np.nan
        live[col] = pd.to_numeric(live[col], errors='coerce')
    for col in cats:
        if col not in live.columns:
            live[col] = 'missing'
        live[col] = live[col].astype(str).fillna('missing')

    mask = live['division_track'].astype(str).str.upper().eq('FBS') & live['closing_total'].notna()
    live['challenger_pred_market_residual'] = np.nan
    if mask.any():
        live.loc[mask, 'challenger_pred_market_residual'] = model.predict(live.loc[mask, nums + cats])
    live['challenger_projected_total'] = live['closing_total'] + live['challenger_pred_market_residual']
    live['challenger_abs_edge'] = live['challenger_pred_market_residual'].abs()
    live['challenger_status'] = [
        challenger_status(
            float(pred) if pd.notna(pred) else np.nan,
            float(total) if pd.notna(total) else np.nan,
            weather,
            tbd,
        )
        for pred, total, weather, tbd in zip(
            live['challenger_pred_market_residual'],
            live['closing_total'],
            live['nws_status'],
            live.get('start_time_tbd', pd.Series(False, index=live.index)),
        )
    ]
    return live


def main() -> None:
    board = read_df('outputs/weekly_board.csv').copy()
    if board.empty:
        raise RuntimeError('weekly_board.csv is empty.')
    board['game_id'] = pd.to_numeric(board['game_id'], errors='coerce')
    season = int(pd.to_numeric(board['season'], errors='coerce').dropna().max())

    context = live_game_context(season)
    board = board.drop(columns=[c for c in ['venue_id', 'neutral_site', 'conference_game'] if c in board.columns], errors='ignore')
    board = board.merge(context, on='game_id', how='left')
    board = board.merge(orientation_table(), on='venue_id', how='left')
    board = board.merge(fetch_shadow_direction(board), on='game_id', how='left')
    board['field_axis_deg'] = pd.to_numeric(board['field_axis_deg'], errors='coerce')
    board['shadow_wind_direction_degrees'] = pd.to_numeric(board['shadow_wind_direction_degrees'], errors='coerce')
    board['wind_mph'] = pd.to_numeric(board['wind_mph'], errors='coerce')

    board['wind_field_angle_deg'] = np.nan
    board['crosswind_mph'] = np.nan
    board['alongwind_mph'] = np.nan
    valid = board[['field_axis_deg', 'shadow_wind_direction_degrees', 'wind_mph']].notna().all(axis=1)
    if valid.any():
        angle = field_angle(
            board.loc[valid, 'shadow_wind_direction_degrees'].to_numpy(float),
            board.loc[valid, 'field_axis_deg'].to_numpy(float),
        )
        board.loc[valid, 'wind_field_angle_deg'] = angle
        board.loc[valid, 'crosswind_mph'] = board.loc[valid, 'wind_mph'].to_numpy(float) * np.sin(np.deg2rad(angle))
        board.loc[valid, 'alongwind_mph'] = board.loc[valid, 'wind_mph'].to_numpy(float) * np.cos(np.deg2rad(angle))

    scored = score_challenger(board)
    scored['challenger_version'] = CHALLENGER_VERSION
    scored['baseline_status'] = scored['status']
    scored['baseline_pred_market_residual'] = pd.to_numeric(scored['pred_market_residual'], errors='coerce')
    scored['challenger_minus_baseline_edge'] = scored['challenger_pred_market_residual'] - scored['baseline_pred_market_residual']
    scored['status_changed'] = scored['challenger_status'].astype(str) != scored['baseline_status'].astype(str)
    scored['captured_at_utc'] = datetime.now(timezone.utc).isoformat()
    scored['github_run_id'] = os.getenv('GITHUB_RUN_ID', '')
    scored['github_run_attempt'] = os.getenv('GITHUB_RUN_ATTEMPT', '')
    scored['github_sha'] = os.getenv('GITHUB_SHA', '')
    scored['research_only'] = True

    keep = [c for c in [
        'captured_at_utc', 'github_run_id', 'github_run_attempt', 'github_sha', 'challenger_version', 'research_only',
        'season', 'week', 'game_id', 'start_date', 'away_team', 'home_team', 'venue_id', 'venue_name', 'division_track',
        'closing_total', 'line_provider', 'baseline_status', 'baseline_pred_market_residual', 'model_projected_total',
        'challenger_status', 'challenger_pred_market_residual', 'challenger_projected_total', 'challenger_abs_edge',
        'challenger_minus_baseline_edge', 'status_changed', 'temperature_f', 'humidity', 'precipitation', 'wind_mph', 'wind_gust_mph',
        'shadow_wind_direction_degrees', 'field_axis_deg', 'axis_uncertainty_deg', 'wind_field_angle_deg', 'crosswind_mph', 'alongwind_mph',
        'nws_status', 'shadow_weather_status', 'weather_summary'
    ] if c in scored.columns]
    output = scored[keep].copy()

    ensure_dir(SHADOW_DIR)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = os.getenv('GITHUB_RUN_ID', 'local')
    attempt = os.getenv('GITHUB_RUN_ATTEMPT', '1')
    snapshot = SHADOW_DIR / f'shadow_{stamp}_run{run}_a{attempt}.csv'
    output.to_csv(snapshot, index=False)
    output.to_csv(SHADOW_DIR / 'latest.csv', index=False)

    manifest = SHADOW_DIR / 'manifest.csv'
    row = pd.DataFrame([{
        'captured_at_utc': output['captured_at_utc'].iloc[0] if len(output) else datetime.now(timezone.utc).isoformat(),
        'snapshot_file': snapshot.name,
        'challenger_version': CHALLENGER_VERSION,
        'games': len(output),
        'fbs_games': int(output['division_track'].astype(str).str.upper().eq('FBS').sum()),
        'orientation_ready_fbs_games': int((output['division_track'].astype(str).str.upper().eq('FBS') & output['crosswind_mph'].notna()).sum()),
        'status_disagreements': int(output['status_changed'].fillna(False).sum()),
        'github_run_id': run,
        'github_run_attempt': attempt,
        'github_sha': os.getenv('GITHUB_SHA', ''),
    }])
    if manifest.exists():
        old = pd.read_csv(manifest)
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(manifest, index=False)
    print(f'Wrote research-only orientation shadow snapshot: {snapshot}')


if __name__ == '__main__':
    main()
