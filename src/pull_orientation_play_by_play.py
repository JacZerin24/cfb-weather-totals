from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .cfbd_client import CFBDClient
from .stadium_wind_orientation_research import load_data
from .utils import ROOT, ensure_dir, load_yaml, write_df

PROTOCOL_PATH = ROOT / 'config/orientation_pbp_mechanism_2026.yml'
RAW_PATH = ROOT / 'data/raw/orientation_pbp_10_15.csv'
OUT_DIR = ROOT / 'outputs/orientation_research'


def protocol() -> dict:
    return load_yaml('config/orientation_pbp_mechanism_2026.yml')


def target_games() -> pd.DataFrame:
    cfg = protocol()
    frame = load_data().copy()
    low = float(cfg['sample']['raw_wind_mph']['greater_than'])
    high = float(cfg['sample']['raw_wind_mph']['less_than_or_equal'])
    parallel_max = float(cfg['sample']['alignment']['parallel_max_degrees'])
    cross_min = float(cfg['sample']['alignment']['cross_min_degrees'])
    seasons = {int(v) for v in cfg['sample']['seasons']}

    season_num = pd.to_numeric(frame['season'], errors='coerce')
    week_num = pd.to_numeric(frame['week'], errors='coerce')
    angle = pd.to_numeric(frame['wind_field_angle_deg'], errors='coerce')
    wind = pd.to_numeric(frame['wind_mph'], errors='coerce')
    keep = (
        season_num.isin(seasons)
        & frame['division_track'].eq(str(cfg['sample']['division_track']))
        & frame['outdoor'].fillna(False)
        & wind.gt(low)
        & wind.le(high)
        & (angle.le(parallel_max) | angle.ge(cross_min))
        & week_num.notna()
    )
    target = frame.loc[keep].copy()
    target['season'] = season_num.loc[keep].astype(int)
    target['week'] = week_num.loc[keep].astype(int)
    target['game_id'] = pd.to_numeric(target['game_id'], errors='coerce').astype('Int64')
    target['alignment'] = np.where(angle.loc[keep].ge(cross_min), 'cross', 'parallel')
    target = target.dropna(subset=['game_id']).drop_duplicates('game_id')
    cols = [c for c in [
        'game_id', 'season', 'week', 'home_team', 'away_team', 'venue_id', 'venue_name',
        'alignment', 'wind_mph', 'wind_direction_degrees', 'field_axis_deg',
        'wind_field_angle_deg', 'temperature_f', 'humidity', 'precipitation',
        'dewpoint_f', 'pressure', 'closing_total', 'neutral_site', 'line_provider',
    ] if c in target.columns]
    return target[cols].sort_values(['season', 'week', 'game_id']).reset_index(drop=True)


def normalize_plays(records: list[dict]) -> pd.DataFrame:
    plays = pd.json_normalize(records)
    if plays.empty:
        return plays
    plays = plays.rename(columns={
        'id': 'play_id',
        'gameId': 'game_id',
        'driveId': 'drive_id',
        'driveNumber': 'drive_number',
        'playNumber': 'play_number',
        'offenseConference': 'offense_conference',
        'offenseScore': 'offense_score',
        'defenseConference': 'defense_conference',
        'defenseScore': 'defense_score',
        'yardsToGoal': 'yards_to_goal',
        'yardsGained': 'yards_gained',
        'playType': 'play_type',
        'playText': 'play_text',
    })
    plays['game_id'] = pd.to_numeric(plays.get('game_id'), errors='coerce').astype('Int64')
    return plays


def main() -> None:
    cfg = protocol()
    targets = target_games()
    if targets.empty:
        raise RuntimeError('No locked-regime target games were found.')

    ensure_dir(RAW_PATH.parent)
    ensure_dir(OUT_DIR)
    write_df(targets, 'outputs/orientation_research/pbp_target_games.csv')

    client = CFBDClient()
    try:
        types = pd.json_normalize(client.get('/plays/types'))
    except Exception as exc:
        print(f'Play-type lookup unavailable: {type(exc).__name__}: {exc}')
        types = pd.DataFrame()
    write_df(types, 'outputs/orientation_research/pbp_play_types.csv')

    parts: list[pd.DataFrame] = []
    pairs = targets[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
    for season, week in pairs.itertuples(index=False):
        wanted = set(
            targets.loc[(targets['season'].eq(season)) & (targets['week'].eq(week)), 'game_id']
            .dropna().astype(int).tolist()
        )
        if not wanted:
            continue
        print(f'Pulling play-by-play for {season} week {week}: {len(wanted)} locked target game(s)...')
        records = client.get('/plays', {
            'year': int(season),
            'week': int(week),
            'seasonType': 'regular',
            'classification': 'fbs',
        })
        plays = normalize_plays(records)
        if plays.empty:
            continue
        plays = plays[plays['game_id'].isin(wanted)].copy()
        if plays.empty:
            continue
        plays.insert(0, 'season', int(season))
        plays.insert(1, 'week', int(week))
        parts.append(plays)

    all_plays = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if all_plays.empty:
        raise RuntimeError('CFBD returned no play-by-play rows for the locked target games.')

    all_plays.to_csv(RAW_PATH, index=False)
    coverage = targets[['game_id', 'season', 'week', 'alignment']].copy()
    counts = all_plays.groupby('game_id', as_index=False).size().rename(columns={'size': 'play_rows'})
    coverage = coverage.merge(counts, on='game_id', how='left')
    coverage['play_rows'] = coverage['play_rows'].fillna(0).astype(int)
    coverage['has_pbp'] = coverage['play_rows'].gt(0)
    write_df(coverage, 'outputs/orientation_research/pbp_game_coverage.csv')

    digest = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    manifest = pd.DataFrame([{
        'study_version': cfg['study_version'],
        'protocol_sha256': hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        'target_games': len(targets),
        'games_with_pbp': int(coverage['has_pbp'].sum()),
        'play_rows': len(all_plays),
        'pbp_sha256': digest,
    }])
    write_df(manifest, 'outputs/orientation_research/pbp_pull_manifest.csv')
    print(manifest.to_string(index=False))


if __name__ == '__main__':
    main()
