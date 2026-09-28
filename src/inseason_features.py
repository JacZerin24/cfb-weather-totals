from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import requests

from .build_team_features import clean_name
from .cfbd_client import CFBDClient


ADVANCED_PATHS = [
    'offense.ppa',
    'offense.successRate',
    'offense.explosiveness',
    'offense.pointsPerOpportunity',
    'offense.plays',
    'offense.drives',
    'offense.passingPlays.ppa',
    'offense.passingPlays.successRate',
    'offense.passingPlays.explosiveness',
    'offense.rushingPlays.ppa',
    'offense.rushingPlays.successRate',
    'offense.rushingPlays.explosiveness',
    'offense.stuffRate',
    'offense.powerSuccess',
    'offense.lineYards',
    'offense.secondLevelYards',
    'offense.openFieldYards',
    'defense.ppa',
    'defense.successRate',
    'defense.explosiveness',
    'defense.pointsPerOpportunity',
    'defense.plays',
    'defense.drives',
    'defense.passingPlays.ppa',
    'defense.passingPlays.successRate',
    'defense.passingPlays.explosiveness',
    'defense.rushingPlays.ppa',
    'defense.rushingPlays.successRate',
    'defense.rushingPlays.explosiveness',
    'defense.stuffRate',
    'defense.powerSuccess',
    'defense.lineYards',
    'defense.secondLevelYards',
    'defense.openFieldYards',
]


def _feature_name(path: str) -> str:
    return 'adv_' + clean_name(path.replace('.', '_'))


def normalize_advanced_snapshot(
    records: list[dict[str, Any]],
    season: int,
    target_week: int,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=['season', 'week', 'feature_team'])

    raw = pd.json_normalize(records, sep='.')
    if raw.empty or 'team' not in raw.columns:
        return pd.DataFrame(columns=['season', 'week', 'feature_team'])

    out = pd.DataFrame({
        'season': int(season),
        'week': int(target_week),
        'feature_team': raw['team'].astype(str),
    })
    if 'conference' in raw.columns:
        out['inseason_conference'] = raw['conference'].astype(str)

    for path in ADVANCED_PATHS:
        if path in raw.columns:
            out[_feature_name(path)] = pd.to_numeric(raw[path], errors='coerce')

    out['inseason_through_week'] = int(target_week) - 1
    feature_cols = [c for c in out.columns if c.startswith('adv_')]
    if feature_cols:
        out = out.dropna(subset=feature_cols, how='all')
    return out.drop_duplicates(['season', 'week', 'feature_team'])


def fetch_advanced_snapshot(
    client: CFBDClient,
    season: int,
    target_week: int,
) -> pd.DataFrame:
    if int(target_week) <= 1:
        return pd.DataFrame(columns=['season', 'week', 'feature_team'])
    try:
        records = client.get('/stats/season/advanced', {
            'year': int(season),
            'startWeek': 1,
            'endWeek': int(target_week) - 1,
            'classification': 'fbs',
            'excludeGarbageTime': True,
        })
    except requests.HTTPError as exc:
        print(
            f'WARNING: advanced in-season snapshot unavailable for {season} '
            f'through week {int(target_week) - 1}: {exc}'
        )
        return pd.DataFrame(columns=['season', 'week', 'feature_team'])
    return normalize_advanced_snapshot(records, int(season), int(target_week))


def build_historical_inseason_features(
    client: CFBDClient,
    modeling: pd.DataFrame,
) -> pd.DataFrame:
    seasons = pd.to_numeric(modeling.get('season'), errors='coerce')
    weeks = pd.to_numeric(modeling.get('week'), errors='coerce')
    schedule = (
        pd.DataFrame({'season': seasons, 'week': weeks})
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values(['season', 'week'])
    )

    frames: list[pd.DataFrame] = []
    for season, group in schedule.groupby('season'):
        for week in group['week'].tolist():
            if int(week) <= 1:
                continue
            print(f'Pulling leakage-safe advanced stats for {season} through week {int(week) - 1}...')
            snap = fetch_advanced_snapshot(client, int(season), int(week))
            if not snap.empty:
                frames.append(snap)
    if not frames:
        return pd.DataFrame(columns=['season', 'week', 'feature_team'])
    return pd.concat(frames, ignore_index=True)


def merge_inseason_features(dataset: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return dataset.copy()

    out = dataset.copy()
    out['season'] = pd.to_numeric(out['season'], errors='coerce').astype('Int64')
    out['week'] = pd.to_numeric(out['week'], errors='coerce').astype('Int64')

    feature_cols = [c for c in features.columns if c not in {'season', 'week', 'feature_team'}]
    home = features.rename(
        columns={'feature_team': 'home_team', **{c: f'home_inseason_{c}' for c in feature_cols}}
    )
    away = features.rename(
        columns={'feature_team': 'away_team', **{c: f'away_inseason_{c}' for c in feature_cols}}
    )

    out = out.merge(home, on=['season', 'week', 'home_team'], how='left')
    out = out.merge(away, on=['season', 'week', 'away_team'], how='left')
    return out


def live_inseason_features(
    client: CFBDClient,
    board: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    weeks = pd.to_numeric(board.get('week'), errors='coerce').dropna().astype(int).unique().tolist()
    frames = [fetch_advanced_snapshot(client, int(season), int(week)) for week in sorted(weeks) if int(week) > 1]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=['season', 'week', 'feature_team'])
    return pd.concat(frames, ignore_index=True)
