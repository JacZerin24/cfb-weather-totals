from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ROOT, get_settings, read_df, write_df


def pick_total(lines: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(columns=['game_id', 'closing_total', 'line_provider'])
    lines = lines.copy()
    if 'id' in lines.columns and 'game_id' not in lines.columns:
        lines = lines.rename(columns={'id': 'game_id'})
    if 'overUnder' in lines.columns and 'over_under' not in lines.columns:
        lines = lines.rename(columns={'overUnder': 'over_under'})
    if 'game_id' not in lines.columns or 'over_under' not in lines.columns:
        return pd.DataFrame(columns=['game_id', 'closing_total', 'line_provider'])
    lines['over_under'] = pd.to_numeric(lines['over_under'], errors='coerce')
    lines = lines.dropna(subset=['game_id', 'over_under'])
    if lines.empty:
        return pd.DataFrame(columns=['game_id', 'closing_total', 'line_provider'])

    provider_rank = {p.lower(): i for i, p in enumerate(preferred)}
    if 'provider' not in lines.columns:
        lines['provider'] = 'unknown'
    lines['provider_rank'] = lines['provider'].astype(str).str.lower().map(provider_rank).fillna(999).astype(int)
    best = (
        lines.sort_values(['game_id', 'provider_rank'])
        .groupby('game_id', as_index=False)
        .first()[['game_id', 'over_under', 'provider']]
        .rename(columns={'over_under': 'closing_total', 'provider': 'line_provider'})
    )
    return best


def merge_prior_team_features(dataset: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / 'data/processed/team_prior_features.csv'
    if not path.exists():
        print('No prior team feature file found. Skipping team stat controls.')
        return dataset
    feats = read_df(path)
    if feats.empty:
        print('Prior team feature file is empty. Skipping team stat controls.')
        return dataset

    dataset = dataset.copy()
    stat_cols = [c for c in feats.columns if c not in {'season', 'feature_team'}]
    if not stat_cols:
        return dataset

    home = feats.rename(columns={'feature_team': 'home_team'})
    home = home.rename(columns={c: f'home_prior_{c}' for c in stat_cols})
    dataset = dataset.merge(home, on=['season', 'home_team'], how='left')

    away = feats.rename(columns={'feature_team': 'away_team'})
    away = away.rename(columns={c: f'away_prior_{c}' for c in stat_cols})
    dataset = dataset.merge(away, on=['season', 'away_team'], how='left')

    added = len([c for c in dataset.columns if c.startswith(('home_prior_', 'away_prior_'))])
    print(f'Added {added:,} prior-season team stat feature columns.')
    return dataset


def main() -> None:
    settings = get_settings()
    preferred = settings['cfbd']['preferred_line_providers']

    games = read_df('data/raw/cfbd_games.csv').copy()
    lines = read_df('data/raw/cfbd_lines.csv')
    weather = read_df('data/raw/cfbd_weather.csv').copy()

    games = games.rename(columns={
        'id': 'game_id',
        'homePoints': 'home_points',
        'awayPoints': 'away_points',
        'homeTeam': 'home_team',
        'awayTeam': 'away_team',
        'seasonType': 'season_type',
        'startDate': 'start_date',
        'conferenceGame': 'conference_game',
        'neutralSite': 'neutral_site',
        'homeClassification': 'home_classification',
        'awayClassification': 'away_classification',
        'homeConference': 'home_conference',
        'awayConference': 'away_conference',
        'homePregameElo': 'home_pregame_elo',
        'awayPregameElo': 'away_pregame_elo',
        'venueId': 'venue_id',
    })
    if 'game_id' not in games.columns:
        raise RuntimeError('Could not find game id column in games data.')

    for col in ['home_points', 'away_points']:
        if col not in games.columns:
            raise RuntimeError(f'Missing required game score column: {col}')
    games['actual_total_points'] = pd.to_numeric(games['home_points'], errors='coerce') + pd.to_numeric(games['away_points'], errors='coerce')

    totals = pick_total(lines, preferred)

    weather = weather.rename(columns={
        'id': 'game_id',
        'gameId': 'game_id',
        'gameIndoors': 'game_indoors',
        'windSpeed': 'wind_mph',
        'windDirection': 'wind_direction_degrees',
        'temperature': 'temperature_f',
        'dewPoint': 'dewpoint_f',
        'weatherCondition': 'weather_condition',
        'weatherConditionCode': 'weather_condition_code',
    })
    if 'game_id' not in weather.columns:
        weather['game_id'] = np.nan

    keep_game_cols = [c for c in [
        'game_id', 'season', 'week', 'season_type', 'start_date',
        'home_team', 'away_team', 'venue_id', 'venue',
        'home_classification', 'away_classification',
        'home_conference', 'away_conference',
        'home_pregame_elo', 'away_pregame_elo',
        'attendance', 'conference_game', 'neutral_site',
        'actual_total_points',
    ] if c in games.columns]
    dataset = games[keep_game_cols].merge(totals, on='game_id', how='left')

    weather_cols = [c for c in ['game_id', 'game_indoors', 'temperature_f', 'dewpoint_f', 'humidity', 'precipitation', 'snowfall', 'wind_direction_degrees', 'wind_mph', 'pressure', 'weather_condition_code', 'weather_condition'] if c in weather.columns]
    if 'game_id' in weather_cols:
        dataset = dataset.merge(weather[weather_cols].drop_duplicates('game_id'), on='game_id', how='left')

    dataset = merge_prior_team_features(dataset)

    dataset['closing_total'] = pd.to_numeric(dataset['closing_total'], errors='coerce')
    dataset['market_residual'] = dataset['actual_total_points'] - dataset['closing_total']
    dataset['went_over'] = dataset['actual_total_points'] > dataset['closing_total']
    dataset['went_under'] = dataset['actual_total_points'] < dataset['closing_total']
    dataset['push'] = dataset['actual_total_points'] == dataset['closing_total']

    if 'wind_mph' in dataset.columns:
        dataset['wind_mph'] = pd.to_numeric(dataset['wind_mph'], errors='coerce')
        dataset['wind_bin'] = pd.cut(dataset['wind_mph'], bins=[-1, 5, 10, 15, 20, 200], labels=['0-5', '5-10', '10-15', '15-20', '20+'])

    if 'temperature_f' in dataset.columns:
        dataset['temperature_f'] = pd.to_numeric(dataset['temperature_f'], errors='coerce')
        dataset['temp_bin'] = pd.cut(dataset['temperature_f'], bins=[-100, 35, 50, 70, 85, 200], labels=['<=35', '35-50', '50-70', '70-85', '85+'])

    out = write_df(dataset, 'data/processed/modeling_dataset.csv')
    print(f'Wrote {len(dataset):,} rows to {out}')
    print(f"Rows with closing totals: {dataset['closing_total'].notna().sum():,}")


if __name__ == '__main__':
    main()
