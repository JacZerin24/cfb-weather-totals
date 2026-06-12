from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import read_df, write_df, get_settings


def pick_total(lines: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(columns=['game_id', 'closing_total', 'line_provider'])
    lines = lines.copy()
    lines['over_under'] = pd.to_numeric(lines['over_under'], errors='coerce')
    lines = lines.dropna(subset=['game_id', 'over_under'])
    if lines.empty:
        return pd.DataFrame(columns=['game_id', 'closing_total', 'line_provider'])

    provider_rank = {p: i for i, p in enumerate(preferred)}
    lines['provider_rank'] = lines['provider'].map(provider_rank).fillna(999).astype(int)
    best = (
        lines.sort_values(['game_id', 'provider_rank'])
        .groupby('game_id', as_index=False)
        .first()[['game_id', 'over_under', 'provider']]
        .rename(columns={'over_under': 'closing_total', 'provider': 'line_provider'})
    )
    return best


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def main() -> None:
    settings = get_settings()
    preferred = settings['cfbd']['preferred_line_providers']

    games = read_df('data/raw/cfbd_games.csv')
    lines = read_df('data/raw/cfbd_lines.csv')
    weather = read_df('data/raw/cfbd_weather.csv')

    games = games.copy()
    game_id_col = first_existing(games, ['id', 'game_id'])
    if game_id_col is None:
        raise RuntimeError('Could not find game id column in games data.')
    games = games.rename(columns={game_id_col: 'game_id'})

    for col in ['home_points', 'away_points']:
        if col not in games.columns:
            raise RuntimeError(f'Missing required game score column: {col}')
    games['actual_total_points'] = pd.to_numeric(games['home_points'], errors='coerce') + pd.to_numeric(games['away_points'], errors='coerce')

    totals = pick_total(lines, preferred)

    weather = weather.copy()
    weather_id_col = first_existing(weather, ['id', 'game_id'])
    if weather_id_col:
        weather = weather.rename(columns={weather_id_col: 'game_id'})
    else:
        weather['game_id'] = np.nan

    keep_game_cols = [c for c in ['game_id', 'season', 'week', 'season_type', 'start_date', 'home_team', 'away_team', 'venue', 'conference_game', 'neutral_site', 'actual_total_points'] if c in games.columns]
    dataset = games[keep_game_cols].merge(totals, on='game_id', how='left')

    weather_cols = [c for c in weather.columns if c == 'game_id' or c.startswith(('temperature', 'humidity', 'precipitation', 'wind', 'weather'))]
    if 'game_id' in weather_cols:
        dataset = dataset.merge(weather[weather_cols].drop_duplicates('game_id'), on='game_id', how='left')

    dataset['closing_total'] = pd.to_numeric(dataset['closing_total'], errors='coerce')
    dataset['market_residual'] = dataset['actual_total_points'] - dataset['closing_total']
    dataset['went_over'] = dataset['actual_total_points'] > dataset['closing_total']
    dataset['went_under'] = dataset['actual_total_points'] < dataset['closing_total']
    dataset['push'] = dataset['actual_total_points'] == dataset['closing_total']

    wind_col = first_existing(dataset, ['wind_speed', 'windSpeed', 'wind_speed_mph'])
    if wind_col:
        dataset['wind_mph'] = pd.to_numeric(dataset[wind_col], errors='coerce')
        dataset['wind_bin'] = pd.cut(dataset['wind_mph'], bins=[-1, 5, 10, 15, 20, 200], labels=['0-5', '5-10', '10-15', '15-20', '20+'])

    temp_col = first_existing(dataset, ['temperature', 'temperature_f', 'temp'])
    if temp_col:
        dataset['temperature_f'] = pd.to_numeric(dataset[temp_col], errors='coerce')
        dataset['temp_bin'] = pd.cut(dataset['temperature_f'], bins=[-100, 35, 50, 70, 85, 200], labels=['<=35', '35-50', '50-70', '70-85', '85+'])

    out = write_df(dataset, 'data/processed/modeling_dataset.csv')
    print(f'Wrote {len(dataset):,} rows to {out}')
    print(f"Rows with closing totals: {dataset['closing_total'].notna().sum():,}")


if __name__ == '__main__':
    main()
