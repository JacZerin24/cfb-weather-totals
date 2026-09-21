from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from ..utils import load_yaml, write_df


CONNECT_TIMEOUT_SECONDS = 15
READ_TIMEOUT_SECONDS = 90


def main() -> None:
    settings = load_yaml('config/nfl_settings.yml')
    data_cfg = settings['data']
    url = str(data_cfg['source_url'])

    response = requests.get(url, timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS))
    response.raise_for_status()
    games = pd.read_csv(StringIO(response.text), low_memory=False)

    required = {
        'game_id', 'season', 'game_type', 'week', 'gameday', 'away_team', 'home_team',
        'away_score', 'home_score', 'total_line', 'roof', 'temp', 'wind',
    }
    missing = sorted(required - set(games.columns))
    if missing:
        raise RuntimeError(f'nflverse schedule source is missing required columns: {missing}')

    games['season'] = pd.to_numeric(games['season'], errors='coerce')
    start_year = int(data_cfg['start_year'])
    end_year = int(data_cfg['end_year'])
    game_types = {str(value).upper() for value in data_cfg.get('game_types', ['REG'])}

    keep = (
        games['season'].between(start_year, end_year, inclusive='both')
        & games['game_type'].astype(str).str.upper().isin(game_types)
    )
    filtered = games.loc[keep].copy().sort_values(['season', 'week', 'gameday', 'game_id'])
    if filtered.empty:
        raise RuntimeError('NFL historical pull produced zero rows after configured filters.')

    out = write_df(filtered, data_cfg['raw_path'])
    played = filtered['home_score'].notna() & filtered['away_score'].notna()
    with_total = pd.to_numeric(filtered['total_line'], errors='coerce').notna()
    with_weather = (
        pd.to_numeric(filtered['wind'], errors='coerce').notna()
        | pd.to_numeric(filtered['temp'], errors='coerce').notna()
    )
    print(f'Wrote {len(filtered):,} NFL games to {out}')
    print(f'Played games: {int(played.sum()):,}')
    print(f'Games with closing total: {int(with_total.sum()):,}')
    print(f'Games with temperature and/or wind: {int(with_weather.sum()):,}')


if __name__ == '__main__':
    main()
