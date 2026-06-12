from __future__ import annotations

import pandas as pd

from .cfbd_client import CFBDClient
from .utils import get_settings, write_df


def flatten_lines(records: list[dict]) -> pd.DataFrame:
    rows = []
    for game in records:
        game_id = game.get('id')
        base = {k: v for k, v in game.items() if k != 'lines'}
        for line in game.get('lines') or []:
            row = dict(base)
            row['game_id'] = game_id
            row['provider'] = line.get('provider')
            row['spread'] = line.get('spread')
            row['formatted_spread'] = line.get('formattedSpread')
            row['over_under'] = line.get('overUnder')
            row['home_moneyline'] = line.get('homeMoneyline')
            row['away_moneyline'] = line.get('awayMoneyline')
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    settings = get_settings()
    client = CFBDClient()
    start = int(settings['data']['start_year'])
    end = int(settings['data']['end_year'])
    season_type = settings['data']['season_type']

    frames = []
    for year in range(start, end + 1):
        print(f'Pulling betting lines for {year}...')
        records = client.get('/lines', {'year': year, 'seasonType': season_type})
        df = flatten_lines(records)
        if not df.empty:
            df['season'] = year
            frames.append(df)

    lines = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = write_df(lines, 'data/raw/cfbd_lines.csv')
    print(f'Wrote {len(lines):,} rows to {out}')


if __name__ == '__main__':
    main()
