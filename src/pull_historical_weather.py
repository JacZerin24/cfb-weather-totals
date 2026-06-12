from __future__ import annotations

import pandas as pd

from .cfbd_client import CFBDClient
from .utils import get_settings, write_df


def main() -> None:
    settings = get_settings()
    client = CFBDClient()
    start = int(settings['data']['start_year'])
    end = int(settings['data']['end_year'])
    season_type = settings['data']['season_type']

    frames = []
    for year in range(start, end + 1):
        print(f'Pulling game weather for {year}...')
        df = client.get_df('/games/weather', {'year': year, 'seasonType': season_type})
        if not df.empty:
            df['season'] = year
            frames.append(df)

    weather = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = write_df(weather, 'data/raw/cfbd_weather.csv')
    print(f'Wrote {len(weather):,} rows to {out}')


if __name__ == '__main__':
    main()
