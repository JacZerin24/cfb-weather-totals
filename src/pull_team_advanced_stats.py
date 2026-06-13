from __future__ import annotations

import pandas as pd
import requests

from .cfbd_client import CFBDClient
from .utils import get_settings, write_df


def main() -> None:
    settings = get_settings()
    client = CFBDClient()
    start = int(settings['data']['start_year']) - 1
    end = int(settings['data']['end_year'])
    season_type = settings['data']['season_type']

    frames = []
    for year in range(start, end + 1):
        print(f'Pulling advanced team stats for {year}...')
        try:
            df = client.get_df('/stats/season/advanced', {'year': year, 'seasonType': season_type})
        except requests.HTTPError as e:
            print(f'WARNING: advanced team stats pull failed for {year}: {e}')
            continue
        if not df.empty:
            df['season'] = year
            frames.append(df)

    stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = write_df(stats, 'data/raw/cfbd_team_advanced_stats.csv')
    print(f'Wrote {len(stats):,} rows to {out}')


if __name__ == '__main__':
    main()
