from __future__ import annotations

import argparse

import pandas as pd

from .cfbd_client import CFBDClient
from .utils import read_df, write_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pull CFBD team box-score stats for historical FBS game weeks used by orientation research.')
    parser.add_argument('--output', default='data/raw/cfbd_game_team_stats.csv')
    return parser.parse_args()


def flatten(records: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for game in records:
        gid = game.get('id')
        for team in game.get('teams') or []:
            base = {
                'game_id': gid,
                'team_id': team.get('teamId'),
                'team': team.get('team'),
                'conference': team.get('conference'),
                'home_away': team.get('homeAway'),
                'points': team.get('points'),
            }
            for stat in team.get('stats') or []:
                rows.append({**base, 'category': stat.get('category'), 'stat': stat.get('stat')})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    games = read_df('data/raw/cfbd_games.csv').copy()
    if games.empty:
        raise RuntimeError('Historical games file is empty.')

    games['season'] = pd.to_numeric(games.get('season'), errors='coerce')
    games['week'] = pd.to_numeric(games.get('week'), errors='coerce')
    h = games.get('homeClassification', pd.Series('', index=games.index)).astype(str).str.lower()
    a = games.get('awayClassification', pd.Series('', index=games.index)).astype(str).str.lower()
    weeks = (
        games[h.eq('fbs') & a.eq('fbs')]
        .dropna(subset=['season', 'week'])[['season', 'week']]
        .drop_duplicates()
        .sort_values(['season', 'week'])
    )

    client = CFBDClient()
    frames: list[pd.DataFrame] = []
    for row in weeks.itertuples(index=False):
        year, week = int(row.season), int(row.week)
        print(f'Pulling FBS team box stats: {year} week {week}...')
        records = client.get('/games/teams', {
            'year': year,
            'week': week,
            'seasonType': 'regular',
            'classification': 'fbs',
        })
        frame = flatten(records)
        if not frame.empty:
            frame['season'] = year
            frame['week'] = week
            frames.append(frame)

    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = write_df(out_df, args.output)
    print(f'Wrote {len(out_df):,} long-format team-stat rows to {out}')
    if not out_df.empty:
        cats = sorted(out_df['category'].dropna().astype(str).unique())
        print('Available stat categories: ' + ', '.join(cats))


if __name__ == '__main__':
    main()
