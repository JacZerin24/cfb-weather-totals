from __future__ import annotations

import re

import pandas as pd

from .utils import read_df, write_df


def clean_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r'(?<!^)(?=[A-Z])', '_', value)
    value = re.sub(r'[^0-9a-zA-Z]+', '_', value).strip('_').lower()
    return value or 'unknown_stat'


def main() -> None:
    stats = read_df('data/raw/cfbd_team_season_stats.csv')
    if stats.empty:
        out = write_df(pd.DataFrame(), 'data/processed/team_prior_features.csv')
        print(f'No team stats available. Wrote empty file to {out}')
        return

    stats = stats.copy()
    if 'statName' in stats.columns and 'stat_name' not in stats.columns:
        stats = stats.rename(columns={'statName': 'stat_name'})
    if 'statValue' in stats.columns and 'stat_value' not in stats.columns:
        stats = stats.rename(columns={'statValue': 'stat_value'})

    required = {'season', 'team', 'stat_name', 'stat_value'}
    missing = required - set(stats.columns)
    if missing:
        raise RuntimeError(f'Missing required team stat columns: {sorted(missing)}')

    stats['stat_name'] = stats['stat_name'].map(clean_name)
    stats['stat_value'] = pd.to_numeric(stats['stat_value'], errors='coerce')
    stats = stats.dropna(subset=['season', 'team', 'stat_name', 'stat_value'])

    pivot = (
        stats.pivot_table(
            index=['season', 'team'],
            columns='stat_name',
            values='stat_value',
            aggfunc='sum',
        )
        .reset_index()
    )
    pivot.columns = [str(c) for c in pivot.columns]

    # Shift stats forward one year so the 2023 row is used as a prior-season feature for 2024 games.
    pivot['season'] = pd.to_numeric(pivot['season'], errors='coerce') + 1
    pivot = pivot.rename(columns={'team': 'feature_team'})

    stat_cols = [c for c in pivot.columns if c not in {'season', 'feature_team'}]
    keep = ['season', 'feature_team'] + stat_cols
    out = write_df(pivot[keep], 'data/processed/team_prior_features.csv')
    print(f'Wrote {len(pivot):,} rows and {len(stat_cols):,} prior-season stat columns to {out}')


if __name__ == '__main__':
    main()
