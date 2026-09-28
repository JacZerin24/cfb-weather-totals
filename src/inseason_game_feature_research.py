from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error

from .cfbd_client import CFBDClient
from .deep_research import prep
from .inseason_feature_research import grade_under_screen, run_variant
from .inseason_features import merge_inseason_features
from .utils import ensure_dir, read_df, write_df


SIDE_FIELDS = ('offense', 'defense')


def _numeric(raw: pd.DataFrame, name: str) -> pd.Series:
    if name not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype=float)
    return pd.to_numeric(raw[name], errors='coerce')


def season_snapshots(records: list[dict], season: int) -> pd.DataFrame:
    raw = pd.json_normalize(records, sep='.')
    if raw.empty or not {'team', 'week'} <= set(raw.columns):
        return pd.DataFrame(columns=['season', 'week', 'feature_team'])

    raw['season'] = int(season)
    raw['week'] = pd.to_numeric(raw['week'], errors='coerce')
    raw = raw.dropna(subset=['week', 'team']).copy()
    raw['week'] = raw['week'].astype(int)

    pieces = []
    for side in SIDE_FIELDS:
        plays = _numeric(raw, f'{side}.plays').fillna(0.0)
        drives = _numeric(raw, f'{side}.drives').fillna(0.0)
        total_ppa = _numeric(raw, f'{side}.totalPPA').fillna(0.0)
        success = _numeric(raw, f'{side}.successRate')
        pieces.append(pd.DataFrame({
            'season': int(season),
            'week': raw['week'],
            'feature_team': raw['team'].astype(str),
            'games': 1.0,
            f'{side}_plays': plays,
            f'{side}_drives': drives,
            f'{side}_total_ppa': total_ppa,
            f'{side}_successes_est': success.fillna(0.0) * plays,
        }))

    weekly = pieces[0].merge(
        pieces[1],
        on=['season', 'week', 'feature_team', 'games'],
        how='outer',
    )
    value_cols = [c for c in weekly.columns if c not in {'season', 'week', 'feature_team'}]
    weekly = weekly.groupby(['season', 'week', 'feature_team'], as_index=False)[value_cols].sum()

    rows = []
    for team, group in weekly.groupby('feature_team'):
        group = group.sort_values('week').copy()
        cumulative = group[value_cols].cumsum().shift(1).fillna(0.0)
        for idx, row in group.iterrows():
            pos = group.index.get_loc(idx)
            cur = cumulative.iloc[pos]
            games = float(cur.get('games', 0.0))
            if games <= 0:
                continue
            rec = {
                'season': int(season),
                'week': int(row['week']),
                'feature_team': str(team),
                'adv_games_played': games,
            }
            for side in SIDE_FIELDS:
                plays = float(cur.get(f'{side}_plays', 0.0))
                drives = float(cur.get(f'{side}_drives', 0.0))
                total_ppa = float(cur.get(f'{side}_total_ppa', 0.0))
                successes = float(cur.get(f'{side}_successes_est', 0.0))
                rec[f'adv_{side}_ppa'] = total_ppa / plays if plays > 0 else np.nan
                rec[f'adv_{side}_success_rate'] = successes / plays if plays > 0 else np.nan
                rec[f'adv_{side}_plays_per_drive'] = plays / drives if drives > 0 else np.nan
                rec[f'adv_{side}_plays'] = plays
                rec[f'adv_{side}_drives'] = drives
            rows.append(rec)
    return pd.DataFrame(rows)


def pull_features(client: CFBDClient, modeling: pd.DataFrame) -> pd.DataFrame:
    frames = []
    seasons = sorted(pd.to_numeric(modeling['season'], errors='coerce').dropna().astype(int).unique())
    for season in seasons:
        print(f'Pulling game-level advanced stats for {season}...')
        try:
            records = client.get('/stats/game/advanced', {
                'year': int(season),
                'seasonType': 'regular',
                'excludeGarbageTime': True,
            })
        except requests.HTTPError as exc:
            print(f'WARNING: game-level advanced stats unavailable for {season}: {exc}')
            continue
        snap = season_snapshots(records, int(season))
        if not snap.empty:
            frames.append(snap)
    if not frames:
        return pd.DataFrame(columns=['season', 'week', 'feature_team'])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    base = prep(read_df('data/processed/modeling_dataset.csv'))
    features = pull_features(CFBDClient(), base)
    write_df(features, 'data/processed/team_inseason_game_advanced_features.csv')
    enriched = merge_inseason_features(base, features)

    current_cols = [c for c in enriched.columns if c.startswith(('home_inseason_adv_', 'away_inseason_adv_'))]
    enriched['inseason_feature_ready'] = enriched[current_cols].notna().any(axis=1) if current_cols else False

    summaries = []
    seasons = []
    diagnostics = []
    for variant in ['baseline_prior', 'inseason_only', 'hybrid']:
        pred, diag, seasonal = run_variant(enriched, variant)
        if pred.empty:
            continue
        row = grade_under_screen(pred, variant)
        row['mae'] = mean_absolute_error(pred['market_residual'], pred['pred_market_residual'])
        row['test_games'] = len(pred)
        row['inseason_ready_games'] = int(pred.get('inseason_feature_ready', pd.Series(False, index=pred.index)).sum())
        summaries.append(row)
        seasons.extend(seasonal)
        diagnostics.extend(diag)

    summary = pd.DataFrame(summaries)
    by_season = pd.DataFrame(seasons)
    diag_df = pd.DataFrame(diagnostics)
    write_df(summary, 'outputs/inseason_game_feature_experiment.csv')
    write_df(by_season, 'outputs/inseason_game_feature_experiment_by_season.csv')

    lines = [
        '# In-Season Game-Level Feature Experiment',
        '',
        'Leakage-safe current-season features are reconstructed from game-level advanced stats, cumulatively shifted so a week-N game uses only earlier weeks.',
        '',
        'Features: offense/defense PPA, success rate, plays per drive, cumulative plays/drives, and games played.',
        '',
        '## Overall production-screen comparison',
        '',
        summary.to_markdown(index=False) if not summary.empty else '_No results._',
        '',
        '## By-season production-screen comparison',
        '',
        by_season.to_markdown(index=False) if not by_season.empty else '_No results._',
        '',
        '## Diagnostics',
        '',
        diag_df.to_markdown(index=False) if not diag_df.empty else '_No diagnostics._',
    ]
    (ensure_dir('outputs') / 'inseason_game_feature_experiment.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
