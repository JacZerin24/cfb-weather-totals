from __future__ import annotations

import pandas as pd

from .utils import read_df, write_df, ensure_dir, get_settings


def summarize_group(df: pd.DataFrame, group_col: str, min_games: int) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    g = (
        df.dropna(subset=[group_col, 'closing_total', 'market_residual'])
        .groupby(group_col, observed=True)
        .agg(
            games=('game_id', 'count'),
            avg_closing_total=('closing_total', 'mean'),
            avg_actual_total=('actual_total_points', 'mean'),
            avg_market_residual=('market_residual', 'mean'),
            under_rate=('went_under', 'mean'),
            over_rate=('went_over', 'mean'),
        )
        .reset_index()
    )
    return g[g['games'] >= min_games].sort_values('avg_market_residual')


def main() -> None:
    settings = get_settings()
    min_games = int(settings['modeling']['min_games_for_group'])
    df = read_df('data/processed/modeling_dataset.csv')

    summaries = []
    for col in ['wind_bin', 'temp_bin']:
        s = summarize_group(df, col, min_games)
        if not s.empty:
            s.insert(0, 'grouping', col)
            summaries.append(s)

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if not summary.empty:
        write_df(summary, 'outputs/weather_group_summary.csv')

    ensure_dir('outputs')
    lines = ['# Weather Totals Research Summary', '']
    lines.append(f'Total games in dataset: {len(df):,}')
    lines.append(f"Games with closing totals: {df['closing_total'].notna().sum():,}")
    lines.append('')
    if summary.empty:
        lines.append('No weather group summaries were available yet. Check weather column names after pulling data.')
    else:
        lines.append('## Group summaries')
        lines.append('')
        lines.append(summary.to_markdown(index=False))
    out = ensure_dir('outputs') / 'research_summary.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
