from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import ensure_dir, load_yaml, read_df, write_df


BREAKEVEN = 110 / 210


def _units(result: pd.Series) -> float:
    return float(result.map({'win': 100 / 110, 'loss': -1.0, 'push': 0.0}).fillna(0).sum())


def _settle(df: pd.DataFrame, side: str) -> pd.Series:
    residual = df['market_residual']
    win = residual.lt(0) if side == 'under' else residual.gt(0)
    return pd.Series(
        np.where(residual.eq(0), 'push', np.where(win, 'win', 'loss')),
        index=df.index,
    )


def _wilson(wins: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = 1.96
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def _rule_row(name: str, side: str, sample: pd.DataFrame) -> dict:
    result = _settle(sample, side)
    wins = int(result.eq('win').sum())
    losses = int(result.eq('loss').sum())
    n = wins + losses
    low, high = _wilson(wins, n)
    units = _units(result)
    return {
        'rule': name,
        'side': side,
        'games': int(len(sample)),
        'graded': n,
        'wins': wins,
        'losses': losses,
        'pushes': int(result.eq('push').sum()),
        'hit_rate': wins / n if n else np.nan,
        'hit_rate_minus_breakeven': wins / n - BREAKEVEN if n else np.nan,
        'wilson_low': low,
        'wilson_high': high,
        'net_units_1u_each': units,
        'roi_per_1u': units / n if n else np.nan,
        'avg_market_residual': float(sample['market_residual'].mean()) if len(sample) else np.nan,
        'avg_actual_total': float(sample['actual_total_points'].mean()) if len(sample) else np.nan,
        'avg_closing_total': float(sample['closing_total'].mean()) if len(sample) else np.nan,
    }


def main() -> None:
    settings = load_yaml('config/nfl_settings.yml')
    df = read_df(settings['data']['processed_path']).copy()
    df = df.dropna(subset=['season', 'closing_total', 'actual_total_points', 'market_residual'])
    outdoor = df[df['outdoor'].fillna(False)].copy()

    group_rows: list[dict] = []
    for column in ['wind_bin', 'temp_bin', 'roof', 'total_bin']:
        if column not in df.columns:
            continue
        source = outdoor if column in {'wind_bin', 'temp_bin'} else df
        for value, group in source.dropna(subset=[column]).groupby(column, observed=True):
            if len(group) < 25:
                continue
            group_rows.append({
                'grouping': column,
                'value': str(value),
                'games': int(len(group)),
                'avg_closing_total': float(group['closing_total'].mean()),
                'avg_actual_total': float(group['actual_total_points'].mean()),
                'avg_market_residual': float(group['market_residual'].mean()),
                'under_rate': float(group['went_under'].mean()),
                'over_rate': float(group['went_over'].mean()),
            })
    groups = pd.DataFrame(group_rows)

    rules = [
        ('outdoor_wind_10_under', 'under', outdoor['wind_mph'] >= 10),
        ('outdoor_wind_15_under', 'under', outdoor['wind_mph'] >= 15),
        ('outdoor_wind_20_under', 'under', outdoor['wind_mph'] >= 20),
        (
            'outdoor_wind_15_total_45plus_under',
            'under',
            (outdoor['wind_mph'] >= 15) & (outdoor['closing_total'] >= 45),
        ),
        ('outdoor_freezing_under', 'under', outdoor['temperature_f'] <= 32),
        (
            'outdoor_cold_windy_under',
            'under',
            (outdoor['temperature_f'] <= 40) & (outdoor['wind_mph'] >= 12),
        ),
        (
            'outdoor_calm_mild_over',
            'over',
            (outdoor['wind_mph'] <= 5) & outdoor['temperature_f'].between(50, 80),
        ),
        ('indoor_over', 'over', df['game_indoors_bool'].fillna(False)),
    ]
    rule_rows = []
    for name, side, mask in rules:
        base = outdoor if name.startswith('outdoor_') else df
        sample = base.loc[mask.fillna(False)].copy()
        rule_rows.append(_rule_row(name, side, sample))
    rule_table = pd.DataFrame(rule_rows)

    by_season = (
        df.groupby('season', as_index=False)
        .agg(
            games=('game_id', 'count'),
            avg_closing_total=('closing_total', 'mean'),
            avg_actual_total=('actual_total_points', 'mean'),
            avg_market_residual=('market_residual', 'mean'),
            outdoor_games=('outdoor', 'sum'),
        )
    )

    write_df(groups, 'outputs/nfl/weather_group_summary.csv')
    write_df(rule_table, 'outputs/nfl/simple_rule_screen.csv')
    write_df(by_season, 'outputs/nfl/data_quality_by_season.csv')

    lines = [
        '# NFL Weather / Totals Baseline Research',
        '',
        'Target: `market_residual = actual_total_points - closing_total`.',
        '',
        f'Usable games: {len(df):,}',
        f"Seasons: {int(df['season'].min())}-{int(df['season'].max())}",
        f'Outdoor games: {int(df["outdoor"].sum()):,}',
        '',
        '## Why residuals matter',
        '',
        'Raw scoring differences are not enough to establish a betting edge because the closing market can already price weather. The key question is whether a condition is associated with a systematic residual after subtracting the closing total.',
        '',
        '## Weather groups',
        '',
        groups.to_markdown(index=False) if not groups.empty else '_No group results._',
        '',
        '## Simple rule screens',
        '',
        rule_table.to_markdown(index=False),
        '',
        '## Guardrails',
        '',
        '- These rule rows are exploratory screens, not production thresholds.',
        '- The nflverse temperature/wind fields describe observed game conditions. They are useful for effect discovery but are not yet a forecast-realistic backtest.',
        '- Any candidate edge must survive chronological testing, season sensitivity, threshold perturbation, and a later forecast-at-lead-time validation before it can become a weekly signal.',
        '- No NFL production qualifier is enabled by this research scaffold.',
    ]
    out = ensure_dir('outputs/nfl') / 'weather_research_summary.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote NFL weather research outputs under {out.parent}')


if __name__ == '__main__':
    main()
