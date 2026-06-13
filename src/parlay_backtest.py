from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable

import numpy as np
import pandas as pd

from .deep_research import BREAKEVEN, wilson
from .utils import ensure_dir, read_df, write_df

AMERICAN_PRICE = -110
LEG_DECIMAL_ODDS = 1 + 100 / abs(AMERICAN_PRICE)
MAX_COMBOS_PER_WEEK = 25000


def parlay_units(results: list[str]) -> float:
    wins = sum(r == 'win' for r in results)
    losses = sum(r == 'loss' for r in results)
    pushes = sum(r == 'push' for r in results)
    if losses:
        return -1.0
    if wins == 0 and pushes > 0:
        return 0.0
    return float((LEG_DECIMAL_ODDS ** wins) - 1)


def summarize(name: str, legs: int, parlays: pd.DataFrame) -> dict:
    if parlays.empty:
        return {'strategy': name, 'legs': legs, 'parlays': 0}
    graded = parlays[parlays['result'] != 'push'].copy()
    wins = int((graded['result'] == 'win').sum())
    losses = int((graded['result'] == 'loss').sum())
    pushes = int((parlays['result'] == 'push').sum())
    n = wins + losses
    hit = wins / n if n else np.nan
    low, high = wilson(wins, n)
    net = float(parlays['units'].sum())
    return {
        'strategy': name,
        'legs': legs,
        'parlays': int(len(parlays)),
        'graded': int(n),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': hit,
        'breakeven_hit_rate_at_minus_110_legs': 1 / (LEG_DECIMAL_ODDS ** legs),
        'hit_rate_wilson_low': low,
        'hit_rate_wilson_high': high,
        'net_units_1u_parlay': net,
        'roi_per_1u_parlay': net / n if n else np.nan,
        'avg_abs_pred_edge_per_leg': float(parlays['avg_abs_pred_edge'].mean()) if len(parlays) else np.nan,
        'avg_market_residual_per_leg': float(parlays['avg_market_residual'].mean()) if len(parlays) else np.nan,
    }


def normalize_plays(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['threshold'] = pd.to_numeric(out['threshold'], errors='coerce')
    out['season'] = pd.to_numeric(out['season'], errors='coerce').astype('Int64')
    out['week'] = pd.to_numeric(out.get('week'), errors='coerce').astype('Int64') if 'week' in out.columns else pd.Series([pd.NA] * len(out), dtype='Int64')
    out['game_id'] = out['game_id'].astype(str) if 'game_id' in out.columns else out.index.astype(str)
    out['abs_pred_edge'] = pd.to_numeric(out['pred_market_residual'], errors='coerce').abs()
    out['market_residual'] = pd.to_numeric(out['market_residual'], errors='coerce')
    return out


def strategy_filters() -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    high_total = ['56-63', '63+']
    return {
        'hgb_under_3p5_all': lambda d: d[(d['model'] == 'hist_gradient_boosting') & (d['threshold'] == 3.5) & (d['side'] == 'under')],
        'hgb_under_3p5_consensus': lambda d: d[(d['model'] == 'hist_gradient_boosting') & (d['threshold'] == 3.5) & (d['side'] == 'under') & (d['line_provider'] == 'consensus')],
        'hgb_under_3p5_high_total': lambda d: d[(d['model'] == 'hist_gradient_boosting') & (d['threshold'] == 3.5) & (d['side'] == 'under') & (d['total_bin'].isin(high_total))],
        'hgb_under_5p0_all': lambda d: d[(d['model'] == 'hist_gradient_boosting') & (d['threshold'] == 5.0) & (d['side'] == 'under')],
        'hgb_under_5p0_consensus': lambda d: d[(d['model'] == 'hist_gradient_boosting') & (d['threshold'] == 5.0) & (d['side'] == 'under') & (d['line_provider'] == 'consensus')],
        'hgb_under_5p0_high_total': lambda d: d[(d['model'] == 'hist_gradient_boosting') & (d['threshold'] == 5.0) & (d['side'] == 'under') & (d['total_bin'].isin(high_total))],
        'hgb_all_sides_5p0_consensus': lambda d: d[(d['model'] == 'hist_gradient_boosting') & (d['threshold'] == 5.0) & (d['line_provider'] == 'consensus')],
    }


def build_parlays(strategy_name: str, plays: pd.DataFrame, legs: int) -> pd.DataFrame:
    rows = []
    if plays.empty:
        return pd.DataFrame()
    keys = ['season', 'week']
    for (season, week), group in plays.groupby(keys, dropna=False):
        group = group.drop_duplicates('game_id').copy()
        group = group.sort_values('abs_pred_edge', ascending=False)
        n = len(group)
        if n < legs:
            continue
        if comb(n, legs) > MAX_COMBOS_PER_WEEK:
            # Keep the strongest candidate legs if an extreme week creates too many combinations.
            cap = legs + 18
            group = group.head(cap).copy()
        records = group.to_dict('records')
        for combo in combinations(records, legs):
            results = [r['result'] for r in combo]
            unit_result = parlay_units(results)
            if unit_result > 0:
                result = 'win'
            elif unit_result < 0:
                result = 'loss'
            else:
                result = 'push'
            rows.append({
                'strategy': strategy_name,
                'legs': legs,
                'season': int(season) if pd.notna(season) else None,
                'week': int(week) if pd.notna(week) else None,
                'game_ids': '|'.join(str(r['game_id']) for r in combo),
                'matchups': ' | '.join(f"{r.get('away_team','')} at {r.get('home_team','')}" for r in combo),
                'sides': '|'.join(str(r['side']) for r in combo),
                'results': '|'.join(results),
                'result': result,
                'units': unit_result,
                'avg_abs_pred_edge': float(np.mean([r['abs_pred_edge'] for r in combo])),
                'min_abs_pred_edge': float(np.min([r['abs_pred_edge'] for r in combo])),
                'avg_market_residual': float(np.mean([r['market_residual'] for r in combo])),
            })
    return pd.DataFrame(rows)


def by_season(parlays: pd.DataFrame) -> pd.DataFrame:
    if parlays.empty:
        return pd.DataFrame()
    rows = []
    for (strategy, legs, season), g in parlays.groupby(['strategy', 'legs', 'season'], dropna=False):
        rows.append(summarize(f'{strategy}__{season}', int(legs), g) | {'base_strategy': strategy, 'season': season})
    return pd.DataFrame(rows)


def write_summary(summary: pd.DataFrame, season_summary: pd.DataFrame) -> None:
    out = ensure_dir('outputs') / 'parlay_validation_summary.md'
    lines = [
        '# Parlay Backtest Summary',
        '',
        'Historical parlay tests use the model edge plays generated by `validate_model_edges.py`.',
        '',
        'Assumption: each leg is priced at -110. A 2-leg parlay wins about +2.64 units per 1 unit risked if both legs win. Pushes reduce the parlay to the remaining live legs.',
        '',
        '## Overall parlay results',
        '',
    ]
    if summary.empty:
        lines.append('_No parlay results generated._')
    else:
        lines.append(summary.sort_values(['roi_per_1u_parlay', 'graded'], ascending=[False, False]).to_markdown(index=False))
    lines.extend(['', '## Season-by-season results', ''])
    if season_summary.empty:
        lines.append('_No season-by-season rows generated._')
    else:
        lines.append(season_summary.sort_values(['base_strategy', 'legs', 'season']).to_markdown(index=False))
    lines.extend([
        '',
        '## Interpretation guardrails',
        '',
        '- Parlays are tested only as a research comparison against straight betting.',
        '- Positive ROI does not automatically mean parlays are better; they increase variance and can hide weak individual legs.',
        '- Prefer 2-leg results with large samples and stable season-by-season performance.',
        '- Do not use parlays for Week 1 unless treated as entertainment or micro-stakes testing.',
    ])
    out.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    plays = normalize_plays(read_df('outputs/model_edge_plays.csv'))
    all_parlays = []
    summary_rows = []
    for name, filt in strategy_filters().items():
        strategy_plays = filt(plays).copy()
        for legs in [2, 3]:
            parlays = build_parlays(name, strategy_plays, legs)
            if not parlays.empty:
                all_parlays.append(parlays)
            summary_rows.append(summarize(name, legs, parlays))
    parlay_df = pd.concat(all_parlays, ignore_index=True) if all_parlays else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    season_summary = by_season(parlay_df)
    write_df(parlay_df, 'outputs/parlay_backtest_plays.csv')
    write_df(summary, 'outputs/parlay_backtest_summary.csv')
    write_df(season_summary, 'outputs/parlay_by_season.csv')
    write_summary(summary, season_summary)
    print('Wrote parlay backtest outputs')


if __name__ == '__main__':
    main()
