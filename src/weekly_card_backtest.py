from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from .parlay_backtest import LEG_DECIMAL_ODDS, parlay_units
from .utils import ensure_dir, read_df, write_df

MAX_CANDIDATE_COMBOS_PER_WEEK = 15000
STRAIGHT_WIN_UNITS = 100 / 110


def leg_units(result: str) -> float:
    if result == 'win':
        return STRAIGHT_WIN_UNITS
    if result == 'loss':
        return -1.0
    return 0.0


def normalize_plays(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['threshold'] = pd.to_numeric(out['threshold'], errors='coerce')
    out['season'] = pd.to_numeric(out['season'], errors='coerce').astype('Int64')
    out['week'] = pd.to_numeric(out.get('week'), errors='coerce').astype('Int64') if 'week' in out.columns else pd.Series([pd.NA] * len(out), dtype='Int64')
    out['game_id'] = out['game_id'].astype(str) if 'game_id' in out.columns else out.index.astype(str)
    out['abs_pred_edge'] = pd.to_numeric(out['pred_market_residual'], errors='coerce').abs()
    out['market_residual'] = pd.to_numeric(out['market_residual'], errors='coerce')
    return out


def strategy_pool(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    high_total = ['56-63', '63+']
    base = df[(df['model'] == 'hist_gradient_boosting') & (df['side'] == 'under')].copy()
    if strategy == 'hgb_under_3p5_high_total':
        return base[(base['threshold'] == 3.5) & (base['total_bin'].isin(high_total))]
    if strategy == 'hgb_under_3p5_consensus':
        return base[(base['threshold'] == 3.5) & (base['line_provider'] == 'consensus')]
    if strategy == 'hgb_under_3p5_all':
        return base[base['threshold'] == 3.5]
    if strategy == 'hgb_under_5p0_high_total':
        return base[(base['threshold'] == 5.0) & (base['total_bin'].isin(high_total))]
    if strategy == 'hgb_under_5p0_consensus':
        return base[(base['threshold'] == 5.0) & (base['line_provider'] == 'consensus')]
    raise ValueError(f'Unknown strategy: {strategy}')


def combo_candidates(group: pd.DataFrame, legs: int) -> list[dict]:
    group = group.drop_duplicates('game_id').sort_values('abs_pred_edge', ascending=False).copy()
    if len(group) < legs:
        return []
    if comb(len(group), legs) > MAX_CANDIDATE_COMBOS_PER_WEEK:
        group = group.head(legs + 20).copy()
    records = group.to_dict('records')
    rows = []
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
            'game_ids_set': set(str(r['game_id']) for r in combo),
            'game_ids': '|'.join(str(r['game_id']) for r in combo),
            'matchups': ' | '.join(f"{r.get('away_team','')} at {r.get('home_team','')}" for r in combo),
            'results': '|'.join(results),
            'result': result,
            'units': unit_result,
            'avg_abs_pred_edge': float(np.mean([r['abs_pred_edge'] for r in combo])),
            'min_abs_pred_edge': float(np.min([r['abs_pred_edge'] for r in combo])),
            'avg_market_residual': float(np.mean([r['market_residual'] for r in combo])),
            'leg_records': combo,
        })
    rows.sort(key=lambda r: (r['min_abs_pred_edge'], r['avg_abs_pred_edge']), reverse=True)
    return rows


def select_weekly_card(candidates: list[dict], max_combos: int, no_overlap: bool) -> list[dict]:
    selected = []
    used_games: set[str] = set()
    for cand in candidates:
        if no_overlap and cand['game_ids_set'] & used_games:
            continue
        selected.append(cand)
        used_games |= cand['game_ids_set']
        if len(selected) >= max_combos:
            break
    return selected


def build_cards(plays: pd.DataFrame, strategy: str, legs: int, max_combos_per_week: int, no_overlap: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = strategy_pool(plays, strategy)
    card_rows = []
    leg_rows = []
    card_name = f'{strategy}_{legs}leg_top{max_combos_per_week}' + ('_nonoverlap' if no_overlap else '')
    for (season, week), group in pool.groupby(['season', 'week'], dropna=False):
        candidates = combo_candidates(group, legs)
        selected = select_weekly_card(candidates, max_combos_per_week, no_overlap)
        for rank, cand in enumerate(selected, start=1):
            card_rows.append({
                'card_strategy': card_name,
                'base_strategy': strategy,
                'legs': legs,
                'max_combos_per_week': max_combos_per_week,
                'no_overlap': no_overlap,
                'season': int(season) if pd.notna(season) else None,
                'week': int(week) if pd.notna(week) else None,
                'weekly_rank': rank,
                'game_ids': cand['game_ids'],
                'matchups': cand['matchups'],
                'results': cand['results'],
                'result': cand['result'],
                'units': cand['units'],
                'avg_abs_pred_edge': cand['avg_abs_pred_edge'],
                'min_abs_pred_edge': cand['min_abs_pred_edge'],
                'avg_market_residual': cand['avg_market_residual'],
            })
            for leg in cand['leg_records']:
                leg_rows.append({
                    'card_strategy': card_name,
                    'base_strategy': strategy,
                    'legs': legs,
                    'season': int(season) if pd.notna(season) else None,
                    'week': int(week) if pd.notna(week) else None,
                    'weekly_rank': rank,
                    'game_id': str(leg['game_id']),
                    'matchup': f"{leg.get('away_team','')} at {leg.get('home_team','')}",
                    'side': leg['side'],
                    'result': leg['result'],
                    'straight_units': leg_units(leg['result']),
                    'abs_pred_edge': float(leg['abs_pred_edge']),
                    'market_residual': float(leg['market_residual']),
                    'total_bin': leg.get('total_bin'),
                    'line_provider': leg.get('line_provider'),
                })
    return pd.DataFrame(card_rows), pd.DataFrame(leg_rows)


def max_drawdown(units: pd.Series) -> float:
    if units.empty:
        return 0.0
    cumulative = units.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    return float(drawdown.min())


def summarize_cards(cards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if cards.empty:
        return pd.DataFrame()
    sort_cols = ['season', 'week', 'weekly_rank']
    for name, g in cards.groupby('card_strategy', dropna=False):
        ordered = g.sort_values(sort_cols)
        graded = ordered[ordered['result'] != 'push']
        wins = int((graded['result'] == 'win').sum())
        losses = int((graded['result'] == 'loss').sum())
        pushes = int((ordered['result'] == 'push').sum())
        n = wins + losses
        rows.append({
            'card_strategy': name,
            'legs': int(ordered['legs'].iloc[0]),
            'cards': int(len(ordered)),
            'graded': int(n),
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'hit_rate': wins / n if n else np.nan,
            'breakeven_hit_rate': 1 / (LEG_DECIMAL_ODDS ** int(ordered['legs'].iloc[0])),
            'net_units': float(ordered['units'].sum()),
            'roi_per_card': float(ordered['units'].sum()) / n if n else np.nan,
            'avg_cards_per_week': float(ordered.groupby(['season', 'week']).size().mean()),
            'max_drawdown_units': max_drawdown(ordered['units']),
            'avg_min_abs_pred_edge': float(ordered['min_abs_pred_edge'].mean()),
            'avg_market_residual': float(ordered['avg_market_residual'].mean()),
        })
    return pd.DataFrame(rows)


def summarize_cards_by_season(cards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if cards.empty:
        return pd.DataFrame()
    for (name, season), g in cards.groupby(['card_strategy', 'season'], dropna=False):
        graded = g[g['result'] != 'push']
        wins = int((graded['result'] == 'win').sum())
        losses = int((graded['result'] == 'loss').sum())
        n = wins + losses
        rows.append({
            'card_strategy': name,
            'season': season,
            'cards': int(len(g)),
            'graded': int(n),
            'wins': wins,
            'losses': losses,
            'hit_rate': wins / n if n else np.nan,
            'net_units': float(g['units'].sum()),
            'roi_per_card': float(g['units'].sum()) / n if n else np.nan,
            'max_drawdown_units': max_drawdown(g.sort_values(['week', 'weekly_rank'])['units']),
        })
    return pd.DataFrame(rows)


def summarize_straight_legs(legs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if legs_df.empty:
        return pd.DataFrame()
    unique = legs_df.drop_duplicates(['card_strategy', 'season', 'week', 'game_id']).copy()
    for name, g in unique.groupby('card_strategy', dropna=False):
        graded = g[g['result'] != 'push']
        wins = int((graded['result'] == 'win').sum())
        losses = int((graded['result'] == 'loss').sum())
        n = wins + losses
        rows.append({
            'card_strategy': name,
            'unique_straight_legs': int(len(g)),
            'graded': int(n),
            'wins': wins,
            'losses': losses,
            'hit_rate': wins / n if n else np.nan,
            'net_units_flat_1u_each': float(g['straight_units'].sum()),
            'roi_per_leg': float(g['straight_units'].sum()) / n if n else np.nan,
            'avg_abs_pred_edge': float(g['abs_pred_edge'].mean()),
            'avg_market_residual': float(g['market_residual'].mean()),
        })
    return pd.DataFrame(rows)


def write_methodology(summary: pd.DataFrame, by_season: pd.DataFrame, straight: pd.DataFrame) -> None:
    out = ensure_dir('outputs') / 'weekly_card_methodology_summary.md'
    lines = [
        '# Weekly Card Backtest Methodology Summary',
        '',
        'This test is a more realistic bridge between all-combination research and a weekly live card.',
        '',
        'Instead of grading every possible same-week combination, it selects only the top-ranked combinations per week by model edge. When multiple weekly combinations are tested, the non-overlap setting prevents the same game from being reused within that weekly card.',
        '',
        '## Weekly card summary',
        '',
        summary.sort_values(['roi_per_card', 'graded'], ascending=[False, False]).to_markdown(index=False) if not summary.empty else '_No rows._',
        '',
        '## Straight-leg equivalent summary',
        '',
        straight.sort_values(['roi_per_leg', 'graded'], ascending=[False, False]).to_markdown(index=False) if not straight.empty else '_No rows._',
        '',
        '## Season-by-season summary',
        '',
        by_season.sort_values(['card_strategy', 'season']).to_markdown(index=False) if not by_season.empty else '_No rows._',
        '',
        '## Interpretation notes',
        '',
        '- Prefer strategies that remain positive season-by-season and do not depend on one outlier year.',
        '- Compare the weekly combination result against the straight-leg equivalent.',
        '- A strong historical card still needs live tracking with current lines and forecast weather.',
        '- This is the most relevant historical test for what an actual weekly process could look like.',
    ]
    out.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    plays = normalize_plays(read_df('outputs/model_edge_plays.csv'))
    configs = [
        ('hgb_under_3p5_high_total', 2, 1, True),
        ('hgb_under_3p5_high_total', 2, 2, True),
        ('hgb_under_3p5_high_total', 3, 1, True),
        ('hgb_under_3p5_consensus', 2, 1, True),
        ('hgb_under_3p5_all', 2, 1, True),
        ('hgb_under_5p0_high_total', 2, 1, True),
        ('hgb_under_5p0_consensus', 2, 1, True),
    ]
    card_parts = []
    leg_parts = []
    for strategy, legs, max_cards, no_overlap in configs:
        cards, legs_df = build_cards(plays, strategy, legs, max_cards, no_overlap)
        if not cards.empty:
            card_parts.append(cards)
        if not legs_df.empty:
            leg_parts.append(legs_df)
    cards_all = pd.concat(card_parts, ignore_index=True) if card_parts else pd.DataFrame()
    legs_all = pd.concat(leg_parts, ignore_index=True) if leg_parts else pd.DataFrame()
    summary = summarize_cards(cards_all)
    by_season = summarize_cards_by_season(cards_all)
    straight = summarize_straight_legs(legs_all)
    write_df(cards_all, 'outputs/weekly_card_backtest_cards.csv')
    write_df(legs_all, 'outputs/weekly_card_backtest_legs.csv')
    write_df(summary, 'outputs/weekly_card_backtest_summary.csv')
    write_df(by_season, 'outputs/weekly_card_backtest_by_season.csv')
    write_df(straight, 'outputs/weekly_card_straight_equivalent_summary.csv')
    write_methodology(summary, by_season, straight)
    print('Wrote weekly card backtest outputs')


if __name__ == '__main__':
    main()
