from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .utils import write_df

RNG_SEED = 20260829
REPS = 50_000
DECIMAL_110 = 1.0 + 100.0 / 110.0
PRIMARY = 'primary_general_3leg_edge3p5_total56'


def parlay_units(results: list[str]) -> float:
    if any(r == 'loss' for r in results):
        return -1.0
    wins = sum(r == 'win' for r in results)
    if wins == 0:
        return 0.0
    return float(DECIMAL_110 ** wins - 1.0)


def sign_test_two_sided(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return np.nan
    k = min(positive, negative)
    one_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * one_tail))


def main() -> None:
    legs = pd.read_csv('outputs/three_leg_selected_legs.csv')
    legs = legs[legs['config'].eq(PRIMARY)].copy()
    if legs.empty:
        raise RuntimeError('Primary selected-leg output is required.')

    rows = []
    for card_id, group in legs.groupby('card_id'):
        group = group.sort_values('leg_rank')
        if len(group) != 3:
            raise RuntimeError(f'Primary card {card_id} does not contain exactly three legs.')
        results = group['result'].astype(str).tolist()
        rows.append({
            'card_id': card_id,
            'season': int(group['season'].iloc[0]),
            'week': int(group['week'].iloc[0]),
            'top2_units_110': parlay_units(results[:2]),
            'top3_units_110': parlay_units(results),
            'third_leg_result': results[2],
            'first_two_result': (
                'loss' if 'loss' in results[:2]
                else 'push' if all(r == 'push' for r in results[:2])
                else 'win'
            ),
        })
    matched = pd.DataFrame(rows).sort_values(['season', 'week']).reset_index(drop=True)
    matched['delta_top3_minus_top2'] = matched['top3_units_110'] - matched['top2_units_110']

    season = matched.groupby('season').agg(
        cards=('card_id', 'size'),
        top2_net_units=('top2_units_110', 'sum'),
        top3_net_units=('top3_units_110', 'sum'),
        delta_net_units=('delta_top3_minus_top2', 'sum'),
    ).reset_index()
    season['top2_roi'] = season['top2_net_units'] / season['cards']
    season['top3_roi'] = season['top3_net_units'] / season['cards']
    season['delta_roi'] = season['top3_roi'] - season['top2_roi']

    groups = {
        int(s): g[['top2_units_110', 'top3_units_110']].to_numpy(dtype=float)
        for s, g in matched.groupby('season')
    }
    seasons = np.array(sorted(groups))
    rng = np.random.default_rng(RNG_SEED)
    diffs = np.empty(REPS, dtype=float)
    for i in range(REPS):
        sampled = rng.choice(seasons, size=len(seasons), replace=True)
        values = np.concatenate([groups[int(s)] for s in sampled])
        diffs[i] = float(np.mean(values[:, 1] - values[:, 0]))

    positive_season_deltas = int(season['delta_roi'].gt(0).sum())
    negative_season_deltas = int(season['delta_roi'].lt(0).sum())
    ties = int(season['delta_roi'].eq(0).sum())
    overview = pd.DataFrame([{
        'matched_weeks': int(len(matched)),
        'top2_net_units_110': float(matched['top2_units_110'].sum()),
        'top2_roi_110': float(matched['top2_units_110'].mean()),
        'top3_net_units_110': float(matched['top3_units_110'].sum()),
        'top3_roi_110': float(matched['top3_units_110'].mean()),
        'delta_top3_minus_top2_roi': float(matched['delta_top3_minus_top2'].mean()),
        'season_block_bootstrap_delta_low': float(np.quantile(diffs, 0.025)),
        'season_block_bootstrap_delta_median': float(np.quantile(diffs, 0.5)),
        'season_block_bootstrap_delta_high': float(np.quantile(diffs, 0.975)),
        'bootstrap_fraction_top3_better': float(np.mean(diffs > 0)),
        'seasons_top3_better': positive_season_deltas,
        'seasons_top3_worse': negative_season_deltas,
        'seasons_tied': ties,
        'two_sided_sign_test_p': sign_test_two_sided(positive_season_deltas, negative_season_deltas),
    }])

    top2_wins = matched[matched['first_two_result'].eq('win')].copy()
    graded_third = top2_wins[top2_wins['third_leg_result'].isin(['win', 'loss'])]
    conditional = pd.DataFrame([{
        'weeks_first_two_win': int(len(top2_wins)),
        'third_leg_graded_after_first_two_win': int(len(graded_third)),
        'third_leg_wins_after_first_two_win': int(graded_third['third_leg_result'].eq('win').sum()),
        'third_leg_losses_after_first_two_win': int(graded_third['third_leg_result'].eq('loss').sum()),
        'conditional_third_leg_hit_rate': float(graded_third['third_leg_result'].eq('win').mean()) if len(graded_third) else np.nan,
    }])

    write_df(matched, 'outputs/three_leg_matched_week_cards.csv')
    write_df(season, 'outputs/three_leg_matched_two_vs_three_by_season.csv')
    write_df(overview, 'outputs/three_leg_matched_two_vs_three_summary.csv')
    write_df(conditional, 'outputs/three_leg_third_leg_conditional.csv')

    print('=== POST-HOC MATCHED-WEEK TWO VS THREE DIAGNOSTIC ===')
    print(overview.to_string(index=False))
    print('=== MATCHED-WEEK BY SEASON ===')
    print(season.to_string(index=False))
    print('=== CONDITIONAL THIRD LEG ===')
    print(conditional.to_string(index=False))


if __name__ == '__main__':
    main()
