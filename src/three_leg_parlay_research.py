from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .deep_research import prep, wilson
from .model_bakeoff import feature_lists, prep_features, reg_models
from .utils import ensure_dir, read_df, write_df

# This module is deliberately research-only. It does not alter the live card.
RNG_SEED = 20260829
BOOTSTRAP_REPS = 20_000
RANDOM_CARD_REPS = 20_000

DYNAMIC_WEATHER_NUMS = {
    'wind_mph', 'temperature_f', 'humidity', 'precipitation',
    'snowfall', 'dewpoint_f', 'pressure',
}
DYNAMIC_WEATHER_CATS = {'wind_bin', 'temp_bin'}

MODEL_VARIANTS = {
    'current': (set(), set()),
    'no_dynamic_weather': (DYNAMIC_WEATHER_NUMS, DYNAMIC_WEATHER_CATS),
    'no_line_provider': (set(), {'line_provider'}),
    'no_weather_or_provider': (DYNAMIC_WEATHER_NUMS, DYNAMIC_WEATHER_CATS | {'line_provider'}),
}


@dataclass(frozen=True)
class CardConfig:
    name: str
    role: str
    variant: str
    scope: str
    legs: int
    min_edge: float
    min_total: float


# The primary hypothesis is fixed to the current production leg rule and a realistic
# one-card-per-week implementation. Sensitivities are intentionally few and predeclared.
CARD_CONFIGS = [
    CardConfig(
        'primary_general_3leg_edge3p5_total56',
        'PRIMARY', 'current', 'general_live_scope', 3, 3.5, 56.0,
    ),
    CardConfig(
        'comparator_general_2leg_edge3p5_total56',
        'COMPARATOR', 'current', 'general_live_scope', 2, 3.5, 56.0,
    ),
    CardConfig(
        'sensitivity_general_3leg_edge5_total56',
        'SENSITIVITY', 'current', 'general_live_scope', 3, 5.0, 56.0,
    ),
    CardConfig(
        'sensitivity_fbs_3leg_edge3p5_total56',
        'SENSITIVITY', 'current', 'fbs_vs_fbs', 3, 3.5, 56.0,
    ),
    CardConfig(
        'ablation_no_weather_3leg_edge3p5_total56',
        'ABLATION', 'no_dynamic_weather', 'general_live_scope', 3, 3.5, 56.0,
    ),
    CardConfig(
        'ablation_no_provider_3leg_edge3p5_total56',
        'ABLATION', 'no_line_provider', 'general_live_scope', 3, 3.5, 56.0,
    ),
    CardConfig(
        'ablation_no_weather_provider_3leg_edge3p5_total56',
        'ABLATION', 'no_weather_or_provider', 'general_live_scope', 3, 3.5, 56.0,
    ),
    CardConfig(
        'diagnostic_mixed_3leg_edge3p5_total56',
        'LEGACY_DIAGNOSTIC', 'current', 'mixed_all', 3, 3.5, 56.0,
    ),
]

PRICE_SCENARIOS = (-110, -115, -120)


def decimal_odds(american_price: int) -> float:
    if american_price >= 0:
        return 1.0 + american_price / 100.0
    return 1.0 + 100.0 / abs(american_price)


def parlay_units(results: Iterable[str], american_price: int = -110) -> float:
    results = list(results)
    if any(r == 'loss' for r in results):
        return -1.0
    wins = sum(r == 'win' for r in results)
    if wins == 0:
        return 0.0
    return float(decimal_odds(american_price) ** wins - 1.0)


def settle_under(actual: float, total: float) -> str:
    if actual < total:
        return 'win'
    if actual > total:
        return 'loss'
    return 'push'


def exact_binomial_upper_tail(wins: int, n: int, p0: float) -> float:
    if n <= 0:
        return np.nan
    terms = [
        math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
        for k in range(wins, n + 1)
    ]
    return float(min(1.0, math.fsum(terms)))


def max_drawdown(units: Iterable[float]) -> float:
    arr = np.asarray(list(units), dtype=float)
    if arr.size == 0:
        return 0.0
    equity = np.concatenate(([0.0], np.cumsum(arr)))
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity - peaks))


def longest_losing_streak(results: Iterable[str]) -> int:
    worst = current = 0
    for result in (r for r in results if r != 'push'):
        if result == 'loss':
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return int(worst)


def scope_frame(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    required = {'home_classification', 'away_classification'}
    if not required <= set(df.columns):
        raise RuntimeError('Classification columns are required for production-aligned parlay research.')
    home = df['home_classification'].astype(str).str.lower()
    away = df['away_classification'].astype(str).str.lower()
    if scope == 'general_live_scope':
        return df[~(home.eq('fcs') & away.eq('fcs'))].copy()
    if scope == 'fbs_vs_fbs':
        return df[home.eq('fbs') & away.eq('fbs')].copy()
    if scope == 'mixed_all':
        return df.copy()
    raise ValueError(f'Unknown scope: {scope}')


def walk_forward_hgb(raw: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prep(raw)
    nums, cats = feature_lists(df)
    drop_nums, drop_cats = MODEL_VARIANTS[variant]
    nums = [c for c in nums if c not in drop_nums]
    cats = [c for c in cats if c not in drop_cats]
    df = prep_features(df, cats)

    pred_parts: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for season in sorted(df['season'].dropna().astype(int).unique()):
        train = df[df['season'] < season].copy()
        test = df[df['season'] == season].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        model = reg_models(nums, cats)['hist_gradient_boosting']
        model.fit(train[nums + cats], train['market_residual'])
        pred = model.predict(test[nums + cats])
        test = test.assign(pred_market_residual=pred)
        pred_parts.append(test)
        diagnostics.append({
            'variant': variant,
            'test_season': int(season),
            'train_games': int(len(train)),
            'test_games': int(len(test)),
            'numeric_features': int(len(nums)),
            'categorical_features': int(len(cats)),
            'model_mae': float(mean_absolute_error(test['market_residual'], pred)),
            'zero_residual_baseline_mae': float(
                mean_absolute_error(test['market_residual'], np.zeros(len(test)))
            ),
        })
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return pred_df, pd.DataFrame(diagnostics)


def eligible_pool(pred: pd.DataFrame, cfg: CardConfig) -> pd.DataFrame:
    scoped = scope_frame(pred, cfg.scope)
    pool = scoped[
        pd.to_numeric(scoped['pred_market_residual'], errors='coerce').le(-cfg.min_edge)
        & pd.to_numeric(scoped['closing_total'], errors='coerce').ge(cfg.min_total)
    ].copy()
    pool['season'] = pd.to_numeric(pool['season'], errors='coerce').astype('Int64')
    pool['week'] = pd.to_numeric(pool['week'], errors='coerce').astype('Int64')
    pool = pool.dropna(subset=['season', 'week', 'game_id', 'closing_total', 'actual_total_points'])
    pool['abs_pred_edge'] = pd.to_numeric(pool['pred_market_residual'], errors='coerce').abs()
    pool['result'] = [
        settle_under(float(a), float(t))
        for a, t in zip(pool['actual_total_points'], pool['closing_total'])
    ]
    return pool


def build_weekly_cards(pool: pd.DataFrame, cfg: CardConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    card_rows: list[dict] = []
    leg_rows: list[dict] = []
    for (season, week), group in pool.groupby(['season', 'week'], dropna=False):
        group = group.drop_duplicates('game_id').copy()
        group['game_id_sort'] = group['game_id'].astype(str)
        group = group.sort_values(
            ['abs_pred_edge', 'game_id_sort'], ascending=[False, True], kind='mergesort'
        )
        if len(group) < cfg.legs:
            continue
        selected = group.head(cfg.legs).copy()
        card_id = f'{cfg.name}:{int(season)}:{int(week)}'
        results = selected['result'].tolist()
        push_count = int(sum(r == 'push' for r in results))
        loss_count = int(sum(r == 'loss' for r in results))
        if loss_count:
            result = 'loss'
        elif push_count == cfg.legs:
            result = 'push'
        else:
            result = 'win'

        row = {
            'card_id': card_id,
            'config': cfg.name,
            'role': cfg.role,
            'variant': cfg.variant,
            'scope': cfg.scope,
            'legs': cfg.legs,
            'min_edge_rule': cfg.min_edge,
            'min_total_rule': cfg.min_total,
            'season': int(season),
            'week': int(week),
            'candidate_count': int(len(group)),
            'game_ids': '|'.join(selected['game_id'].astype(str)),
            'matchups': ' | '.join(
                f"{r.get('away_team', '')} at {r.get('home_team', '')}"
                for r in selected.to_dict('records')
            ),
            'leg_results': '|'.join(results),
            'result': result,
            'push_count': push_count,
            'avg_abs_pred_edge': float(selected['abs_pred_edge'].mean()),
            'min_abs_pred_edge': float(selected['abs_pred_edge'].min()),
            'avg_closing_total': float(pd.to_numeric(selected['closing_total']).mean()),
            'avg_market_residual': float(pd.to_numeric(selected['market_residual']).mean()),
        }
        for price in PRICE_SCENARIOS:
            row[f'units_{abs(price)}'] = parlay_units(results, price)
        card_rows.append(row)

        for rank, (_, leg) in enumerate(selected.iterrows(), start=1):
            leg_result = str(leg['result'])
            leg_rows.append({
                'card_id': card_id,
                'config': cfg.name,
                'role': cfg.role,
                'variant': cfg.variant,
                'scope': cfg.scope,
                'legs': cfg.legs,
                'season': int(season),
                'week': int(week),
                'leg_rank': rank,
                'game_id': str(leg['game_id']),
                'matchup': f"{leg.get('away_team', '')} at {leg.get('home_team', '')}",
                'closing_total': float(leg['closing_total']),
                'pred_market_residual': float(leg['pred_market_residual']),
                'abs_pred_edge': float(leg['abs_pred_edge']),
                'market_residual': float(leg['market_residual']),
                'result': leg_result,
                'straight_units_110': (
                    100 / 110 if leg_result == 'win' else -1.0 if leg_result == 'loss' else 0.0
                ),
            })
    return pd.DataFrame(card_rows), pd.DataFrame(leg_rows)


def season_block_bootstrap(cards: pd.DataFrame, unit_col: str, reps: int = BOOTSTRAP_REPS) -> tuple[float, float]:
    if cards.empty or cards['season'].nunique() < 2:
        return np.nan, np.nan
    groups = {
        int(season): group[unit_col].to_numpy(dtype=float)
        for season, group in cards.groupby('season')
    }
    seasons = np.array(sorted(groups))
    rng = np.random.default_rng(RNG_SEED)
    rois = np.empty(reps, dtype=float)
    for i in range(reps):
        sampled = rng.choice(seasons, size=len(seasons), replace=True)
        units = np.concatenate([groups[int(s)] for s in sampled])
        rois[i] = units.sum() / len(units) if len(units) else np.nan
    return float(np.nanquantile(rois, 0.025)), float(np.nanquantile(rois, 0.975))


def summarize_cards(cards: pd.DataFrame, cfg: CardConfig) -> dict:
    if cards.empty:
        return {
            'config': cfg.name, 'role': cfg.role, 'variant': cfg.variant, 'scope': cfg.scope,
            'legs': cfg.legs, 'min_edge': cfg.min_edge, 'min_total': cfg.min_total, 'cards': 0,
        }
    ordered = cards.sort_values(['season', 'week']).copy()
    full = ordered[ordered['push_count'].eq(0)].copy()
    full_wins = int(full['result'].eq('win').sum())
    full_losses = int(full['result'].eq('loss').sum())
    full_n = full_wins + full_losses
    full_hit = full_wins / full_n if full_n else np.nan
    p0 = 1.0 / (decimal_odds(-110) ** cfg.legs)
    low, high = wilson(full_wins, full_n)
    p_nominal = exact_binomial_upper_tail(full_wins, full_n, p0) if full_n else np.nan
    net110 = float(ordered['units_110'].sum())
    boot_low, boot_high = season_block_bootstrap(ordered, 'units_110')

    season_roi = ordered.groupby('season')['units_110'].agg(['sum', 'count'])
    season_roi['roi'] = season_roi['sum'] / season_roi['count']
    positive_seasons = int(season_roi['roi'].gt(0).sum())

    row = {
        'config': cfg.name,
        'role': cfg.role,
        'variant': cfg.variant,
        'scope': cfg.scope,
        'legs': cfg.legs,
        'min_edge': cfg.min_edge,
        'min_total': cfg.min_total,
        'cards': int(len(ordered)),
        'seasons_with_cards': int(ordered['season'].nunique()),
        'positive_seasons': positive_seasons,
        'full_leg_cards': int(full_n),
        'full_leg_wins': full_wins,
        'full_leg_losses': full_losses,
        'push_affected_cards': int(ordered['push_count'].gt(0).sum()),
        'full_card_hit_rate': full_hit,
        'theoretical_breakeven_hit_rate_110': p0,
        'hit_rate_wilson_low': low,
        'hit_rate_wilson_high': high,
        'nominal_one_sided_binom_p_110': p_nominal,
        'net_units_110': net110,
        'roi_per_staked_card_110': net110 / len(ordered),
        'season_block_bootstrap_roi_low_110': boot_low,
        'season_block_bootstrap_roi_high_110': boot_high,
        'max_drawdown_units_110': max_drawdown(ordered['units_110']),
        'longest_losing_streak': longest_losing_streak(ordered['result']),
        'avg_candidates_when_card_available': float(ordered['candidate_count'].mean()),
        'avg_abs_pred_edge': float(ordered['avg_abs_pred_edge'].mean()),
        'avg_min_abs_pred_edge': float(ordered['min_abs_pred_edge'].mean()),
        'avg_closing_total': float(ordered['avg_closing_total'].mean()),
        'avg_market_residual_per_leg': float(ordered['avg_market_residual'].mean()),
    }
    for price in (-115, -120):
        col = f'units_{abs(price)}'
        net = float(ordered[col].sum())
        row[f'net_units_{abs(price)}'] = net
        row[f'roi_per_staked_card_{abs(price)}'] = net / len(ordered)
        row[f'theoretical_breakeven_hit_rate_{abs(price)}'] = 1.0 / (
            decimal_odds(price) ** cfg.legs
        )
    return row


def summarize_by_season(cards: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (config, season), group in cards.groupby(['config', 'season'], dropna=False):
        ordered = group.sort_values('week')
        full = ordered[ordered['push_count'].eq(0)]
        wins = int(full['result'].eq('win').sum())
        losses = int(full['result'].eq('loss').sum())
        n = wins + losses
        rows.append({
            'config': config,
            'season': int(season),
            'cards': int(len(ordered)),
            'full_leg_cards': n,
            'wins': wins,
            'losses': losses,
            'hit_rate': wins / n if n else np.nan,
            'net_units_110': float(ordered['units_110'].sum()),
            'roi_per_staked_card_110': float(ordered['units_110'].sum()) / len(ordered),
            'max_drawdown_units_110': max_drawdown(ordered['units_110']),
            'longest_losing_streak': longest_losing_streak(ordered['result']),
        })
    return pd.DataFrame(rows)


def summarize_by_era(cards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = cards.copy()
    frame['era'] = np.where(frame['season'].le(2020), '2016-2020', '2021-2025')
    for (config, era), group in frame.groupby(['config', 'era']):
        rows.append({
            'config': config,
            'era': era,
            'cards': int(len(group)),
            'wins': int(group['result'].eq('win').sum()),
            'losses': int(group['result'].eq('loss').sum()),
            'push_affected_cards': int(group['push_count'].gt(0).sum()),
            'net_units_110': float(group['units_110'].sum()),
            'roi_per_staked_card_110': float(group['units_110'].sum()) / len(group),
            'max_drawdown_units_110': max_drawdown(group.sort_values(['season', 'week'])['units_110']),
        })
    return pd.DataFrame(rows)


def summarize_straight_legs(legs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, group in legs.groupby('config'):
        graded = group[group['result'].ne('push')]
        wins = int(graded['result'].eq('win').sum())
        losses = int(graded['result'].eq('loss').sum())
        n = wins + losses
        low, high = wilson(wins, n)
        net = float(group['straight_units_110'].sum())
        rows.append({
            'config': config,
            'legs_staked': int(len(group)),
            'graded_legs': n,
            'wins': wins,
            'losses': losses,
            'pushes': int(group['result'].eq('push').sum()),
            'hit_rate': wins / n if n else np.nan,
            'straight_breakeven_hit_rate_110': 110 / 210,
            'hit_rate_wilson_low': low,
            'hit_rate_wilson_high': high,
            'nominal_one_sided_binom_p_110': exact_binomial_upper_tail(wins, n, 110 / 210) if n else np.nan,
            'net_units_1u_each': net,
            'roi_per_unit_staked': net / len(group) if len(group) else np.nan,
            'avg_abs_pred_edge': float(group['abs_pred_edge'].mean()),
        })
    return pd.DataFrame(rows)


def rank_dependence(primary_cards: pd.DataFrame, primary_legs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_ids = set(primary_cards.loc[primary_cards['push_count'].eq(0), 'card_id'])
    legs = primary_legs[primary_legs['card_id'].isin(full_ids)].copy()
    legs['win_binary'] = legs['result'].eq('win').astype(float)

    rank_rows = []
    for rank, group in legs.groupby('leg_rank'):
        wins = int(group['result'].eq('win').sum())
        n = int(len(group))
        rank_rows.append({
            'leg_rank': int(rank),
            'full_card_legs': n,
            'wins': wins,
            'hit_rate': wins / n if n else np.nan,
            'avg_abs_pred_edge': float(group['abs_pred_edge'].mean()),
        })
    rank_df = pd.DataFrame(rank_rows).sort_values('leg_rank') if rank_rows else pd.DataFrame()

    pivot = legs.pivot(index='card_id', columns='leg_rank', values='win_binary')
    corr_rows = []
    for a, b in [(1, 2), (1, 3), (2, 3)]:
        if a not in pivot.columns or b not in pivot.columns:
            continue
        pair = pivot[[a, b]].dropna()
        if len(pair) < 3 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
            corr = np.nan
        else:
            corr = float(pair[a].corr(pair[b]))
        corr_rows.append({'rank_a': a, 'rank_b': b, 'cards': int(len(pair)), 'win_phi_correlation': corr})
    corr_df = pd.DataFrame(corr_rows)
    return rank_df, corr_df


def independence_summary(primary_cards: pd.DataFrame, rank_df: pd.DataFrame) -> pd.DataFrame:
    full = primary_cards[primary_cards['push_count'].eq(0)]
    observed = float(full['result'].eq('win').mean()) if len(full) else np.nan
    if not rank_df.empty and len(rank_df) == 3:
        expected_rank_product = float(rank_df['hit_rate'].prod())
        weighted_leg_hit = float(
            rank_df['wins'].sum() / rank_df['full_card_legs'].sum()
        )
        expected_pooled = weighted_leg_hit ** 3
    else:
        expected_rank_product = np.nan
        weighted_leg_hit = np.nan
        expected_pooled = np.nan
    return pd.DataFrame([{
        'full_three_leg_cards': int(len(full)),
        'observed_three_leg_hit_rate': observed,
        'pooled_selected_leg_hit_rate': weighted_leg_hit,
        'independence_expected_hit_rank_specific': expected_rank_product,
        'independence_expected_hit_pooled_cubed': expected_pooled,
        'observed_minus_rank_independence': observed - expected_rank_product if pd.notna(observed) and pd.notna(expected_rank_product) else np.nan,
    }])


def leave_one_season_out(primary_cards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in sorted(primary_cards['season'].unique()):
        group = primary_cards[primary_cards['season'].ne(season)].sort_values(['season', 'week'])
        if group.empty:
            continue
        rows.append({
            'excluded_season': int(season),
            'cards': int(len(group)),
            'net_units_110': float(group['units_110'].sum()),
            'roi_per_staked_card_110': float(group['units_110'].sum()) / len(group),
            'max_drawdown_units_110': max_drawdown(group['units_110']),
        })
    return pd.DataFrame(rows)


def random_three_null(primary_pool: pd.DataFrame, observed_cards: pd.DataFrame) -> pd.DataFrame:
    weekly = []
    for (season, week), group in primary_pool.groupby(['season', 'week']):
        group = group.drop_duplicates('game_id')
        if len(group) < 3:
            continue
        weekly.append((int(season), int(week), group['result'].to_numpy(dtype=object)))
    if not weekly:
        return pd.DataFrame()

    observed_roi = float(observed_cards['units_110'].sum()) / len(observed_cards)
    rng = np.random.default_rng(RNG_SEED + 1)
    rois = np.empty(RANDOM_CARD_REPS, dtype=float)
    for i in range(RANDOM_CARD_REPS):
        net = 0.0
        for _, _, outcomes in weekly:
            chosen = rng.choice(len(outcomes), size=3, replace=False)
            net += parlay_units(outcomes[chosen], -110)
        rois[i] = net / len(weekly)

    return pd.DataFrame([{
        'simulations': RANDOM_CARD_REPS,
        'weeks': len(weekly),
        'observed_top3_roi_110': observed_roi,
        'random_three_mean_roi_110': float(np.mean(rois)),
        'random_three_median_roi_110': float(np.median(rois)),
        'random_three_roi_2p5': float(np.quantile(rois, 0.025)),
        'random_three_roi_97p5': float(np.quantile(rois, 0.975)),
        'fraction_random_at_least_observed': float(np.mean(rois >= observed_roi)),
        'observed_percentile_vs_random': float(np.mean(rois <= observed_roi)),
    }])


def coverage_summary(predictions: dict[str, pd.DataFrame], configs: list[CardConfig]) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        pred = predictions[cfg.variant]
        scope = scope_frame(pred, cfg.scope)
        all_weeks = scope.dropna(subset=['season', 'week']).groupby(['season', 'week']).ngroups
        pool = eligible_pool(pred, cfg)
        counts = pool.groupby(['season', 'week']).size() if not pool.empty else pd.Series(dtype=int)
        eligible = int((counts >= cfg.legs).sum()) if len(counts) else 0
        rows.append({
            'config': cfg.name,
            'scope_weeks_in_oof_sample': int(all_weeks),
            'weeks_with_any_candidate': int(len(counts)),
            'weeks_with_enough_legs_for_card': eligible,
            'card_availability_rate_all_scope_weeks': eligible / all_weeks if all_weeks else np.nan,
        })
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    by_season: pd.DataFrame,
    by_era: pd.DataFrame,
    straight: pd.DataFrame,
    rank_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    independence: pd.DataFrame,
    loo: pd.DataFrame,
    random_null: pd.DataFrame,
    coverage: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    primary = summary[summary['config'].eq('primary_general_3leg_edge3p5_total56')].iloc[0]
    comparator = summary[summary['config'].eq('comparator_general_2leg_edge3p5_total56')].iloc[0]
    lines = [
        '# Dedicated Three-Leg Parlay Research',
        '',
        '## Research posture',
        '',
        'This is a research-only audit. It does not modify the live weekly card. The primary three-leg hypothesis was fixed before this rerun: use the current non-FCS general HGB under screen (predicted under edge >=3.5 and total >=56), and only when at least three games are available in a season/week, take the three strongest model edges as exactly one card. No outcome information is used to choose the weekly legs.',
        '',
        'The model predictions are season walk-forward: each test season is predicted only by a model fit on earlier seasons. The final 3.5/56 leg rule itself was developed from earlier historical research, so this historical parlay test is not a pristine independent confirmation. Operational promotion still requires prospective 2026 shadow tracking.',
        '',
        '## Primary result',
        '',
        f"- Cards: {int(primary['cards'])}; full three-leg wins/losses: {int(primary['full_leg_wins'])}-{int(primary['full_leg_losses'])}; full-card hit rate: {primary['full_card_hit_rate']:.1%} versus theoretical -110 three-leg breakeven {primary['theoretical_breakeven_hit_rate_110']:.1%}.",
        f"- Net at -110-per-leg multiplication: {primary['net_units_110']:.2f} units; ROI per 1-unit card staked: {primary['roi_per_staked_card_110']:.1%}; season-block bootstrap ROI interval: {primary['season_block_bootstrap_roi_low_110']:.1%} to {primary['season_block_bootstrap_roi_high_110']:.1%}.",
        f"- Wilson 95% interval for full-card hit rate: {primary['hit_rate_wilson_low']:.1%} to {primary['hit_rate_wilson_high']:.1%}; nominal one-sided binomial p-value versus parlay breakeven: {primary['nominal_one_sided_binom_p_110']:.4f} (not multiple-testing adjusted).",
        f"- Maximum historical drawdown: {primary['max_drawdown_units_110']:.2f} units; longest losing streak: {int(primary['longest_losing_streak'])} cards.",
        '',
        '## Primary versus same-pool two-leg comparator',
        '',
        f"Three-leg ROI: {primary['roi_per_staked_card_110']:.1%}; two-leg ROI: {comparator['roi_per_staked_card_110']:.1%}. Three-leg max drawdown: {primary['max_drawdown_units_110']:.2f}u; two-leg max drawdown: {comparator['max_drawdown_units_110']:.2f}u. This comparison is descriptive, not proof that the higher-variance structure is superior.",
        '',
        '## All predeclared configurations',
        '',
        summary.to_markdown(index=False),
        '',
        '## Straight-leg equivalent',
        '',
        straight.to_markdown(index=False),
        '',
        '## Season stability',
        '',
        by_season.to_markdown(index=False),
        '',
        '## Early versus recent era',
        '',
        by_era.to_markdown(index=False),
        '',
        '## Leave-one-season-out sensitivity (primary only)',
        '',
        loo.to_markdown(index=False),
        '',
        '## Within-card leg dependence',
        '',
        rank_df.to_markdown(index=False),
        '',
        corr_df.to_markdown(index=False),
        '',
        independence.to_markdown(index=False),
        '',
        '## Does strongest-three ranking add value?',
        '',
        random_null.to_markdown(index=False),
        '',
        'The random-three simulation keeps the same qualifying games in each week and randomly chooses three. It therefore tests the incremental value of ranking by model edge, not whether the underlying 3.5/56 pool itself is valid.',
        '',
        '## Card availability',
        '',
        coverage.to_markdown(index=False),
        '',
        '## Walk-forward model diagnostics',
        '',
        diagnostics.to_markdown(index=False),
        '',
        '## Scientific guardrails',
        '',
        '- **No FCS legs are mixed into the primary card.** FCS has its own shorter-history model and has not independently validated a parlay strategy.',
        '- Historical prices are treated as -110 per leg and multiplied at true parlay odds. ROI is also stress-tested at -115 and -120 per leg.',
        '- Historical `closing_total` values are not timestamp-matched Friday/Saturday entry prices. A historical parlay edge can therefore be useful research evidence without proving executable live ROI.',
        '- Historical weather is not an archived Friday/Saturday NWS forecast information set. The weather ablation is included specifically to test dependence on that mismatch.',
        '- One card per week prevents the old all-combinations approach from inflating the apparent sample with overlapping parlays.',
        '- Season-block bootstrap, era splits, leave-one-season-out checks, drawdown, losing streaks, and within-card dependence are reported because parlay variance makes aggregate ROI alone misleading.',
        '- The nominal binomial p-value is descriptive only: the underlying leg rule was selected during prior research and the repository previously contained exploratory three-leg output.',
        '- **Operational standard:** even a strong historical result should enter 2026 only as a frozen shadow/paper track first. Do not replace or expand the validated two-leg live card from this historical analysis alone.',
    ]
    path = ensure_dir('outputs') / 'three_leg_research_summary.md'
    path.write_text('\n'.join(lines), encoding='utf-8')


def self_test() -> None:
    d = decimal_odds(-110)
    assert abs(parlay_units(['win', 'win', 'win'], -110) - (d ** 3 - 1)) < 1e-12
    assert abs(parlay_units(['win', 'win', 'push'], -110) - (d ** 2 - 1)) < 1e-12
    assert parlay_units(['win', 'loss', 'win'], -110) == -1.0
    assert parlay_units(['push', 'push', 'push'], -110) == 0.0
    assert max_drawdown([-1.0, -1.0, 5.0]) == -2.0
    assert longest_losing_streak(['loss', 'loss', 'push', 'loss', 'win']) == 3
    p = exact_binomial_upper_tail(1, 1, 0.2)
    assert abs(p - 0.2) < 1e-12

    sample = pd.DataFrame([
        {'season': 2020, 'week': 1, 'game_id': 'b', 'abs_pred_edge': 5.0, 'result': 'win', 'closing_total': 60, 'pred_market_residual': -5, 'market_residual': -3, 'actual_total_points': 57, 'home_team': 'H2', 'away_team': 'A2'},
        {'season': 2020, 'week': 1, 'game_id': 'a', 'abs_pred_edge': 7.0, 'result': 'win', 'closing_total': 60, 'pred_market_residual': -7, 'market_residual': -4, 'actual_total_points': 56, 'home_team': 'H1', 'away_team': 'A1'},
        {'season': 2020, 'week': 1, 'game_id': 'c', 'abs_pred_edge': 6.0, 'result': 'loss', 'closing_total': 60, 'pred_market_residual': -6, 'market_residual': 2, 'actual_total_points': 62, 'home_team': 'H3', 'away_team': 'A3'},
        {'season': 2020, 'week': 1, 'game_id': 'd', 'abs_pred_edge': 4.0, 'result': 'win', 'closing_total': 60, 'pred_market_residual': -4, 'market_residual': -1, 'actual_total_points': 59, 'home_team': 'H4', 'away_team': 'A4'},
    ])
    cfg = CardConfig('test', 'TEST', 'current', 'mixed_all', 3, 3.5, 56)
    cards, legs = build_weekly_cards(sample, cfg)
    assert len(cards) == 1 and len(legs) == 3
    assert cards.iloc[0]['game_ids'] == 'a|c|b'
    assert cards.iloc[0]['result'] == 'loss'
    print('three_leg_parlay_research self-test passed')


def main() -> None:
    raw = read_df('data/processed/modeling_dataset.csv')
    if raw.empty:
        raise RuntimeError('data/processed/modeling_dataset.csv is required.')
    required = {
        'season', 'week', 'game_id', 'closing_total', 'actual_total_points',
        'home_classification', 'away_classification', 'market_residual',
    }
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f'Modeling artifact is missing required columns: {sorted(missing)}')

    predictions: dict[str, pd.DataFrame] = {}
    diag_parts = []
    for variant in MODEL_VARIANTS:
        pred, diag = walk_forward_hgb(raw, variant)
        predictions[variant] = pred
        diag_parts.append(diag)

    card_parts = []
    leg_parts = []
    pools: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for cfg in CARD_CONFIGS:
        pool = eligible_pool(predictions[cfg.variant], cfg)
        pools[cfg.name] = pool
        cards, legs = build_weekly_cards(pool, cfg)
        if not cards.empty:
            card_parts.append(cards)
        if not legs.empty:
            leg_parts.append(legs)
        summary_rows.append(summarize_cards(cards, cfg))

    cards_all = pd.concat(card_parts, ignore_index=True) if card_parts else pd.DataFrame()
    legs_all = pd.concat(leg_parts, ignore_index=True) if leg_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    by_season = summarize_by_season(cards_all)
    by_era = summarize_by_era(cards_all)
    straight = summarize_straight_legs(legs_all)
    coverage = coverage_summary(predictions, CARD_CONFIGS)
    diagnostics = pd.concat(diag_parts, ignore_index=True)

    primary_name = 'primary_general_3leg_edge3p5_total56'
    primary_cards = cards_all[cards_all['config'].eq(primary_name)].copy()
    primary_legs = legs_all[legs_all['config'].eq(primary_name)].copy()
    rank_df, corr_df = rank_dependence(primary_cards, primary_legs)
    independence = independence_summary(primary_cards, rank_df)
    loo = leave_one_season_out(primary_cards)
    random_null = random_three_null(pools[primary_name], primary_cards)

    write_df(cards_all, 'outputs/three_leg_cards.csv')
    write_df(legs_all, 'outputs/three_leg_selected_legs.csv')
    write_df(summary, 'outputs/three_leg_config_summary.csv')
    write_df(by_season, 'outputs/three_leg_by_season.csv')
    write_df(by_era, 'outputs/three_leg_by_era.csv')
    write_df(straight, 'outputs/three_leg_straight_equivalent.csv')
    write_df(rank_df, 'outputs/three_leg_rank_performance.csv')
    write_df(corr_df, 'outputs/three_leg_leg_dependence.csv')
    write_df(independence, 'outputs/three_leg_independence_check.csv')
    write_df(loo, 'outputs/three_leg_leave_one_season_out.csv')
    write_df(random_null, 'outputs/three_leg_random_selection_null.csv')
    write_df(coverage, 'outputs/three_leg_card_availability.csv')
    write_df(diagnostics, 'outputs/three_leg_model_diagnostics.csv')
    write_report(
        summary, by_season, by_era, straight, rank_df, corr_df,
        independence, loo, random_null, coverage, diagnostics,
    )

    print('=== THREE-LEG RESEARCH SUMMARY ===')
    print(summary.to_string(index=False))
    print('=== PRIMARY RANDOM-SELECTION NULL ===')
    print(random_null.to_string(index=False))
    print('=== PRIMARY DEPENDENCE CHECK ===')
    print(independence.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rigorous research-only three-leg parlay audit.')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
