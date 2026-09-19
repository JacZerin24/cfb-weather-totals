from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .deep_research import BREAKEVEN, prep, settle, units, wilson
from .edge_refinement import add_weather_flags, max_drawdown, numeric, safe_bool
from .fcs_model import (
    FCS_CATEGORICAL_FEATURES,
    FCS_NUMERIC_FEATURES,
    build_fcs_model,
    historical_fcs_training,
)
from .model_bakeoff import feature_lists, prep_features, reg_models
from .utils import ensure_dir, read_df, write_df

HGB = 'hist_gradient_boosting'
EDGE_THRESHOLDS = [2.5, 3.5, 5.0, 6.0, 7.5]
DYNAMIC_WEATHER_NUMS = {
    'wind_mph', 'temperature_f', 'humidity', 'precipitation',
    'snowfall', 'dewpoint_f', 'pressure',
}
DYNAMIC_WEATHER_CATS = {'wind_bin', 'temp_bin'}
VARIANTS = {
    'current': (set(), set()),
    'no_dynamic_weather': (DYNAMIC_WEATHER_NUMS, DYNAMIC_WEATHER_CATS),
    'no_line_provider': (set(), {'line_provider'}),
    'no_weather_or_provider': (DYNAMIC_WEATHER_NUMS, DYNAMIC_WEATHER_CATS | {'line_provider'}),
}
BOOTSTRAP_DRAWS = 30000
RNG_SEED = 20260829


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    edge: float
    description: str
    fn: Callable[[pd.DataFrame], pd.Series]


def exact_binom_greater(wins: int, n: int, p0: float = BREAKEVEN) -> float:
    if n <= 0:
        return np.nan
    if wins <= 0:
        return 1.0
    logs = []
    for k in range(wins, n + 1):
        logs.append(
            math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p0) + (n - k) * math.log1p(-p0)
        )
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(x - m) for x in logs))


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors='coerce')
    valid = p.dropna().sort_values()
    out = pd.Series(np.nan, index=p.index, dtype='float64')
    m = len(valid)
    if not m:
        return out
    raw = valid.to_numpy(dtype=float)
    adj = raw * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out.loc[valid.index] = adj
    return out


def grade(df: pd.DataFrame, side: str = 'over') -> dict[str, float | int]:
    if df.empty:
        return {
            'games': 0, 'graded': 0, 'wins': 0, 'losses': 0, 'pushes': 0,
            'hit_rate': np.nan, 'roi_per_1u': np.nan, 'wilson_low': np.nan,
            'wilson_high': np.nan, 'p_one_sided_vs_minus110': np.nan,
            'net_units': 0.0, 'max_drawdown_units': 0.0,
        }
    outcomes = settle(df['actual_total_points'], df['closing_total'], side)
    graded = outcomes[outcomes != 'push']
    wins = int((graded == 'win').sum())
    losses = int((graded == 'loss').sum())
    pushes = int((outcomes == 'push').sum())
    n = wins + losses
    hit = wins / n if n else np.nan
    low, high = wilson(wins, n)
    net = float(outcomes.map(units).sum())
    ordered = df.assign(_units=outcomes.map(units)).sort_values(
        [c for c in ['season', 'week', 'game_id'] if c in df.columns]
    )
    return {
        'games': int(len(df)),
        'graded': n,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': hit,
        'roi_per_1u': net / n if n else np.nan,
        'wilson_low': low,
        'wilson_high': high,
        'p_one_sided_vs_minus110': exact_binom_greater(wins, n),
        'net_units': net,
        'max_drawdown_units': max_drawdown(ordered['_units']) if len(ordered) else 0.0,
    }


def scope_frames(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    home = pred.get('home_classification', pd.Series('', index=pred.index)).astype(str).str.lower()
    away = pred.get('away_classification', pd.Series('', index=pred.index)).astype(str).str.lower()
    fcs_fcs = home.eq('fcs') & away.eq('fcs')
    fbs_fbs = home.eq('fbs') & away.eq('fbs')
    return {
        'general_live_scope': pred[~fcs_fcs].copy(),
        'fbs_vs_fbs': pred[fbs_fbs].copy(),
        'mixed_all_diagnostic': pred.copy(),
    }


def general_walk_forward(raw: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prep(raw)
    nums, cats = feature_lists(df)
    drop_nums, drop_cats = VARIANTS[variant]
    nums = [c for c in nums if c not in drop_nums]
    cats = [c for c in cats if c not in drop_cats]
    df = prep_features(df, cats)
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for season in sorted(df['season'].dropna().astype(int).unique()):
        train = df[df['season'] < season].copy()
        test = df[df['season'] == season].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        model = reg_models(nums, cats)[HGB]
        model.fit(train[nums + cats], train['market_residual'])
        pred = model.predict(test[nums + cats])
        test = test.assign(pred_market_residual=pred)
        parts.append(test)
        diagnostics.append({
            'track': 'general',
            'variant': variant,
            'test_season': int(season),
            'train_games': int(len(train)),
            'test_games': int(len(test)),
            'avg_pred_residual': float(np.mean(pred)),
            'avg_actual_residual': float(test['market_residual'].mean()),
        })
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), pd.DataFrame(diagnostics))


def candidate_grid() -> list[Candidate]:
    cands: list[Candidate] = []
    total_bins = {
        'total_le_42': lambda d: numeric(d, 'closing_total') <= 42,
        'total_42_49': lambda d: numeric(d, 'closing_total').gt(42) & numeric(d, 'closing_total').le(49),
        'total_49_56': lambda d: numeric(d, 'closing_total').gt(49) & numeric(d, 'closing_total').lt(56),
        'total_56_63': lambda d: numeric(d, 'closing_total').ge(56) & numeric(d, 'closing_total').lt(63),
        'total_ge_63': lambda d: numeric(d, 'closing_total') >= 63,
    }
    total_caps = [49, 52, 54, 56]
    total_floors = [49, 52, 56, 60, 63]
    weather = {
        'wind_le_5': lambda d: numeric(d, 'wind_mph') <= 5,
        'wind_le_10': lambda d: numeric(d, 'wind_mph') <= 10,
        'wind_ge_10': lambda d: numeric(d, 'wind_mph') >= 10,
        'wind_ge_15': lambda d: numeric(d, 'wind_mph') >= 15,
        'temp_ge_70': lambda d: numeric(d, 'temperature_f') >= 70,
        'temp_ge_80': lambda d: numeric(d, 'temperature_f') >= 80,
        'temp_le_55': lambda d: numeric(d, 'temperature_f') <= 55,
        'humid_le_50': lambda d: numeric(d, 'humidity') <= 50,
        'humid_le_60': lambda d: numeric(d, 'humidity') <= 60,
        'humid_ge_80': lambda d: numeric(d, 'humidity') >= 80,
        'outdoor': lambda d: ~safe_bool(d, 'game_indoors_bool'),
        'indoor': lambda d: safe_bool(d, 'game_indoors_bool'),
        'dry': lambda d: ~safe_bool(d, 'precip_flag'),
        'precip': lambda d: safe_bool(d, 'precip_flag'),
    }

    for edge in EDGE_THRESHOLDS:
        base = lambda d, e=edge: numeric(d, 'pred_market_residual') >= e
        cands.append(Candidate(f'over_{edge:g}_all', 'edge_only', edge, f'HGB OVER edge >= {edge:g}.', base))
        for name, fn in total_bins.items():
            cands.append(Candidate(
                f'over_{edge:g}_{name}', 'total_bucket', edge,
                f'HGB OVER edge >= {edge:g}; {name}.',
                lambda d, e=edge, f=fn: (numeric(d, 'pred_market_residual') >= e) & f(d),
            ))
        for cap in total_caps:
            cands.append(Candidate(
                f'over_{edge:g}_total_le_{cap}', 'total_cap', edge,
                f'HGB OVER edge >= {edge:g}; total <= {cap}.',
                lambda d, e=edge, t=cap: (numeric(d, 'pred_market_residual') >= e) & (numeric(d, 'closing_total') <= t),
            ))
        for floor in total_floors:
            cands.append(Candidate(
                f'over_{edge:g}_total_ge_{floor}', 'total_floor', edge,
                f'HGB OVER edge >= {edge:g}; total >= {floor}.',
                lambda d, e=edge, t=floor: (numeric(d, 'pred_market_residual') >= e) & (numeric(d, 'closing_total') >= t),
            ))
        for name, fn in weather.items():
            cands.append(Candidate(
                f'over_{edge:g}_{name}', 'weather_single', edge,
                f'HGB OVER edge >= {edge:g}; {name}.',
                lambda d, e=edge, f=fn: (numeric(d, 'pred_market_residual') >= e) & f(d),
            ))
        # Predeclared, limited interaction set: low/mid/high totals crossed with the most plausible
        # weather modifiers. These are discovery candidates, not automatically validated rules.
        interaction_totals = {
            'total_le_52': lambda d: numeric(d, 'closing_total') <= 52,
            'total_le_56': lambda d: numeric(d, 'closing_total') <= 56,
            'total_ge_56': lambda d: numeric(d, 'closing_total') >= 56,
        }
        interaction_weather = {
            'wind_le_10': weather['wind_le_10'],
            'temp_ge_70': weather['temp_ge_70'],
            'humid_le_60': weather['humid_le_60'],
            'dry': weather['dry'],
            'indoor': weather['indoor'],
        }
        for tname, tfn in interaction_totals.items():
            for wname, wfn in interaction_weather.items():
                cands.append(Candidate(
                    f'over_{edge:g}_{tname}_{wname}', 'total_weather_combo', edge,
                    f'HGB OVER edge >= {edge:g}; {tname}; {wname}.',
                    lambda d, e=edge, tf=tfn, wf=wfn: (numeric(d, 'pred_market_residual') >= e) & tf(d) & wf(d),
                ))
    return cands


def evaluate_candidates(pred: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = add_weather_flags(pred.copy())
    rows = []
    season_rows = []
    for cand in candidate_grid():
        sub = pred[cand.fn(pred).fillna(False)].copy()
        g = grade(sub, 'over')
        per_season = []
        for season, sg in sub.groupby('season'):
            sg_grade = grade(sg, 'over')
            per_season.append(sg_grade['roi_per_1u'])
            season_rows.append({
                'scope': scope, 'candidate': cand.name, 'family': cand.family,
                'edge': cand.edge, 'season': int(season), **sg_grade,
            })
        valid_seasons = [x for x in per_season if pd.notna(x)]
        positive = sum(x > 0 for x in valid_seasons)
        rows.append({
            'scope': scope,
            'candidate': cand.name,
            'family': cand.family,
            'edge': cand.edge,
            'description': cand.description,
            **g,
            'season_count': len(valid_seasons),
            'positive_seasons': positive,
            'positive_season_rate': positive / len(valid_seasons) if valid_seasons else np.nan,
            'recent_2022_plus_graded': int(grade(sub[sub['season'].astype(int) >= 2022], 'over')['graded']) if not sub.empty else 0,
            'recent_2022_plus_roi': grade(sub[sub['season'].astype(int) >= 2022], 'over')['roi_per_1u'] if not sub.empty else np.nan,
        })
    summary = pd.DataFrame(rows)
    summary['bh_fdr_q'] = bh_adjust(summary['p_one_sided_vs_minus110'])
    m = summary['p_one_sided_vs_minus110'].notna().sum()
    summary['bonferroni_p'] = (summary['p_one_sided_vs_minus110'] * m).clip(upper=1.0)
    summary['robustness_flag'] = np.where(
        (summary['graded'] >= 100)
        & (summary['positive_season_rate'] >= 0.60)
        & (summary['recent_2022_plus_graded'] >= 30)
        & (summary['recent_2022_plus_roi'] > 0)
        & (summary['bh_fdr_q'] < 0.10),
        'passes_exploratory_screen',
        'does_not_pass',
    )
    summary = summary.sort_values(['robustness_flag', 'bh_fdr_q', 'roi_per_1u', 'graded'], ascending=[True, True, False, False])
    return summary, pd.DataFrame(season_rows)


def edge_monotonicity(pred: pd.DataFrame, scope: str) -> pd.DataFrame:
    p = pred[pred['pred_market_residual'] >= EDGE_THRESHOLDS[0]].copy()
    bins = [2.5, 3.5, 5.0, 6.0, 7.5, np.inf]
    labels = ['2.5-3.5', '3.5-5.0', '5.0-6.0', '6.0-7.5', '7.5+']
    p['edge_band'] = pd.cut(p['pred_market_residual'], bins=bins, labels=labels, right=False)
    rows = []
    for band, g in p.groupby('edge_band', observed=True):
        rows.append({'scope': scope, 'edge_band': str(band), **grade(g, 'over')})
    return pd.DataFrame(rows)


def variant_summary(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diag_parts = []
    for variant in VARIANTS:
        pred, diag = general_walk_forward(raw, variant)
        if pred.empty:
            continue
        diag_parts.append(diag)
        for scope, scope_df in scope_frames(pred).items():
            for edge in [3.5, 5.0, 7.5]:
                sub = scope_df[scope_df['pred_market_residual'] >= edge]
                rows.append({'track': 'general', 'variant': variant, 'scope': scope, 'edge': edge, **grade(sub, 'over')})
    return pd.DataFrame(rows), (pd.concat(diag_parts, ignore_index=True) if diag_parts else pd.DataFrame())


def select_prior_rule(history: pd.DataFrame, candidates: list[Candidate]) -> Candidate | None:
    scored = []
    for cand in candidates:
        sub = history[cand.fn(history).fillna(False)]
        g = grade(sub, 'over')
        if int(g['graded']) < 100:
            continue
        season_rois = []
        for _, sg in sub.groupby('season'):
            sr = grade(sg, 'over')['roi_per_1u']
            if pd.notna(sr):
                season_rois.append(sr)
        if len(season_rois) < 3:
            continue
        pos_rate = sum(r > 0 for r in season_rois) / len(season_rois)
        if pos_rate < 0.60:
            continue
        recent = sub[sub['season'].astype(int) >= max(int(history['season'].max()) - 2, int(history['season'].min()))]
        recent_grade = grade(recent, 'over')
        if recent_grade['graded'] >= 30 and recent_grade['roi_per_1u'] <= 0:
            continue
        scored.append((float(g['wilson_low']), int(g['graded']), -cand.edge, cand))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return scored[0][3]


def nested_selection(pred: pd.DataFrame, scope: str) -> pd.DataFrame:
    pred = add_weather_flags(pred.copy())
    candidates = candidate_grid()
    rows = []
    for season in sorted(pred['season'].dropna().astype(int).unique()):
        history = pred[pred['season'] < season].copy()
        test = pred[pred['season'] == season].copy()
        if history['season'].nunique() < 3 or len(history) < 1500:
            continue
        chosen = select_prior_rule(history, candidates)
        if chosen is None:
            rows.append({'scope': scope, 'test_season': season, 'selected_candidate': 'NO_RULE_SELECTED'})
            continue
        train_sub = history[chosen.fn(history).fillna(False)]
        test_sub = test[chosen.fn(test).fillna(False)]
        train_grade = grade(train_sub, 'over')
        test_grade = grade(test_sub, 'over')
        rows.append({
            'scope': scope,
            'test_season': season,
            'selected_candidate': chosen.name,
            'selected_family': chosen.family,
            'selected_edge': chosen.edge,
            'history_graded': train_grade['graded'],
            'history_hit_rate': train_grade['hit_rate'],
            'history_roi': train_grade['roi_per_1u'],
            **{f'test_{k}': v for k, v in test_grade.items()},
        })
    return pd.DataFrame(rows)


def aggregate_nested(nested: pd.DataFrame) -> dict[str, float | int]:
    valid = nested[nested.get('selected_candidate', '').ne('NO_RULE_SELECTED')].copy() if not nested.empty else pd.DataFrame()
    if valid.empty:
        return {}
    wins = int(pd.to_numeric(valid['test_wins'], errors='coerce').fillna(0).sum())
    losses = int(pd.to_numeric(valid['test_losses'], errors='coerce').fillna(0).sum())
    pushes = int(pd.to_numeric(valid['test_pushes'], errors='coerce').fillna(0).sum())
    n = wins + losses
    net = wins * (100 / 110) - losses
    low, high = wilson(wins, n)
    return {
        'graded': n, 'wins': wins, 'losses': losses, 'pushes': pushes,
        'hit_rate': wins / n if n else np.nan,
        'roi_per_1u': net / n if n else np.nan,
        'wilson_low': low, 'wilson_high': high,
        'p_one_sided_vs_minus110': exact_binom_greater(wins, n),
    }


def season_block_bootstrap(pred: pd.DataFrame, edge: float, total_rule: str = 'all') -> dict[str, float]:
    base = pred[pred['pred_market_residual'] >= edge].copy()
    if total_rule == 'total_le_56':
        base = base[base['closing_total'] <= 56]
    elif total_rule == 'total_ge_56':
        base = base[base['closing_total'] >= 56]
    if base.empty:
        return {}
    season_units = []
    for season, g in base.groupby('season'):
        gr = grade(g, 'over')
        season_units.append((int(season), int(gr['graded']), float(gr['net_units'])))
    if not season_units:
        return {}
    rng = np.random.default_rng(RNG_SEED + int(edge * 10))
    rois = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, len(season_units), size=len(season_units))
        graded = sum(season_units[i][1] for i in idx)
        net = sum(season_units[i][2] for i in idx)
        if graded:
            rois.append(net / graded)
    if not rois:
        return {}
    arr = np.asarray(rois)
    return {
        'bootstrap_draws': len(arr),
        'roi_p2_5': float(np.quantile(arr, 0.025)),
        'roi_median': float(np.quantile(arr, 0.5)),
        'roi_p97_5': float(np.quantile(arr, 0.975)),
        'prob_roi_gt_0': float(np.mean(arr > 0)),
    }


def fcs_over_research() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = historical_fcs_training()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    features = FCS_NUMERIC_FEATURES + FCS_CATEGORICAL_FEATURES
    parts = []
    for season in sorted(data['season'].dropna().astype(int).unique()):
        train = data[data['season'] < season].copy()
        test = data[data['season'] == season].copy()
        if len(train) < 500 or len(test) < 100:
            continue
        model = build_fcs_model()
        model.fit(train[features], train['market_residual'])
        test = test.assign(pred_market_residual=model.predict(test[features]))
        parts.append(test)
    if not parts:
        return pd.DataFrame(), pd.DataFrame()
    pred = pd.concat(parts, ignore_index=True)
    rows = []
    season_rows = []
    for edge in EDGE_THRESHOLDS:
        for total_name, mask in {
            'all': pd.Series(True, index=pred.index),
            'total_le_52': pred['closing_total'] <= 52,
            'total_le_56': pred['closing_total'] <= 56,
            'total_ge_56': pred['closing_total'] >= 56,
            'total_ge_60': pred['closing_total'] >= 60,
        }.items():
            sub = pred[(pred['pred_market_residual'] >= edge) & mask].copy()
            g = grade(sub, 'over')
            rois = []
            for season, sg in sub.groupby('season'):
                sg_grade = grade(sg, 'over')
                rois.append(sg_grade['roi_per_1u'])
                season_rows.append({'edge': edge, 'total_rule': total_name, 'season': int(season), **sg_grade})
            valid = [r for r in rois if pd.notna(r)]
            rows.append({
                'edge': edge, 'total_rule': total_name, **g,
                'season_count': len(valid),
                'positive_season_rate': sum(r > 0 for r in valid) / len(valid) if valid else np.nan,
            })
    summary = pd.DataFrame(rows)
    summary['bh_fdr_q'] = bh_adjust(summary['p_one_sided_vs_minus110'])
    return summary.sort_values(['bh_fdr_q', 'roi_per_1u'], ascending=[True, False]), pd.DataFrame(season_rows)


def write_report(
    discovery: pd.DataFrame,
    by_season: pd.DataFrame,
    monotonic: pd.DataFrame,
    variants: pd.DataFrame,
    nested: pd.DataFrame,
    bootstrap: pd.DataFrame,
    fcs: pd.DataFrame,
) -> None:
    out = ensure_dir('outputs') / 'over_edge_research_summary.md'
    nested_agg = aggregate_nested(nested)
    robust = discovery[discovery['robustness_flag'].eq('passes_exploratory_screen')].copy()
    lines = [
        '# Rigorous OVER Edge Research',
        '',
        'This study asks whether the HGB residual model supports any defensible OVER strategy. It is deliberately separate from production and does not change live rules.',
        '',
        '## Anti-overfitting protocol',
        '',
        '- Model predictions are season walk-forward: each test season is predicted using prior seasons only.',
        '- The candidate grid is finite and predeclared in code before the research workflow is run.',
        '- Discovery p-values receive Benjamini-Hochberg FDR and Bonferroni corrections across the full candidate grid.',
        '- A discovery candidate is not considered validated simply because it has the best historical ROI.',
        '- Nested selection chooses a rule using only earlier out-of-fold seasons and grades it on the next untouched season.',
        '- Weather/provider ablations, FBS-only scope, edge monotonicity, season stability, drawdown and season-block bootstrap are reported.',
        '- Historical weather and historical line timing still do not exactly reproduce the live Friday/Saturday information set; 2026 prospective tracking remains the final confirmation layer.',
        '',
        '## Discovery grid candidates passing the exploratory screen',
        '',
        robust.head(30).to_markdown(index=False) if not robust.empty else '_None._',
        '',
        '## Top discovery rows (descriptive, post-selection)',
        '',
        discovery.head(40).to_markdown(index=False) if not discovery.empty else '_No rows._',
        '',
        '## Edge monotonicity',
        '',
        monotonic.to_markdown(index=False) if not monotonic.empty else '_No rows._',
        '',
        '## Model-feature ablations',
        '',
        variants.to_markdown(index=False) if not variants.empty else '_No rows._',
        '',
        '## Nested prior-season-only rule selection',
        '',
        nested.to_markdown(index=False) if not nested.empty else '_No nested rows._',
        '',
        f'Nested aggregate: `{nested_agg}`',
        '',
        '## Season-block bootstrap on predeclared broad OVER rules',
        '',
        bootstrap.to_markdown(index=False) if not bootstrap.empty else '_No rows._',
        '',
        '## FCS OVER check',
        '',
        fcs.head(30).to_markdown(index=False) if not fcs.empty else '_No FCS OVER rows._',
        '',
        '## Scientific decision rule',
        '',
        'An OVER strategy should not be promoted operationally unless it shows positive and reasonably stable straight-bet performance, survives multiple-testing correction or a genuinely prior-only/nested test, does not depend on a single era, and then survives prospective 2026 paper tracking. A negative result is an acceptable and useful conclusion.',
    ]
    out.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    raw = read_df('data/processed/modeling_dataset.csv')
    if raw.empty:
        raise RuntimeError('modeling_dataset.csv is required.')

    current_pred, current_diag = general_walk_forward(raw, 'current')
    if current_pred.empty:
        raise RuntimeError('No general walk-forward predictions generated.')
    scopes = scope_frames(current_pred)

    discovery_parts = []
    season_parts = []
    monotonic_parts = []
    nested_parts = []
    for scope in ['general_live_scope', 'fbs_vs_fbs']:
        scoped = scopes[scope]
        discovery, by_season = evaluate_candidates(scoped, scope)
        discovery_parts.append(discovery)
        season_parts.append(by_season)
        monotonic_parts.append(edge_monotonicity(scoped, scope))
        nested_parts.append(nested_selection(scoped, scope))

    discovery_all = pd.concat(discovery_parts, ignore_index=True)
    seasons_all = pd.concat(season_parts, ignore_index=True)
    monotonic_all = pd.concat(monotonic_parts, ignore_index=True)
    nested_all = pd.concat(nested_parts, ignore_index=True)
    variants, variant_diag = variant_summary(raw)

    bootstrap_rows = []
    for scope in ['general_live_scope', 'fbs_vs_fbs']:
        scoped = scopes[scope]
        for edge in [3.5, 5.0, 7.5]:
            for total_rule in ['all', 'total_le_56', 'total_ge_56']:
                result = season_block_bootstrap(scoped, edge, total_rule)
                if result:
                    bootstrap_rows.append({'scope': scope, 'edge': edge, 'total_rule': total_rule, **result})
    bootstrap = pd.DataFrame(bootstrap_rows)

    fcs, fcs_season = fcs_over_research()

    write_df(discovery_all, 'outputs/over_edge_discovery_grid.csv')
    write_df(seasons_all, 'outputs/over_edge_by_season.csv')
    write_df(monotonic_all, 'outputs/over_edge_monotonicity.csv')
    write_df(variants, 'outputs/over_edge_ablation_summary.csv')
    write_df(pd.concat([current_diag, variant_diag], ignore_index=True), 'outputs/over_edge_model_diagnostics.csv')
    write_df(nested_all, 'outputs/over_edge_nested_selection.csv')
    write_df(bootstrap, 'outputs/over_edge_bootstrap.csv')
    write_df(fcs, 'outputs/over_edge_fcs_summary.csv')
    write_df(fcs_season, 'outputs/over_edge_fcs_by_season.csv')
    write_report(discovery_all, seasons_all, monotonic_all, variants, nested_all, bootstrap, fcs)

    print('=== OVER DISCOVERY: GENERAL LIVE SCOPE TOP 25 ===')
    print(discovery_all[discovery_all['scope'].eq('general_live_scope')].head(25).to_string(index=False))
    print('=== OVER EDGE MONOTONICITY ===')
    print(monotonic_all.to_string(index=False))
    print('=== OVER ABLATIONS ===')
    print(variants.to_string(index=False))
    print('=== OVER NESTED SELECTION ===')
    print(nested_all.to_string(index=False))
    print('=== OVER NESTED AGGREGATE GENERAL ===')
    print(aggregate_nested(nested_all[nested_all['scope'].eq('general_live_scope')]))
    print('=== OVER NESTED AGGREGATE FBS ===')
    print(aggregate_nested(nested_all[nested_all['scope'].eq('fbs_vs_fbs')]))
    print('=== OVER BOOTSTRAP ===')
    print(bootstrap.to_string(index=False))
    print('=== FCS OVER TOP 20 ===')
    print(fcs.head(20).to_string(index=False) if not fcs.empty else 'none')


if __name__ == '__main__':
    main()
