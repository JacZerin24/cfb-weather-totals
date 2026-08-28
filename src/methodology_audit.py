from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .deep_research import prep
from .fcs_model import (
    FCS_CATEGORICAL_FEATURES,
    FCS_NUMERIC_FEATURES,
    historical_fcs_training,
)
from .model_bakeoff import feature_lists, prep_features, reg_models
from .utils import ensure_dir, read_df, write_df

BREAKEVEN = 110 / 210
GENERAL_EDGE = 3.5
GENERAL_TOTAL = 56.0
FCS_EDGE = 7.5
FCS_TOTAL = 56.0

DYNAMIC_WEATHER_NUMS = {
    'wind_mph', 'temperature_f', 'humidity', 'precipitation',
    'snowfall', 'dewpoint_f', 'pressure',
}
DYNAMIC_WEATHER_CATS = {'wind_bin', 'temp_bin'}

GENERAL_VARIANTS = {
    'current': (set(), set()),
    'no_dynamic_weather': (DYNAMIC_WEATHER_NUMS, DYNAMIC_WEATHER_CATS),
    'no_line_provider': (set(), {'line_provider'}),
    'no_weather_or_provider': (DYNAMIC_WEATHER_NUMS, DYNAMIC_WEATHER_CATS | {'line_provider'}),
}
FCS_VARIANTS = GENERAL_VARIANTS

GENERAL_EDGE_GRID = [2.5, 3.5, 5.0, 6.0, 7.0]
GENERAL_TOTAL_GRID = [0.0, 52.0, 54.0, 56.0, 58.0, 60.0, 63.0, 66.0, 70.0]
FCS_EDGE_GRID = [1.5, 2.5, 3.5, 5.0, 6.0, 7.5]
FCS_TOTAL_GRID = [0.0, 49.0, 52.0, 54.0, 56.0, 58.0, 60.0]


def _wilson(wins: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = 1.96
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return c - m, c + m


def grade_under(frame: pd.DataFrame, edge: float, minimum_total: float) -> dict[str, float | int]:
    mask = pd.to_numeric(frame['pred_market_residual'], errors='coerce').le(-edge)
    if minimum_total > 0:
        mask &= pd.to_numeric(frame['closing_total'], errors='coerce').ge(minimum_total)
    sub = frame[mask].copy()
    diff = pd.to_numeric(sub['actual_total_points'], errors='coerce') - pd.to_numeric(sub['closing_total'], errors='coerce')
    wins = int(diff.lt(0).sum())
    losses = int(diff.gt(0).sum())
    pushes = int(diff.eq(0).sum())
    graded = wins + losses
    hit = wins / graded if graded else np.nan
    net = wins * (100 / 110) - losses
    low, high = _wilson(wins, graded)
    p_value = float(binomtest(wins, graded, BREAKEVEN, alternative='greater').pvalue) if graded else np.nan
    return {
        'games': int(len(sub)),
        'graded': int(graded),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': hit,
        'roi_per_1u': net / graded if graded else np.nan,
        'wilson_low': low,
        'wilson_high': high,
        'one_sided_binom_p_vs_-110': p_value,
    }


def by_season(frame: pd.DataFrame, edge: float, minimum_total: float) -> pd.DataFrame:
    rows = []
    for season, group in frame.groupby('season'):
        rows.append({'season': int(season), **grade_under(group, edge, minimum_total)})
    return pd.DataFrame(rows)


def general_oof(raw: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prep(raw)
    nums, cats = feature_lists(df)
    drop_nums, drop_cats = GENERAL_VARIANTS[variant]
    nums = [c for c in nums if c not in drop_nums]
    cats = [c for c in cats if c not in drop_cats]
    df = prep_features(df, cats)

    parts: list[pd.DataFrame] = []
    diag: list[dict] = []
    for season in sorted(df['season'].dropna().astype(int).unique()):
        train = df[df['season'] < season].copy()
        test = df[df['season'] == season].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        model = reg_models(nums, cats)['hist_gradient_boosting']
        model.fit(train[nums + cats], train['market_residual'])
        pred = model.predict(test[nums + cats])
        test = test.assign(pred_market_residual=pred)
        parts.append(test)
        diag.append({
            'track': 'general',
            'variant': variant,
            'test_season': int(season),
            'train_games': int(len(train)),
            'test_games': int(len(test)),
            'numeric_features': int(len(nums)),
            'categorical_features': int(len(cats)),
            'model_mae': float(mean_absolute_error(test['market_residual'], pred)),
            'zero_residual_baseline_mae': float(mean_absolute_error(test['market_residual'], np.zeros(len(test)))),
        })
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), pd.DataFrame(diag))


def _fcs_model(nums: list[str], cats: list[str]) -> Pipeline:
    prep_pipe = ColumnTransformer([
        (
            'num',
            Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
            ]),
            nums,
        ),
        (
            'cat',
            Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]),
            cats,
        ),
    ])
    return Pipeline([
        ('prep', prep_pipe),
        ('model', HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.04,
            l2_regularization=0.5,
            min_samples_leaf=35,
            random_state=42,
        )),
    ])


def fcs_oof(data: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    drop_nums, drop_cats = FCS_VARIANTS[variant]
    nums = [c for c in FCS_NUMERIC_FEATURES if c not in drop_nums]
    cats = [c for c in FCS_CATEGORICAL_FEATURES if c not in drop_cats]
    parts: list[pd.DataFrame] = []
    diag: list[dict] = []
    for season in sorted(data['season'].dropna().astype(int).unique()):
        train = data[data['season'] < season].copy()
        test = data[data['season'] == season].copy()
        if len(train) < 500 or len(test) < 100:
            continue
        model = _fcs_model(nums, cats)
        model.fit(train[nums + cats], train['market_residual'])
        pred = model.predict(test[nums + cats])
        test = test.assign(pred_market_residual=pred)
        parts.append(test)
        diag.append({
            'track': 'FCS',
            'variant': variant,
            'test_season': int(season),
            'train_games': int(len(train)),
            'test_games': int(len(test)),
            'numeric_features': int(len(nums)),
            'categorical_features': int(len(cats)),
            'model_mae': float(mean_absolute_error(test['market_residual'], pred)),
            'zero_residual_baseline_mae': float(mean_absolute_error(test['market_residual'], np.zeros(len(test)))),
        })
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), pd.DataFrame(diag))


def _positive_season_rate(frame: pd.DataFrame, edge: float, total: float) -> float:
    vals = []
    for _, group in frame.groupby('season'):
        g = grade_under(group, edge, total)
        if g['graded']:
            vals.append(float(g['roi_per_1u']) > 0)
    return float(np.mean(vals)) if vals else np.nan


def choose_rule(
    history: pd.DataFrame,
    edge_grid: list[float],
    total_grid: list[float],
    min_graded: int,
) -> dict | None:
    rows = []
    for edge in edge_grid:
        for total in total_grid:
            g = grade_under(history, edge, total)
            if int(g['graded']) < min_graded:
                continue
            psr = _positive_season_rate(history, edge, total)
            if pd.notna(psr) and psr < 0.60:
                continue
            rows.append({
                'edge': edge,
                'minimum_total': total,
                'positive_season_rate': psr,
                **g,
            })
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda r: (
            float(r['wilson_low']) if pd.notna(r['wilson_low']) else -1,
            int(r['graded']),
            -float(r['edge']),
        ),
        reverse=True,
    )[0]


def nested_general(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seasons = sorted(pred['season'].dropna().astype(int).unique())
    for season in seasons:
        history = pred[pred['season'] < season].copy()
        test = pred[pred['season'] == season].copy()
        if history['season'].nunique() < 3 or len(history) < 1500:
            continue
        selected = choose_rule(history, GENERAL_EDGE_GRID, GENERAL_TOTAL_GRID, min_graded=100)
        if not selected:
            continue
        result = grade_under(test, float(selected['edge']), float(selected['minimum_total']))
        rows.append({
            'test_season': season,
            'selected_edge_from_prior_oof': selected['edge'],
            'selected_total_from_prior_oof': selected['minimum_total'],
            'selection_history_graded': selected['graded'],
            'selection_history_hit_rate': selected['hit_rate'],
            'selection_history_wilson_low': selected['wilson_low'],
            'selection_history_positive_season_rate': selected['positive_season_rate'],
            **{f'test_{k}': v for k, v in result.items()},
        })
    return pd.DataFrame(rows)


def nested_fcs(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seasons = sorted(pred['season'].dropna().astype(int).unique())
    for season in seasons:
        history = pred[pred['season'] < season].copy()
        test = pred[pred['season'] == season].copy()
        if history.empty:
            continue
        min_graded = 10 if history['season'].nunique() == 1 else 20
        selected = choose_rule(history, FCS_EDGE_GRID, FCS_TOTAL_GRID, min_graded=min_graded)
        if not selected:
            continue
        result = grade_under(test, float(selected['edge']), float(selected['minimum_total']))
        rows.append({
            'test_season': season,
            'selected_edge_from_prior_oof': selected['edge'],
            'selected_total_from_prior_oof': selected['minimum_total'],
            'selection_history_graded': selected['graded'],
            'selection_history_hit_rate': selected['hit_rate'],
            'selection_history_wilson_low': selected['wilson_low'],
            'selection_history_positive_season_rate': selected['positive_season_rate'],
            **{f'test_{k}': v for k, v in result.items()},
        })
    return pd.DataFrame(rows)


def provider_summary(fcs: pd.DataFrame) -> pd.DataFrame:
    if 'line_provider' not in fcs.columns:
        return pd.DataFrame()
    out = (
        fcs.groupby(['season', 'line_provider'], dropna=False)
        .size()
        .reset_index(name='games')
        .sort_values(['season', 'games'], ascending=[True, False])
    )
    out['season_total'] = out.groupby('season')['games'].transform('sum')
    out['share'] = out['games'] / out['season_total']
    return out


def weather_coverage(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna(subset=['closing_total', 'actual_total_points', 'season']).copy()
    dynamic = [c for c in DYNAMIC_WEATHER_NUMS if c in df.columns]
    rows = []
    for season, group in df.groupby('season'):
        any_weather = group[dynamic].notna().any(axis=1) if dynamic else pd.Series(False, index=group.index)
        rows.append({
            'season': int(season),
            'games_with_total_result': int(len(group)),
            'games_with_any_dynamic_weather': int(any_weather.sum()),
            'weather_coverage': float(any_weather.mean()) if len(group) else np.nan,
        })
    return pd.DataFrame(rows)


def _aggregate_nested(nested: pd.DataFrame) -> dict[str, float | int]:
    if nested.empty:
        return {}
    wins = int(nested['test_wins'].sum())
    losses = int(nested['test_losses'].sum())
    pushes = int(nested['test_pushes'].sum())
    graded = wins + losses
    net = wins * (100 / 110) - losses
    low, high = _wilson(wins, graded)
    return {
        'graded': graded,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / graded if graded else np.nan,
        'roi_per_1u': net / graded if graded else np.nan,
        'wilson_low': low,
        'wilson_high': high,
    }


def _fmt_pct(x: float) -> str:
    return 'n/a' if pd.isna(x) else f'{x:.1%}'


def write_report(
    variant_summary: pd.DataFrame,
    season_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    nested_general_df: pd.DataFrame,
    nested_fcs_df: pd.DataFrame,
    weather_cov: pd.DataFrame,
    provider_df: pd.DataFrame,
) -> None:
    out = ensure_dir('outputs') / 'methodology_audit_summary.md'
    general = variant_summary[(variant_summary.track == 'general') & (variant_summary.variant == 'current')].iloc[0]
    fcs = variant_summary[(variant_summary.track == 'FCS') & (variant_summary.variant == 'current')].iloc[0]
    nested_g = _aggregate_nested(nested_general_df)
    nested_f = _aggregate_nested(nested_fcs_df)

    lines = [
        '# Methodology Audit',
        '',
        'This audit is intentionally stricter than the production dashboard. It distinguishes chronological model validation from post-hoc strategy selection.',
        '',
        '## Executive result',
        '',
        '- The season walk-forward model fitting is chronologically correct: every test season is scored from models fit only on prior seasons.',
        '- The current production screens are **research hypotheses, not fully independent confirmatory validation**, because model/threshold/filter choices were made after reviewing outcomes on the walk-forward prediction pool.',
        '- Historical weather and live NWS weather are not yet a matched information set: the historical file contains CFBD game-weather values, while live decisions use a forecast available before kickoff.',
        '- Historical backtests use the historical CFBD `overUnder` value, while the live system acts on a Friday/Saturday current market. Timestamp-matched historical entry lines are not available in this dataset.',
        '',
        '## Frozen current-screen results on out-of-fold predictions',
        '',
        f"- General HGB under {GENERAL_EDGE:g}+ with total {GENERAL_TOTAL:g}+: {int(general.wins)}-{int(general.losses)} over {int(general.graded)} graded, hit {_fmt_pct(general.hit_rate)}, paper ROI {_fmt_pct(general.roi_per_1u)}, Wilson 95% {_fmt_pct(general.wilson_low)}–{_fmt_pct(general.wilson_high)}, nominal one-sided p={general.one_sided_binom_p_vs_-110:.4f}.",
        f"- FCS HGB under {FCS_EDGE:g}+ with total {FCS_TOTAL:g}+: {int(fcs.wins)}-{int(fcs.losses)} over {int(fcs.graded)} graded, hit {_fmt_pct(fcs.hit_rate)}, paper ROI {_fmt_pct(fcs.roi_per_1u)}, Wilson 95% {_fmt_pct(fcs.wilson_low)}–{_fmt_pct(fcs.wilson_high)}, nominal one-sided p={fcs.one_sided_binom_p_vs_-110:.4f}.",
        '',
        'Nominal p-values above do **not** correct for the many model, threshold, total, provider, weather, and interaction screens examined during research.',
        '',
        '## Ablation tests',
        '',
        variant_summary.to_markdown(index=False),
        '',
        'Interpretation: if removing dynamic weather produces similar or better performance, the historical evidence does not establish that weather adds incremental predictive skill beyond the market/team/context features.',
        '',
        '## Current screens by test season',
        '',
        season_summary.to_markdown(index=False),
        '',
        '## Global point-forecast diagnostics',
        '',
        diagnostics.to_markdown(index=False),
        '',
        'A model MAE above the zero-residual baseline means the sportsbook total is a better unconditional point forecast than adding the model residual. Selective subsets can still be useful, but they require stronger anti-selection safeguards.',
        '',
        '## Illustrative nested rule-selection test',
        '',
        'The selector below chooses a rule using only earlier out-of-fold seasons, then grades the next untouched season. This is stricter than choosing one rule from the full out-of-fold pool, but it is still an audit designed after the fact and is not a substitute for a truly untouched 2026 holdout.',
        '',
        '### General',
        nested_general_df.to_markdown(index=False) if not nested_general_df.empty else '_No nested general rows._',
        '',
        f"Nested general aggregate: {nested_g if nested_g else 'n/a'}",
        '',
        '### FCS',
        nested_fcs_df.to_markdown(index=False) if not nested_fcs_df.empty else '_No nested FCS rows._',
        '',
        f"Nested FCS aggregate: {nested_f if nested_f else 'n/a'}",
        '',
        '## Historical weather coverage',
        '',
        weather_cov.to_markdown(index=False),
        '',
        'High historical weather completeness does not prove that those values were forecasts available at the same Friday/Saturday lead time as the live NWS inputs. Forecast provenance/issuance time must be matched before calling weather impact prospectively validated.',
        '',
        '## FCS historical line-provider mix',
        '',
        provider_df.head(30).to_markdown(index=False) if not provider_df.empty else '_No provider data._',
        '',
        'Because `line_provider` is a model feature and the live OddsPapi provider labels differ from historical CFBD provider labels, provider ablation is included above. Unknown live provider categories are ignored by one-hot encoding.',
        '',
        '## Scientific disposition',
        '',
        '1. **Sound for exploratory research:** chronological model fitting, prior-season team controls, pinned dependencies, explicit no-play rules, and paper tracking are strong.',
        '2. **Not yet sound for a confirmatory claim of betting edge:** final rule selection is post-hoc relative to the OOF pool, the weather information set is not replay-matched, and the entry-line timing is not historically matched.',
        '3. **2026 should be treated as the first true prospective validation period.** Freeze rules, store every Friday/Saturday snapshot immutably, capture the later closing line, and do not tune thresholds from 2026 results until a predeclared review point.',
    ]
    out.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    raw = read_df('data/processed/modeling_dataset.csv')
    if raw.empty:
        raise RuntimeError('modeling_dataset.csv is required for the methodology audit.')

    variant_rows = []
    season_rows = []
    diag_parts = []
    general_current = pd.DataFrame()

    for variant in GENERAL_VARIANTS:
        pred, diag = general_oof(raw, variant)
        if pred.empty:
            continue
        g = grade_under(pred, GENERAL_EDGE, GENERAL_TOTAL)
        variant_rows.append({'track': 'general', 'variant': variant, **g})
        season = by_season(pred, GENERAL_EDGE, GENERAL_TOTAL)
        season['track'] = 'general'
        season['variant'] = variant
        season_rows.append(season)
        diag_parts.append(diag)
        if variant == 'current':
            general_current = pred

    fcs_data = historical_fcs_training()
    fcs_current = pd.DataFrame()
    for variant in FCS_VARIANTS:
        pred, diag = fcs_oof(fcs_data, variant)
        if pred.empty:
            continue
        g = grade_under(pred, FCS_EDGE, FCS_TOTAL)
        variant_rows.append({'track': 'FCS', 'variant': variant, **g})
        season = by_season(pred, FCS_EDGE, FCS_TOTAL)
        season['track'] = 'FCS'
        season['variant'] = variant
        season_rows.append(season)
        diag_parts.append(diag)
        if variant == 'current':
            fcs_current = pred

    variants = pd.DataFrame(variant_rows)
    seasons = pd.concat(season_rows, ignore_index=True) if season_rows else pd.DataFrame()
    diagnostics = pd.concat(diag_parts, ignore_index=True) if diag_parts else pd.DataFrame()
    nested_g = nested_general(general_current) if not general_current.empty else pd.DataFrame()
    nested_f = nested_fcs(fcs_current) if not fcs_current.empty else pd.DataFrame()
    weather_cov = weather_coverage(raw)
    providers = provider_summary(fcs_data)

    write_df(variants, 'outputs/methodology_audit_variants.csv')
    write_df(seasons, 'outputs/methodology_audit_by_season.csv')
    write_df(diagnostics, 'outputs/methodology_audit_diagnostics.csv')
    write_df(nested_g, 'outputs/methodology_audit_nested_general.csv')
    write_df(nested_f, 'outputs/methodology_audit_nested_fcs.csv')
    write_df(weather_cov, 'outputs/methodology_audit_weather_coverage.csv')
    write_df(providers, 'outputs/methodology_audit_fcs_providers.csv')
    write_report(variants, seasons, diagnostics, nested_g, nested_f, weather_cov, providers)

    print('=== METHODOLOGY AUDIT KEY RESULTS ===')
    print(variants.to_string(index=False))
    print('=== NESTED GENERAL ===')
    print(nested_g.to_string(index=False) if not nested_g.empty else 'none')
    print('=== NESTED FCS ===')
    print(nested_f.to_string(index=False) if not nested_f.empty else 'none')


if __name__ == '__main__':
    main()
