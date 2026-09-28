from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .cfbd_client import CFBDClient
from .deep_research import prep, settle, units
from .inseason_features import build_historical_inseason_features, merge_inseason_features
from .model_bakeoff import prep_features, reg_models
from .utils import ensure_dir, read_df, write_df


PRODUCTION_EDGE = 4.0
PRODUCTION_TOTAL = 56.0
BASE_NUMS = [
    'closing_total', 'wind_mph', 'temperature_f', 'humidity', 'precipitation',
    'snowfall', 'dewpoint_f', 'pressure', 'home_pregame_elo', 'away_pregame_elo',
]
BASE_CATS = [
    'game_indoors_bool', 'neutral_site', 'conference_game', 'line_provider',
    'wind_bin', 'temp_bin', 'total_bin', 'fbs_vs_fbs', 'home_conference', 'away_conference',
]


def feature_set(df: pd.DataFrame, variant: str) -> tuple[list[str], list[str]]:
    base = [c for c in BASE_NUMS if c in df.columns and pd.to_numeric(df[c], errors='coerce').notna().any()]
    prior = [
        c for c in df.columns
        if c.startswith(('home_prior_', 'away_prior_'))
        and pd.to_numeric(df[c], errors='coerce').notna().any()
    ]
    current = [
        c for c in df.columns
        if c.startswith(('home_inseason_', 'away_inseason_'))
        and c not in {'home_inseason_inseason_conference', 'away_inseason_inseason_conference'}
        and pd.to_numeric(df[c], errors='coerce').notna().any()
    ]

    if variant == 'baseline_prior':
        nums = base + prior
    elif variant == 'inseason_only':
        nums = base + current
    elif variant == 'hybrid':
        nums = base + prior + current
    else:
        raise ValueError(variant)

    cats = [c for c in BASE_CATS if c in df.columns]
    return nums, cats


def grade_under_screen(pred: pd.DataFrame, variant: str, season: int | None = None) -> dict:
    mask = (
        pd.to_numeric(pred['pred_market_residual'], errors='coerce').le(-PRODUCTION_EDGE)
        & pd.to_numeric(pred['closing_total'], errors='coerce').ge(PRODUCTION_TOTAL)
    )
    plays = pred[mask].copy()
    results = settle(plays['actual_total_points'], plays['closing_total'], 'under') if not plays.empty else pd.Series(dtype=str)
    graded = results[results.ne('push')]
    wins = int(graded.eq('win').sum())
    losses = int(graded.eq('loss').sum())
    pushes = int(results.eq('push').sum())
    n = wins + losses
    net = float(results.map(units).sum()) if len(results) else 0.0
    return {
        'variant': variant,
        'test_season': season if season is not None else 'ALL',
        'screen': 'under_edge_4_total_56',
        'games': int(len(plays)),
        'graded': n,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / n if n else np.nan,
        'net_units_1u_each': net,
        'roi_per_1u': net / n if n else np.nan,
        'avg_pred_edge': float(plays['pred_market_residual'].abs().mean()) if len(plays) else np.nan,
        'avg_actual_residual': float((plays['actual_total_points'] - plays['closing_total']).mean()) if len(plays) else np.nan,
    }


def run_variant(df: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    nums, cats = feature_set(df, variant)
    work = prep_features(df.copy(), cats)
    model = reg_models(nums, cats)['hist_gradient_boosting']
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    seasonal: list[dict] = []

    for season in sorted(work['season'].dropna().astype(int).unique()):
        train = work[work['season'] < season].copy()
        test = work[work['season'] == season].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        model.fit(train[nums + cats], train['market_residual'])
        pred = model.predict(test[nums + cats])
        test = test.assign(pred_market_residual=pred)
        parts.append(test)
        seasonal.append(grade_under_screen(test, variant, season))
        diagnostics.append({
            'variant': variant,
            'test_season': season,
            'train_games': len(train),
            'test_games': len(test),
            'numeric_features': len(nums),
            'categorical_features': len(cats),
            'prior_feature_count': len([c for c in nums if c.startswith(('home_prior_', 'away_prior_'))]),
            'inseason_feature_count': len([c for c in nums if c.startswith(('home_inseason_', 'away_inseason_'))]),
            'mae': mean_absolute_error(test['market_residual'], pred),
        })

    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), diagnostics, seasonal


def main() -> None:
    base = prep(read_df('data/processed/modeling_dataset.csv'))
    client = CFBDClient()

    features = build_historical_inseason_features(client, base)
    write_df(features, 'data/processed/team_inseason_advanced_features.csv')
    enriched = merge_inseason_features(base, features)

    feature_cols = [c for c in enriched.columns if c.startswith(('home_inseason_adv_', 'away_inseason_adv_'))]
    enriched['inseason_feature_ready'] = enriched[feature_cols].notna().any(axis=1) if feature_cols else False

    summary_rows: list[dict] = []
    season_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    for variant in ['baseline_prior', 'inseason_only', 'hybrid']:
        pred, diag, seasonal = run_variant(enriched, variant)
        if pred.empty:
            continue
        row = grade_under_screen(pred, variant)
        row['mae'] = mean_absolute_error(pred['market_residual'], pred['pred_market_residual'])
        row['test_games'] = len(pred)
        row['inseason_ready_games'] = int(pred.get('inseason_feature_ready', pd.Series(False, index=pred.index)).sum())
        summary_rows.append(row)
        season_rows.extend(seasonal)
        diagnostic_rows.extend(diag)

    summary = pd.DataFrame(summary_rows)
    by_season = pd.DataFrame(season_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    write_df(summary, 'outputs/inseason_feature_experiment.csv')
    write_df(by_season, 'outputs/inseason_feature_experiment_by_season.csv')

    lines = [
        '# In-Season Team Feature Experiment',
        '',
        'Purpose: compare the current production HGB prior-season team controls with leakage-safe current-season advanced team features.',
        '',
        'Every current-season snapshot uses only weeks completed before the game being predicted.',
        'The production decision screen is UNDER residual <= -4.0 with market total >= 56.',
        '',
        '## Overall comparison',
        '',
        summary.to_markdown(index=False) if not summary.empty else '_No results._',
        '',
        '## By-season production screen',
        '',
        by_season.to_markdown(index=False) if not by_season.empty else '_No seasonal results._',
        '',
        '## Walk-forward diagnostics',
        '',
        diagnostics.to_markdown(index=False) if not diagnostics.empty else '_No diagnostics._',
        '',
        'Promotion guardrail: do not replace the operational feature set based on intuition alone. Prefer a challenger that improves the exact production-screen out-of-sample hit rate/ROI without a material MAE regression and without depending on one isolated test season.',
    ]
    out = ensure_dir('outputs') / 'inseason_feature_experiment.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
