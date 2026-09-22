from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from .deep_research import prep
from .model_bakeoff import feature_lists, prep_features, preprocessor
from .utils import ensure_dir, read_df, write_df

EDGE_GRID = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
TOTAL_GRID = [52.0, 54.0, 56.0, 58.0, 60.0]


def hgb(nums: list[str], cats: list[str], early_stopping: str | bool) -> Pipeline:
    return Pipeline([
        ('prep', preprocessor(nums, cats)),
        ('model', HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.04,
            l2_regularization=0.5,
            min_samples_leaf=35,
            random_state=42,
            early_stopping=early_stopping,
        )),
    ])


def units(wins: int, losses: int) -> float:
    return wins * (100.0 / 110.0) - losses


def grade(plays: pd.DataFrame) -> dict[str, float | int]:
    if plays.empty:
        return {'games': 0, 'graded': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'hit_rate': np.nan, 'net_units': 0.0, 'roi': np.nan}
    diff = pd.to_numeric(plays['actual_total_points'], errors='coerce') - pd.to_numeric(plays['closing_total'], errors='coerce')
    pushes = int((diff == 0).sum())
    wins = int((diff < 0).sum())
    losses = int((diff > 0).sum())
    graded = wins + losses
    net = units(wins, losses)
    return {
        'games': int(len(plays)),
        'graded': graded,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / graded if graded else np.nan,
        'net_units': net,
        'roi': net / graded if graded else np.nan,
    }


def walk_forward(df: pd.DataFrame, nums: list[str], cats: list[str], regime: str, early_stopping: str | bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_parts: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for season in sorted(df['season'].dropna().astype(int).unique()):
        train = df[df['season'] < season].copy()
        test = df[df['season'] == season].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        model = hgb(nums, cats, early_stopping)
        model.fit(train[nums + cats], train['market_residual'])
        pred = model.predict(test[nums + cats])
        scored = test.assign(pred_market_residual=pred, regime=regime)
        pred_parts.append(scored)
        estimator = model.named_steps['model']
        fbs = scored[scored['fbs_vs_fbs'].astype(bool)]
        diagnostics.append({
            'regime': regime,
            'test_season': int(season),
            'train_games': int(len(train)),
            'test_games': int(len(test)),
            'fbs_test_games': int(len(fbs)),
            'n_iter': int(estimator.n_iter_),
            'avg_abs_pred_edge_all': float(np.mean(np.abs(pred))),
            'avg_abs_pred_edge_fbs': float(fbs['pred_market_residual'].abs().mean()) if len(fbs) else np.nan,
            'min_fbs_pred_residual': float(fbs['pred_market_residual'].min()) if len(fbs) else np.nan,
            'fbs_under_edge_3p5_count': int((fbs['pred_market_residual'] <= -3.5).sum()) if len(fbs) else 0,
        })
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return predictions, pd.DataFrame(diagnostics)


def threshold_grid(pred: pd.DataFrame) -> pd.DataFrame:
    fbs = pred[pred['fbs_vs_fbs'].astype(bool)].copy()
    rows: list[dict] = []
    for edge in EDGE_GRID:
        for total in TOTAL_GRID:
            base = fbs[(fbs['pred_market_residual'] <= -edge) & (fbs['closing_total'] >= total)].copy()
            all_stats = grade(base)
            recent = base[base['season'].astype(int) >= 2022]
            recent_stats = grade(recent)
            holdout = base[base['season'].astype(int) == 2025]
            holdout_stats = grade(holdout)

            season_rois = []
            for _, g in base.groupby(base['season'].astype(int)):
                season_rois.append(grade(g)['roi'])
            season_rois = [x for x in season_rois if pd.notna(x)]
            recent_season_rois = []
            for _, g in recent.groupby(recent['season'].astype(int)):
                recent_season_rois.append(grade(g)['roi'])
            recent_season_rois = [x for x in recent_season_rois if pd.notna(x)]

            rows.append({
                'edge_threshold': edge,
                'total_min': total,
                **{f'all_{k}': v for k, v in all_stats.items()},
                **{f'recent_2022_plus_{k}': v for k, v in recent_stats.items()},
                **{f'holdout_2025_{k}': v for k, v in holdout_stats.items()},
                'positive_seasons': int(sum(x > 0 for x in season_rois)),
                'season_count': int(len(season_rois)),
                'recent_positive_seasons': int(sum(x > 0 for x in recent_season_rois)),
                'recent_season_count': int(len(recent_season_rois)),
            })
    out = pd.DataFrame(rows)
    out['recent_positive_season_rate'] = out['recent_positive_seasons'] / out['recent_season_count'].replace(0, np.nan)
    out['research_eligible'] = (
        (out['all_graded'] >= 150)
        & (out['recent_2022_plus_graded'] >= 50)
        & (out['holdout_2025_graded'] >= 10)
        & (out['recent_2022_plus_roi'] > 0)
        & (out['holdout_2025_roi'] > 0)
        & (out['recent_positive_season_rate'] >= 0.75)
    )
    return out.sort_values(
        ['research_eligible', 'recent_2022_plus_roi', 'holdout_2025_roi', 'all_graded'],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def write_summary(diag: pd.DataFrame, grid: pd.DataFrame) -> None:
    path = ensure_dir('outputs') / 'hgb_regime_audit.md'
    lines = [
        '# HGB Training-Regime Audit',
        '',
        'Research-only comparison of scikit-learn early_stopping=auto versus an explicitly pinned early_stopping=False production candidate.',
        'Walk-forward fits use only prior seasons. The threshold grid below is FBS-vs-FBS UNDER only and is not automatically promoted to production.',
        '',
        '## Regime diagnostics',
        '',
        diag.to_markdown(index=False) if not diag.empty else '_No diagnostics produced._',
        '',
        '## Fixed-regime FBS threshold grid',
        '',
        grid.head(30).to_markdown(index=False) if not grid.empty else '_No grid results produced._',
        '',
        '## Guardrails',
        '',
        '- research_eligible requires at least 150 all-period graded plays, 50 graded plays since 2022, and 10 graded plays in the 2025 holdout.',
        '- It also requires positive ROI since 2022, positive ROI in 2025, and positive ROI in at least 75% of recent test seasons.',
        '- This report diagnoses model/training behavior. It does not change the frozen live production thresholds by itself.',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    df = prep(read_df('data/processed/modeling_dataset.csv'))
    nums, cats = feature_lists(df)
    df = prep_features(df, cats)

    auto_pred, auto_diag = walk_forward(df, nums, cats, 'auto', 'auto')
    fixed_pred, fixed_diag = walk_forward(df, nums, cats, 'fixed_false', False)
    diagnostics = pd.concat([auto_diag, fixed_diag], ignore_index=True)
    grid = threshold_grid(fixed_pred)

    write_df(diagnostics, 'outputs/hgb_regime_diagnostics.csv')
    write_df(grid, 'outputs/hgb_fixed_threshold_grid.csv')
    write_summary(diagnostics, grid)

    print(diagnostics.to_string(index=False))
    print('\nTop fixed-regime threshold candidates:')
    print(grid.head(15).to_string(index=False))


if __name__ == '__main__':
    main()
