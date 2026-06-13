from __future__ import annotations

import numpy as np
import pandas as pd

from .deep_research import BREAKEVEN, prep, settle, units, wilson
from .model_bakeoff import EDGE_THRESHOLDS, feature_lists, prep_features, reg_models
from .utils import ensure_dir, read_df, write_df

MODELS_TO_VALIDATE = ['hist_gradient_boosting', 'elastic_net', 'extra_trees', 'ridge']


def grade_frame(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    rows = []
    for labels, g in df.groupby(label_cols, dropna=False, observed=True):
        if not isinstance(labels, tuple):
            labels = (labels,)
        outcomes = settle(g['actual_total_points'], g['closing_total'], g['side'])
        graded = outcomes[outcomes != 'push']
        wins = int((graded == 'win').sum())
        losses = int((graded == 'loss').sum())
        pushes = int((outcomes == 'push').sum())
        n = wins + losses
        low, high = wilson(wins, n)
        hit = wins / n if n else np.nan
        net = float(outcomes.map(units).sum()) if len(outcomes) else 0.0
        row = dict(zip(label_cols, labels))
        row.update({
            'games': int(len(g)),
            'graded': int(n),
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'hit_rate': hit,
            'hit_rate_minus_breakeven': hit - BREAKEVEN if n else np.nan,
            'hit_rate_wilson_low': low,
            'hit_rate_wilson_high': high,
            'net_units_1u_each': net,
            'roi_per_1u': net / n if n else np.nan,
            'avg_market_residual': float(g['market_residual'].mean()) if len(g) else np.nan,
            'avg_abs_pred_edge': float(g['pred_market_residual'].abs().mean()) if len(g) else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def collect_model_plays(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nums, cats = feature_lists(df)
    df = prep_features(df, cats)
    all_plays = []
    diag = []
    models = reg_models(nums, cats)
    for model_name in MODELS_TO_VALIDATE:
        model = models[model_name]
        pred_parts = []
        for season in sorted(df['season'].dropna().astype(int).unique()):
            train = df[df['season'] < season].copy()
            test = df[df['season'] == season].copy()
            if len(train) < 1000 or len(test) < 100:
                continue
            model.fit(train[nums + cats], train['market_residual'])
            pred = model.predict(test[nums + cats])
            test = test.assign(model=model_name, pred_market_residual=pred)
            pred_parts.append(test)
            diag.append({
                'model': model_name,
                'test_season': season,
                'train_games': len(train),
                'test_games': len(test),
                'avg_abs_pred_edge': float(np.abs(pred).mean()),
                'avg_pred_residual': float(np.mean(pred)),
                'avg_actual_residual': float(test['market_residual'].mean()),
            })
        if not pred_parts:
            continue
        pred_df = pd.concat(pred_parts, ignore_index=True)
        for threshold in EDGE_THRESHOLDS:
            plays = pred_df[pred_df['pred_market_residual'].abs() >= threshold].copy()
            if plays.empty:
                continue
            plays['threshold'] = threshold
            plays['side'] = np.where(plays['pred_market_residual'] > 0, 'over', 'under')
            plays['result'] = settle(plays['actual_total_points'], plays['closing_total'], plays['side'])
            keep = [c for c in [
                'model', 'threshold', 'season', 'week', 'game_id', 'home_team', 'away_team',
                'line_provider', 'closing_total', 'actual_total_points', 'market_residual',
                'pred_market_residual', 'side', 'result', 'total_bin', 'wind_bin', 'temp_bin',
                'game_indoors_bool', 'precip_flag', 'snow_flag', 'home_conference', 'away_conference',
                'fbs_vs_fbs',
            ] if c in plays.columns]
            all_plays.append(plays[keep])
    if not all_plays:
        return pd.DataFrame(), pd.DataFrame(diag)
    return pd.concat(all_plays, ignore_index=True), pd.DataFrame(diag)


def recent_period(season: int) -> str:
    if season >= 2022:
        return '2022-2025'
    if season >= 2019:
        return '2019-2021'
    return '2016-2018'


def write_validation_summary(by_season: pd.DataFrame, by_side: pd.DataFrame, by_total: pd.DataFrame, by_wind: pd.DataFrame, by_provider: pd.DataFrame, by_recent: pd.DataFrame) -> None:
    out = ensure_dir('outputs') / 'model_edge_validation_summary.md'
    lines = [
        '# Model Edge Validation Summary',
        '',
        'This report stress-tests the model bake-off edges by season, side, total range, weather bin, provider, and recent period.',
        '',
        '## Best overall rows by recent-period ROI',
        '',
    ]
    if by_recent.empty:
        lines.append('_No recent-period rows generated._')
    else:
        lines.append(by_recent.sort_values(['roi_per_1u', 'graded'], ascending=[False, False]).head(30).to_markdown(index=False))
    lines.extend(['', '## By season', ''])
    lines.append(by_season.sort_values(['model', 'threshold', 'season']).to_markdown(index=False) if not by_season.empty else '_No season rows._')
    lines.extend(['', '## By side', ''])
    lines.append(by_side.sort_values(['model', 'threshold', 'side']).to_markdown(index=False) if not by_side.empty else '_No side rows._')
    lines.extend(['', '## By total bin', ''])
    lines.append(by_total.sort_values(['model', 'threshold', 'total_bin']).to_markdown(index=False) if not by_total.empty else '_No total-bin rows._')
    lines.extend(['', '## By wind bin', ''])
    lines.append(by_wind.sort_values(['model', 'threshold', 'wind_bin']).to_markdown(index=False) if not by_wind.empty else '_No wind-bin rows._')
    lines.extend(['', '## By provider', ''])
    lines.append(by_provider.sort_values(['model', 'threshold', 'line_provider']).to_markdown(index=False) if not by_provider.empty else '_No provider rows._')
    lines.extend([
        '',
        '## Interpretation notes',
        '',
        '- A promising edge should be positive in recent seasons, not only early seasons.',
        '- A promising edge should not rely on one side, one provider, or one narrow bin unless that is the intended strategy.',
        '- Small samples can look excellent by chance; prioritize rows with enough graded plays and Wilson intervals above or near breakeven.',
        '- These results still use historical observations and historical market data, so live paper tracking remains required.',
    ])
    out.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    df = prep(read_df('data/processed/modeling_dataset.csv'))
    plays, diag = collect_model_plays(df)
    write_df(plays, 'outputs/model_edge_plays.csv')
    write_df(diag, 'outputs/model_edge_validation_diagnostics.csv')
    if plays.empty:
        print('No model edge plays generated.')
        return
    plays['recent_period'] = plays['season'].astype(int).map(recent_period)
    by_season = grade_frame(plays, ['model', 'threshold', 'season'])
    by_side = grade_frame(plays, ['model', 'threshold', 'side'])
    by_recent = grade_frame(plays, ['model', 'threshold', 'recent_period'])
    by_total = grade_frame(plays, ['model', 'threshold', 'total_bin']) if 'total_bin' in plays.columns else pd.DataFrame()
    by_wind = grade_frame(plays, ['model', 'threshold', 'wind_bin']) if 'wind_bin' in plays.columns else pd.DataFrame()
    by_temp = grade_frame(plays, ['model', 'threshold', 'temp_bin']) if 'temp_bin' in plays.columns else pd.DataFrame()
    by_provider = grade_frame(plays, ['model', 'threshold', 'line_provider']) if 'line_provider' in plays.columns else pd.DataFrame()
    write_df(by_season, 'outputs/model_edge_by_season.csv')
    write_df(by_side, 'outputs/model_edge_by_side.csv')
    write_df(by_recent, 'outputs/model_edge_by_recent_period.csv')
    write_df(by_total, 'outputs/model_edge_by_total_bin.csv')
    write_df(by_wind, 'outputs/model_edge_by_wind_bin.csv')
    write_df(by_temp, 'outputs/model_edge_by_temp_bin.csv')
    write_df(by_provider, 'outputs/model_edge_by_provider.csv')
    write_validation_summary(by_season, by_side, by_total, by_wind, by_provider, by_recent)
    print('Wrote model edge validation outputs')


if __name__ == '__main__':
    main()
