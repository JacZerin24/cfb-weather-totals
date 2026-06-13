from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .deep_research import BREAKEVEN, prep, settle, summarize
from .utils import ensure_dir, read_df, write_df


EDGE_THRESHOLDS = [1.5, 2.5, 3.5, 5.0]
PROB_THRESHOLDS = [0.535, 0.55, 0.565, 0.58]


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    base_nums = [
        'closing_total', 'wind_mph', 'temperature_f', 'humidity', 'precipitation',
        'snowfall', 'dewpoint_f', 'pressure', 'home_pregame_elo', 'away_pregame_elo',
    ]
    prior_nums = [c for c in df.columns if c.startswith(('home_prior_', 'away_prior_'))]
    nums = [c for c in base_nums + prior_nums if c in df.columns]
    cats = [
        'game_indoors_bool', 'neutral_site', 'conference_game', 'line_provider',
        'wind_bin', 'temp_bin', 'total_bin', 'fbs_vs_fbs', 'home_conference', 'away_conference',
    ]
    cats = [c for c in cats if c in df.columns]
    return nums, cats


def prep_features(df: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cats:
        out[c] = out[c].astype(str).fillna('missing')
    return out


def preprocessor(nums: list[str], cats: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), nums),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cats),
    ])


def reg_models(nums: list[str], cats: list[str]) -> dict[str, Pipeline]:
    return {
        'ridge': Pipeline([('prep', preprocessor(nums, cats)), ('model', Ridge(alpha=25.0))]),
        'elastic_net': Pipeline([('prep', preprocessor(nums, cats)), ('model', ElasticNet(alpha=0.05, l1_ratio=0.15, max_iter=20000))]),
        'random_forest': Pipeline([('prep', preprocessor(nums, cats)), ('model', RandomForestRegressor(n_estimators=250, min_samples_leaf=25, random_state=42, n_jobs=-1))]),
        'extra_trees': Pipeline([('prep', preprocessor(nums, cats)), ('model', ExtraTreesRegressor(n_estimators=250, min_samples_leaf=25, random_state=42, n_jobs=-1))]),
        'hist_gradient_boosting': Pipeline([('prep', preprocessor(nums, cats)), ('model', HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, l2_regularization=0.5, min_samples_leaf=35, random_state=42))]),
    }


def class_models(nums: list[str], cats: list[str]) -> dict[str, Pipeline]:
    return {
        'logistic_over_prob': Pipeline([('prep', preprocessor(nums, cats)), ('model', LogisticRegression(max_iter=20000, C=0.25, class_weight='balanced'))]),
    }


def grade_regression_predictions(pred_df: pd.DataFrame, model_name: str) -> list[dict]:
    rows = []
    for th in EDGE_THRESHOLDS:
        plays = pred_df[pred_df['pred_market_residual'].abs() >= th].copy()
        if plays.empty:
            rows.append({'model': model_name, 'model_type': 'regression', 'threshold': th, 'games': 0})
            continue
        plays['side'] = np.where(plays['pred_market_residual'] > 0, 'over', 'under')
        row = summarize(model_name, 'model', plays, settle(plays['actual_total_points'], plays['closing_total'], plays['side']))
        row.update({
            'model': model_name,
            'model_type': 'regression',
            'threshold': th,
            'avg_abs_model_edge': float(plays['pred_market_residual'].abs().mean()),
            'test_seasons': f"{int(pred_df['season'].min())}-{int(pred_df['season'].max())}",
        })
        rows.append(row)
    return rows


def grade_classifier_predictions(pred_df: pd.DataFrame, model_name: str) -> list[dict]:
    rows = []
    pred_df = pred_df.copy()
    pred_df['prob_edge'] = (pred_df['pred_over_prob'] - 0.5).abs()
    for th in PROB_THRESHOLDS:
        plays = pred_df[(pred_df['pred_over_prob'] >= th) | (pred_df['pred_over_prob'] <= 1 - th)].copy()
        if plays.empty:
            rows.append({'model': model_name, 'model_type': 'classifier', 'threshold': th, 'games': 0})
            continue
        plays['side'] = np.where(plays['pred_over_prob'] >= 0.5, 'over', 'under')
        row = summarize(model_name, 'model', plays, settle(plays['actual_total_points'], plays['closing_total'], plays['side']))
        row.update({
            'model': model_name,
            'model_type': 'classifier',
            'threshold': th,
            'avg_abs_model_edge': float(plays['prob_edge'].mean()),
            'test_seasons': f"{int(pred_df['season'].min())}-{int(pred_df['season'].max())}",
        })
        rows.append(row)
    return rows


def run_regression_models(df: pd.DataFrame, nums: list[str], cats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diag = []
    models = reg_models(nums, cats)
    for model_name, model in models.items():
        pred_parts = []
        for season in sorted(df['season'].dropna().astype(int).unique()):
            train = df[df['season'] < season].copy()
            test = df[df['season'] == season].copy()
            if len(train) < 1000 or len(test) < 100:
                continue
            model.fit(train[nums + cats], train['market_residual'])
            pred = model.predict(test[nums + cats])
            test = test.assign(pred_market_residual=pred)
            pred_parts.append(test)
            diag.append({
                'model': model_name,
                'model_type': 'regression',
                'test_season': season,
                'train_games': len(train),
                'test_games': len(test),
                'model_mae': mean_absolute_error(test['market_residual'], pred),
                'zero_residual_baseline_mae': mean_absolute_error(test['market_residual'], np.zeros(len(test))),
                'avg_pred_residual': float(np.mean(pred)),
                'avg_actual_residual': float(test['market_residual'].mean()),
            })
        if pred_parts:
            rows.extend(grade_regression_predictions(pd.concat(pred_parts, ignore_index=True), model_name))
    return pd.DataFrame(rows), pd.DataFrame(diag)


def run_classifier_models(df: pd.DataFrame, nums: list[str], cats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    no_push = df[df['actual_total_points'] != df['closing_total']].copy()
    no_push['over_target'] = (no_push['actual_total_points'] > no_push['closing_total']).astype(int)
    rows = []
    diag = []
    models = class_models(nums, cats)
    for model_name, model in models.items():
        pred_parts = []
        for season in sorted(no_push['season'].dropna().astype(int).unique()):
            train = no_push[no_push['season'] < season].copy()
            test = no_push[no_push['season'] == season].copy()
            if len(train) < 1000 or len(test) < 100 or train['over_target'].nunique() < 2:
                continue
            model.fit(train[nums + cats], train['over_target'])
            prob = model.predict_proba(test[nums + cats])[:, 1]
            test = test.assign(pred_over_prob=prob)
            pred_parts.append(test)
            diag.append({
                'model': model_name,
                'model_type': 'classifier',
                'test_season': season,
                'train_games': len(train),
                'test_games': len(test),
                'log_loss': log_loss(test['over_target'], np.clip(prob, 0.001, 0.999)),
                'base_over_rate': float(train['over_target'].mean()),
                'test_over_rate': float(test['over_target'].mean()),
                'avg_pred_over_prob': float(np.mean(prob)),
            })
        if pred_parts:
            rows.extend(grade_classifier_predictions(pd.concat(pred_parts, ignore_index=True), model_name))
    return pd.DataFrame(rows), pd.DataFrame(diag)


def write_summary(summary: pd.DataFrame, diag: pd.DataFrame) -> None:
    out = ensure_dir('outputs') / 'model_bakeoff_summary.md'
    lines = [
        '# Model Bake-off Summary',
        '',
        'All models are tested walk-forward. Each test season is predicted using only prior seasons.',
        '',
        '## Strategy results',
        '',
    ]
    if summary.empty:
        lines.append('_No model results produced._')
    else:
        view = summary.sort_values(['roi_per_1u', 'graded'], ascending=[False, False]).copy()
        lines.append(view.to_markdown(index=False))
    lines.extend(['', '## Diagnostics', ''])
    lines.append(diag.head(80).to_markdown(index=False) if not diag.empty else '_No diagnostics produced._')
    lines.extend([
        '',
        '## Interpretation guardrails',
        '',
        '- A model should not be considered usable unless it beats the -110 breakeven rate out of sample.',
        '- Prefer models that perform across several thresholds rather than one narrow setting.',
        '- A higher hit rate with very few plays needs paper tracking before trust.',
        '- This is still research, not live betting advice.',
    ])
    out.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    df = prep(read_df('data/processed/modeling_dataset.csv'))
    nums, cats = feature_lists(df)
    df = prep_features(df, cats)
    reg_summary, reg_diag = run_regression_models(df, nums, cats)
    cls_summary, cls_diag = run_classifier_models(df, nums, cats)
    summary = pd.concat([reg_summary, cls_summary], ignore_index=True)
    diag = pd.concat([reg_diag, cls_diag], ignore_index=True)
    write_df(summary, 'outputs/model_bakeoff_summary.csv')
    write_df(diag, 'outputs/model_bakeoff_diagnostics.csv')
    write_summary(summary, diag)
    print('Wrote model bake-off outputs')


if __name__ == '__main__':
    main()
