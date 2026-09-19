from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..utils import ensure_dir, load_yaml, read_df, write_df


BREAKEVEN = 110 / 210
WEATHER_NUMS = [
    'temperature_f', 'wind_mph', 'wind_excess_10', 'wind_excess_15',
    'cold_excess_below_32', 'extreme_cold_below_20', 'wind_10plus',
    'wind_15plus', 'cold_32_or_less', 'cold_windy',
]
CONTEXT_NUMS = ['closing_total', 'spread_line', 'week', 'home_rest', 'away_rest']
CONTEXT_CATS = ['div_game', 'location', 'weekday', 'month']
WEATHER_CATS = ['roof', 'surface', 'wind_bin', 'temp_bin', 'total_bin']


def _available_numeric(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        c for c in columns
        if c in df.columns and pd.to_numeric(df[c], errors='coerce').notna().any()
    ]


def _available_categorical(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns and df[c].notna().any()]


def _prep(df: pd.DataFrame, nums: list[str], cats: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in nums:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    for c in cats:
        out[c] = out[c].astype(str).fillna('missing')
    return out


def _preprocessor(nums: list[str], cats: list[str]) -> ColumnTransformer:
    transformers = []
    if nums:
        transformers.append((
            'num',
            Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
            ]),
            nums,
        ))
    if cats:
        transformers.append((
            'cat',
            Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]),
            cats,
        ))
    return ColumnTransformer(transformers)


def _models(nums: list[str], cats: list[str]) -> dict[str, Pipeline]:
    return {
        'ridge': Pipeline([
            ('prep', _preprocessor(nums, cats)),
            ('model', Ridge(alpha=25.0)),
        ]),
        'elastic_net': Pipeline([
            ('prep', _preprocessor(nums, cats)),
            ('model', ElasticNet(alpha=0.05, l1_ratio=0.15, max_iter=20000)),
        ]),
        'random_forest': Pipeline([
            ('prep', _preprocessor(nums, cats)),
            (
                'model',
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=25,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),
        'extra_trees': Pipeline([
            ('prep', _preprocessor(nums, cats)),
            (
                'model',
                ExtraTreesRegressor(
                    n_estimators=300,
                    min_samples_leaf=25,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),
        'hist_gradient_boosting': Pipeline([
            ('prep', _preprocessor(nums, cats)),
            (
                'model',
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.035,
                    l2_regularization=1.0,
                    max_leaf_nodes=15,
                    min_samples_leaf=35,
                    random_state=42,
                ),
            ),
        ]),
    }


def _settle(df: pd.DataFrame) -> pd.Series:
    side_under = df['pred_market_residual'] < 0
    actual_under = df['market_residual'] < 0
    actual_over = df['market_residual'] > 0
    push = df['market_residual'] == 0
    win = (side_under & actual_under) | (~side_under & actual_over)
    return pd.Series(
        np.where(push, 'push', np.where(win, 'win', 'loss')),
        index=df.index,
    )


def _strategy_rows(
    pred: pd.DataFrame,
    feature_set: str,
    model_name: str,
    thresholds: list[float],
) -> list[dict]:
    rows: list[dict] = []
    for threshold in thresholds:
        plays = pred[pred['pred_market_residual'].abs() >= threshold].copy()
        if plays.empty:
            rows.append({
                'feature_set': feature_set,
                'model': model_name,
                'threshold': threshold,
                'games': 0,
            })
            continue
        result = _settle(plays)
        wins = int(result.eq('win').sum())
        losses = int(result.eq('loss').sum())
        n = wins + losses
        units = float(
            result.map({'win': 100 / 110, 'loss': -1.0, 'push': 0.0}).sum()
        )
        rows.append({
            'feature_set': feature_set,
            'model': model_name,
            'threshold': threshold,
            'games': int(len(plays)),
            'graded': n,
            'wins': wins,
            'losses': losses,
            'pushes': int(result.eq('push').sum()),
            'hit_rate': wins / n if n else np.nan,
            'hit_rate_minus_breakeven': wins / n - BREAKEVEN if n else np.nan,
            'net_units_1u_each': units,
            'roi_per_1u': units / n if n else np.nan,
            'avg_abs_pred_edge': float(plays['pred_market_residual'].abs().mean()),
            'test_seasons': f"{int(pred['season'].min())}-{int(pred['season'].max())}",
        })
    return rows


def _run_feature_set(
    df: pd.DataFrame,
    feature_set: str,
    nums: list[str],
    cats: list[str],
    min_train: int,
    min_test: int,
    thresholds: list[float],
) -> tuple[list[dict], list[dict]]:
    work = _prep(df, nums, cats)
    strategy_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    for model_name, model in _models(nums, cats).items():
        pred_parts: list[pd.DataFrame] = []
        for season in sorted(work['season'].dropna().astype(int).unique()):
            train = work[work['season'] < season].copy()
            test = work[work['season'] == season].copy()
            if len(train) < min_train or len(test) < min_test:
                continue
            model.fit(train[nums + cats], train['market_residual'])
            prediction = model.predict(test[nums + cats])
            test = test.assign(pred_market_residual=prediction)
            pred_parts.append(test)
            zero_mae = mean_absolute_error(
                test['market_residual'],
                np.zeros(len(test)),
            )
            model_mae = mean_absolute_error(test['market_residual'], prediction)
            diagnostic_rows.append({
                'feature_set': feature_set,
                'model': model_name,
                'test_season': int(season),
                'train_games': int(len(train)),
                'test_games': int(len(test)),
                'model_mae': model_mae,
                'zero_residual_baseline_mae': zero_mae,
                'mae_improvement_vs_market': zero_mae - model_mae,
                'avg_pred_residual': float(np.mean(prediction)),
                'avg_actual_residual': float(test['market_residual'].mean()),
            })
        if pred_parts:
            pred = pd.concat(pred_parts, ignore_index=True)
            strategy_rows.extend(
                _strategy_rows(pred, feature_set, model_name, thresholds)
            )

    return strategy_rows, diagnostic_rows


def main() -> None:
    settings = load_yaml('config/nfl_settings.yml')
    model_cfg = settings['modeling']
    df = read_df(settings['data']['processed_path']).copy()
    df = df.dropna(
        subset=['season', 'closing_total', 'actual_total_points', 'market_residual']
    )

    context_nums = _available_numeric(df, CONTEXT_NUMS)
    context_cats = _available_categorical(df, CONTEXT_CATS)
    weather_nums = _available_numeric(df, CONTEXT_NUMS + WEATHER_NUMS)
    weather_cats = _available_categorical(df, CONTEXT_CATS + WEATHER_CATS)

    min_train = int(model_cfg.get('min_train_games', 1000))
    min_test = int(model_cfg.get('min_test_games', 150))
    thresholds = [
        float(v)
        for v in model_cfg.get(
            'edge_thresholds',
            [1.0, 1.5, 2.0, 2.5, 3.0],
        )
    ]

    all_strategy: list[dict] = []
    all_diag: list[dict] = []
    for feature_set, nums, cats in [
        ('context_only', context_nums, context_cats),
        ('context_plus_weather', weather_nums, weather_cats),
    ]:
        strategy, diag = _run_feature_set(
            df,
            feature_set,
            nums,
            cats,
            min_train,
            min_test,
            thresholds,
        )
        all_strategy.extend(strategy)
        all_diag.extend(diag)

    strategy_df = pd.DataFrame(all_strategy)
    diag_df = pd.DataFrame(all_diag)
    write_df(strategy_df, 'outputs/nfl/model_bakeoff_summary.csv')
    write_df(diag_df, 'outputs/nfl/model_bakeoff_diagnostics.csv')

    lines = [
        '# NFL Residual Model Bake-off',
        '',
        'Every test season is predicted using only prior seasons. The target is the game total residual versus the closing market.',
        '',
        'Two feature sets are run deliberately: `context_only` and `context_plus_weather`. The incremental comparison is as important as the model-family ranking because it tests whether weather adds predictive information after the market total and basic game context are already known.',
        '',
        '## Strategy screens',
        '',
        (
            strategy_df.sort_values(
                ['roi_per_1u', 'graded'],
                ascending=[False, False],
            ).to_markdown(index=False)
            if not strategy_df.empty
            else '_No strategy rows produced._'
        ),
        '',
        '## Season-by-season diagnostics',
        '',
        diag_df.to_markdown(index=False) if not diag_df.empty else '_No diagnostics produced._',
        '',
        '## Selection rules for the next phase',
        '',
        '- Do not choose a model from a single best ROI row. Prefer lower out-of-sample MAE, stable direction across seasons, and threshold robustness.',
        '- Weather should earn its place by improving chronological diagnostics or producing a stable selective residual signal compared with the context-only version.',
        '- Because historical `temp` and `wind` are observed conditions, any apparent edge remains provisional until weather is reconstructed from forecasts available before kickoff.',
        '- NFL production remains disabled until those validation steps are complete.',
    ]
    out = ensure_dir('outputs/nfl') / 'model_bakeoff_summary.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote NFL model bake-off outputs under {out.parent}')


if __name__ == '__main__':
    main()
