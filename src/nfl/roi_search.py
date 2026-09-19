from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..utils import ensure_dir, load_yaml, read_df, write_df
from .model_bakeoff import (
    BREAKEVEN,
    CONTEXT_CATS,
    CONTEXT_NUMS,
    WEATHER_CATS,
    WEATHER_NUMS,
    _available_categorical,
    _available_numeric,
    _prep,
)


REGRESSION_THRESHOLDS = [
    1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
]
CLASSIFIER_THRESHOLDS = [
    0.02, 0.04, 0.06, 0.08, 0.10,
    0.12, 0.14, 0.16, 0.18, 0.20,
]
SIDE_MODES = ['both', 'over', 'under']


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


def _deep_models(nums: list[str], cats: list[str]) -> dict[str, tuple[str, Pipeline]]:
    models: dict[str, tuple[str, Pipeline]] = {}

    for alpha in (10.0, 50.0):
        models[f'reg_ridge_a{int(alpha)}'] = (
            'regression',
            Pipeline([
                ('prep', _preprocessor(nums, cats)),
                ('model', Ridge(alpha=alpha)),
            ]),
        )

    for alpha, label in ((0.02, '002'), (0.08, '008')):
        models[f'reg_elastic_a{label}'] = (
            'regression',
            Pipeline([
                ('prep', _preprocessor(nums, cats)),
                ('model', ElasticNet(
                    alpha=alpha,
                    l1_ratio=0.15,
                    max_iter=20000,
                )),
            ]),
        )

    for leaf in (10, 25, 50):
        models[f'reg_rf_leaf{leaf}'] = (
            'regression',
            Pipeline([
                ('prep', _preprocessor(nums, cats)),
                ('model', RandomForestRegressor(
                    n_estimators=150,
                    min_samples_leaf=leaf,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]),
        )

    for leaf in (15, 35):
        models[f'reg_extra_leaf{leaf}'] = (
            'regression',
            Pipeline([
                ('prep', _preprocessor(nums, cats)),
                ('model', ExtraTreesRegressor(
                    n_estimators=150,
                    min_samples_leaf=leaf,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]),
        )

    for leaf in (20, 35, 60):
        models[f'reg_hgb_leaf{leaf}'] = (
            'regression',
            Pipeline([
                ('prep', _preprocessor(nums, cats)),
                ('model', HistGradientBoostingRegressor(
                    max_iter=180,
                    learning_rate=0.04,
                    l2_regularization=1.0,
                    max_leaf_nodes=15,
                    min_samples_leaf=leaf,
                    random_state=42,
                )),
            ]),
        )

    for c_value, label in ((0.1, '01'), (1.0, '1')):
        models[f'cls_logit_c{label}'] = (
            'classifier',
            Pipeline([
                ('prep', _preprocessor(nums, cats)),
                ('model', LogisticRegression(
                    C=c_value,
                    max_iter=5000,
                )),
            ]),
        )

    for leaf in (10, 25, 50):
        models[f'cls_rf_leaf{leaf}'] = (
            'classifier',
            Pipeline([
                ('prep', _preprocessor(nums, cats)),
                ('model', RandomForestClassifier(
                    n_estimators=150,
                    min_samples_leaf=leaf,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]),
        )

    for leaf in (15, 35):
        models[f'cls_extra_leaf{leaf}'] = (
            'classifier',
            Pipeline([
                ('prep', _preprocessor(nums, cats)),
                ('model', ExtraTreesClassifier(
                    n_estimators=150,
                    min_samples_leaf=leaf,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]),
        )

    for leaf in (20, 35, 60):
        models[f'cls_hgb_leaf{leaf}'] = (
            'classifier',
            Pipeline([
                ('prep', _preprocessor(nums, cats)),
                ('model', HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.04,
                    l2_regularization=1.0,
                    max_leaf_nodes=15,
                    min_samples_leaf=leaf,
                    random_state=42,
                )),
            ]),
        )

    return models


def _results(frame: pd.DataFrame) -> pd.Series:
    under = frame['signal'] < 0
    push = frame['market_residual'] == 0
    win = (
        (under & (frame['market_residual'] < 0))
        | (~under & (frame['market_residual'] > 0))
    )
    return pd.Series(
        np.where(push, 'push', np.where(win, 'win', 'loss')),
        index=frame.index,
    )


def _metrics(frame: pd.DataFrame) -> dict | None:
    if frame.empty:
        return None
    result = _results(frame)
    wins = int(result.eq('win').sum())
    losses = int(result.eq('loss').sum())
    pushes = int(result.eq('push').sum())
    graded = wins + losses
    units = wins * (100 / 110) - losses
    return {
        'games': int(len(frame)),
        'graded': graded,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / graded if graded else np.nan,
        'net_units_1u_each': units,
        'roi_per_1u': units / graded if graded else np.nan,
    }


def _apply_side(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    if side == 'over':
        return frame[frame['signal'] > 0]
    if side == 'under':
        return frame[frame['signal'] < 0]
    return frame


def _thresholds(family: str) -> list[float]:
    return (
        REGRESSION_THRESHOLDS
        if family == 'regression'
        else CLASSIFIER_THRESHOLDS
    )


def _bh_qvalues(p_values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    count = len(valid)
    if count == 0:
        return out
    running = 1.0
    adjusted: dict[int, float] = {}
    for rank in range(count, 0, -1):
        idx = valid.index[rank - 1]
        value = min(running, float(valid.loc[idx]) * count / rank)
        adjusted[idx] = value
        running = value
    for idx, value in adjusted.items():
        out.loc[idx] = value
    return out


def _wilson(wins: int, graded: int) -> tuple[float, float]:
    if graded <= 0:
        return np.nan, np.nan
    z = 1.96
    p = wins / graded
    denom = 1 + z * z / graded
    center = (p + z * z / (2 * graded)) / denom
    margin = z * np.sqrt(
        (p * (1 - p) + z * z / (4 * graded)) / graded
    ) / denom
    return center - margin, center + margin


def _season_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for season, group in frame.groupby('season'):
        metrics = _metrics(group)
        if metrics is None or metrics['graded'] <= 0:
            continue
        rows.append({'season': int(season), **metrics})
    return pd.DataFrame(rows)


def _summarize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for (family, model_name), group in pred.groupby(['family', 'model']):
        for threshold in _thresholds(str(family)):
            thresholded = group[group['signal'].abs() >= threshold]
            for side in SIDE_MODES:
                plays = _apply_side(thresholded, side)
                metrics = _metrics(plays)
                if metrics is None or metrics['graded'] <= 0:
                    continue

                by_season = _season_metrics(plays)
                positive_share = (
                    float((by_season['roi_per_1u'] > 0).mean())
                    if not by_season.empty else np.nan
                )
                median_season_roi = (
                    float(by_season['roi_per_1u'].median())
                    if not by_season.empty else np.nan
                )
                worst_season_roi = (
                    float(by_season['roi_per_1u'].min())
                    if not by_season.empty else np.nan
                )

                last_five = sorted(plays['season'].dropna().astype(int).unique())[-5:]
                recent = plays[plays['season'].isin(last_five)]
                recent_metrics = _metrics(recent)

                rows.append({
                    'family': family,
                    'model': model_name,
                    'threshold': float(threshold),
                    'threshold_units': (
                        'predicted_points'
                        if family == 'regression'
                        else 'probability_margin_from_50pct'
                    ),
                    'side': side,
                    **metrics,
                    'seasons_with_plays': int(len(by_season)),
                    'positive_roi_season_share': positive_share,
                    'median_season_roi': median_season_roi,
                    'worst_season_roi': worst_season_roi,
                    'recent_5_season_roi': (
                        recent_metrics['roi_per_1u']
                        if recent_metrics is not None else np.nan
                    ),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out['neighbor_mean_roi'] = np.nan
    out['neighbor_positive_fraction'] = np.nan

    groups = out.groupby(['family', 'model', 'side']).groups
    for _, indexes in groups.items():
        part = out.loc[indexes].sort_values('threshold')
        ordered = list(part.index)
        for position, idx in enumerate(ordered):
            window = ordered[
                max(0, position - 1):min(len(ordered), position + 2)
            ]
            values = out.loc[window, 'roi_per_1u']
            out.loc[idx, 'neighbor_mean_roi'] = float(values.mean())
            out.loc[idx, 'neighbor_positive_fraction'] = float((values > 0).mean())

    out['p_value_vs_minus110'] = out.apply(
        lambda row: binomtest(
            int(row['wins']),
            int(row['graded']),
            p=BREAKEVEN,
            alternative='greater',
        ).pvalue,
        axis=1,
    )
    out['q_value_bh'] = _bh_qvalues(out['p_value_vs_minus110'])

    intervals = out.apply(
        lambda row: _wilson(int(row['wins']), int(row['graded'])),
        axis=1,
    )
    out['wilson_low'] = [value[0] for value in intervals]
    out['wilson_high'] = [value[1] for value in intervals]

    volume_factor = np.sqrt(np.minimum(out['graded'], 500) / 500)
    out['sustainability_score'] = (
        out['roi_per_1u']
        * out['positive_roi_season_share']
        * volume_factor
        * (0.5 + 0.5 * out['neighbor_positive_fraction'])
    )

    out['passes_sustainability_gates'] = (
        (out['graded'] >= 100)
        & (out['seasons_with_plays'] >= 8)
        & (out['positive_roi_season_share'] >= 0.60)
        & (out['neighbor_mean_roi'] > 0)
        & (out['neighbor_positive_fraction'] >= (2 / 3))
        & (out['recent_5_season_roi'] > 0)
    )
    out['sustainable_10pct_plus'] = (
        out['passes_sustainability_gates']
        & (out['roi_per_1u'] >= 0.10)
    )
    return out


def _generate_predictions(
    df: pd.DataFrame,
    nums: list[str],
    cats: list[str],
    min_train: int,
    min_test: int,
) -> pd.DataFrame:
    work = _prep(df, nums, cats)
    parts: list[pd.DataFrame] = []

    for model_name, (family, model) in _deep_models(nums, cats).items():
        for season in sorted(work['season'].dropna().astype(int).unique()):
            train = work[work['season'] < season].copy()
            test = work[work['season'] == season].copy()
            if len(train) < min_train or len(test) < min_test:
                continue

            if family == 'classifier':
                train = train[train['market_residual'] != 0].copy()
                target = (train['market_residual'] > 0).astype(int)
                model.fit(train[nums + cats], target)
                signal = model.predict_proba(test[nums + cats])[:, 1] - 0.5
            else:
                model.fit(train[nums + cats], train['market_residual'])
                signal = model.predict(test[nums + cats])

            keep = [
                c for c in [
                    'game_id', 'season', 'week', 'gameday',
                    'away_team', 'home_team', 'closing_total',
                    'market_residual',
                ]
                if c in test.columns
            ]
            part = test[keep].copy()
            part['family'] = family
            part['model'] = model_name
            part['signal'] = signal
            parts.append(part)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _select_from_history(history: pd.DataFrame) -> pd.Series | None:
    summary = _summarize_predictions(history)
    if summary.empty:
        return None

    eligible = summary[
        (summary['graded'] >= 60)
        & (summary['seasons_with_plays'] >= 4)
        & (summary['positive_roi_season_share'] >= 0.60)
        & (summary['roi_per_1u'] > 0)
        & (summary['neighbor_mean_roi'] > 0)
        & (summary['neighbor_positive_fraction'] >= (2 / 3))
        & (summary['recent_5_season_roi'] > 0)
    ].copy()

    if eligible.empty:
        return None

    return eligible.sort_values(
        ['sustainability_score', 'roi_per_1u', 'graded'],
        ascending=[False, False, False],
    ).iloc[0]


def _nested_walkforward(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    play_parts: list[pd.DataFrame] = []

    seasons = sorted(pred['season'].dropna().astype(int).unique())
    for target_season in seasons:
        history = pred[pred['season'] < target_season].copy()
        if history['season'].nunique() < 4:
            continue

        selection = _select_from_history(history)
        if selection is None:
            rows.append({
                'season': target_season,
                'selected_model': 'NONE',
                'graded': 0,
            })
            continue

        target = pred[
            (pred['season'] == target_season)
            & (pred['family'] == selection['family'])
            & (pred['model'] == selection['model'])
            & (pred['signal'].abs() >= float(selection['threshold']))
        ].copy()
        target = _apply_side(target, str(selection['side']))
        metrics = _metrics(target)

        if metrics is None:
            metrics = {
                'games': 0, 'graded': 0, 'wins': 0, 'losses': 0,
                'pushes': 0, 'hit_rate': np.nan,
                'net_units_1u_each': 0.0, 'roi_per_1u': np.nan,
            }

        rows.append({
            'season': target_season,
            'selected_family': selection['family'],
            'selected_model': selection['model'],
            'selected_threshold': float(selection['threshold']),
            'selected_threshold_units': selection['threshold_units'],
            'selected_side': selection['side'],
            'historical_roi_at_selection': float(selection['roi_per_1u']),
            'historical_positive_season_share': float(
                selection['positive_roi_season_share']
            ),
            'historical_graded_at_selection': int(selection['graded']),
            **metrics,
        })

        if not target.empty:
            result = _results(target)
            target = target.assign(
                selected_threshold=float(selection['threshold']),
                selected_side=str(selection['side']),
                result=result,
            )
            play_parts.append(target)

    return (
        pd.DataFrame(rows),
        pd.concat(play_parts, ignore_index=True)
        if play_parts else pd.DataFrame(),
    )


def main() -> None:
    settings = load_yaml('config/nfl_settings.yml')
    model_cfg = settings['modeling']
    df = read_df(settings['data']['processed_path']).copy()
    df = df.dropna(
        subset=['season', 'closing_total', 'actual_total_points', 'market_residual']
    )

    nums = _available_numeric(df, CONTEXT_NUMS + WEATHER_NUMS)
    cats = _available_categorical(df, CONTEXT_CATS + WEATHER_CATS)

    predictions = _generate_predictions(
        df,
        nums,
        cats,
        int(model_cfg.get('min_train_games', 1000)),
        int(model_cfg.get('min_test_games', 150)),
    )
    if predictions.empty:
        raise RuntimeError('Deep NFL ROI search produced no predictions.')

    summary = _summarize_predictions(predictions)
    nested, nested_plays = _nested_walkforward(predictions)

    write_df(summary, 'outputs/nfl/roi_deep_search_summary.csv')
    write_df(nested, 'outputs/nfl/roi_nested_walkforward.csv')
    if not nested_plays.empty:
        write_df(nested_plays, 'outputs/nfl/roi_nested_selected_plays.csv')

    sustainable = summary[
        summary['passes_sustainability_gates'].fillna(False)
    ].sort_values(
        ['sustainability_score', 'roi_per_1u', 'graded'],
        ascending=[False, False, False],
    )
    stretch = sustainable[sustainable['roi_per_1u'] >= 0.10].copy()

    nested_valid = nested[pd.to_numeric(
        nested.get('graded', pd.Series(dtype=float)),
        errors='coerce',
    ).fillna(0) > 0].copy()

    if nested_valid.empty:
        nested_record = 'No nested selections produced graded plays.'
        nested_roi = np.nan
        nested_positive = 0
        nested_seasons = 0
    else:
        wins = int(nested_valid['wins'].sum())
        losses = int(nested_valid['losses'].sum())
        pushes = int(nested_valid['pushes'].sum())
        graded = wins + losses
        units = wins * (100 / 110) - losses
        nested_roi = units / graded if graded else np.nan
        nested_positive = int((nested_valid['roi_per_1u'] > 0).sum())
        nested_seasons = int(len(nested_valid))
        nested_record = (
            f'{wins}-{losses}-{pushes} across {graded} graded plays; '
            f'ROI {nested_roi:.2%}; positive ROI in '
            f'{nested_positive}/{nested_seasons} seasons.'
        )

    top_cols = [
        'family', 'model', 'threshold', 'threshold_units', 'side',
        'graded', 'wins', 'losses', 'pushes', 'hit_rate',
        'roi_per_1u', 'seasons_with_plays',
        'positive_roi_season_share', 'median_season_roi',
        'recent_5_season_roi', 'neighbor_mean_roi',
        'p_value_vs_minus110', 'q_value_bh',
        'sustainability_score',
    ]

    lines = [
        '# NFL Deep ROI / Sustainability Search',
        '',
        'This search intentionally targets high ROI without allowing the 10% goal to become the selection rule. Regression and direct over/under classifiers are tested chronologically, with side-specific screens and dense thresholds. Every season-level prediction is out of sample because each model is fit only on earlier seasons.',
        '',
        '## Search space',
        '',
        f'- Model/hyperparameter variants: {summary["model"].nunique()}',
        f'- Strategy rows after threshold and side expansion: {len(summary):,}',
        '- Regression threshold grid: 1.0 to 8.0 predicted points.',
        '- Classifier threshold grid: 2 to 20 percentage points away from 50%.',
        '- Side modes: both, OVER only, UNDER only.',
        '',
        '## Sustainability gates',
        '',
        '- At least 100 graded plays.',
        '- Qualifying plays in at least 8 test seasons.',
        '- Positive flat-stake ROI in at least 60% of seasons with plays.',
        '- Positive mean ROI across the candidate threshold and its immediate neighbors.',
        '- At least two-thirds of that local threshold neighborhood must be profitable.',
        '- Positive aggregate ROI across the most recent five seasons in the available test sample.',
        '',
        '## Best sustainable candidates',
        '',
        (
            sustainable[top_cols].head(30).to_markdown(index=False)
            if not sustainable.empty
            else '_No strategy passed every sustainability gate._'
        ),
        '',
        '## Sustainable candidates at or above 10% ROI',
        '',
        (
            stretch[top_cols].head(30).to_markdown(index=False)
            if not stretch.empty
            else '_No 10%+ ROI strategy passed every sustainability gate._'
        ),
        '',
        '## Nested walk-forward strategy-selection test',
        '',
        'For each outer test season, the selection process can only see earlier out-of-sample seasons. It chooses the highest consistency-adjusted ROI strategy that passes looser history-only eligibility gates, then grades that frozen choice on the next season.',
        '',
        f'- Nested result: {nested_record}',
        '',
        nested.to_markdown(index=False) if not nested.empty else '_No nested rows._',
        '',
        '## Interpretation',
        '',
        '- A 10%+ row is a stretch candidate, not a production claim. It must also survive forecast-realistic weather reconstruction and prospective tracking.',
        '- The Benjamini-Hochberg q-value explicitly reflects the large model/threshold search; attractive raw p-values should not be read in isolation.',
        '- The closing market remains the benchmark. High selective ROI matters more than small changes in overall prediction MAE for this project, but the selection process must remain chronological and leakage-free.',
        '- No NFL strategy should be promoted to production solely because it is the highest row in this development table.',
    ]

    out = ensure_dir('outputs/nfl') / 'roi_deep_search.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote deep NFL ROI search outputs under {out.parent}')


if __name__ == '__main__':
    main()
