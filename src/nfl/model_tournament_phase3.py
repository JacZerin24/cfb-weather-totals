from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)
from sklearn.svm import SVC, SVR

from ..utils import ensure_dir, read_df, write_df
from .forecast_native_bakeoff import (
    CONTEXT_CATS,
    CONTEXT_NUMS,
    _available_categorical,
    _available_numeric,
    _feature_columns,
    _weather_features,
)
from .model_bakeoff import BREAKEVEN
from .roi_search import _deep_models as _v1_deep_models


DATA_PATH = 'data/nfl/processed/model_tournament_dataset.csv'
OUTER_SEASONS = (2021, 2022, 2023, 2024, 2025)
LEAD = 24
RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20261004
MIN_TRAIN = 700

RULES = {
    'qualifies_3pt_60pct': (3.0, 0.60),
    'strong_4pt_60pct': (4.0, 0.60),
}

COMPACT_STACK_REG = [
    'ridge',
    'spline_ridge',
    'random_forest',
    'extra_trees',
    'hist_gb',
    'xgboost',
    'lightgbm',
    'catboost',
    'mlp',
]
COMPACT_STACK_CLS = [
    'logit',
    'spline_logit',
    'random_forest',
    'extra_trees',
    'hist_gb',
    'xgboost',
    'lightgbm',
    'catboost',
    'mlp',
]

REG_FAMILY = {
    'linear': 'linear_spline',
    'ridge': 'linear_spline',
    'elastic_net': 'linear_spline',
    'bayesian_ridge': 'linear_spline',
    'huber': 'linear_spline',
    'spline_ridge': 'linear_spline',
    'random_forest': 'tree_bagging',
    'rf_v1': 'tree_bagging',
    'extra_trees': 'tree_bagging',
    'hist_gb': 'boosting',
    'gradient_boosting': 'boosting',
    'adaboost': 'boosting',
    'xgboost': 'boosting',
    'lightgbm': 'boosting',
    'catboost': 'boosting',
    'svr_rbf': 'nonlinear_neural',
    'mlp': 'nonlinear_neural',
    'dynamic_direct': 'dynamic',
    'market_zero': 'benchmark',
}
CLS_FAMILY = {
    'logit': 'linear_spline',
    'spline_logit': 'linear_spline',
    'random_forest': 'tree_bagging',
    'rf_v1': 'tree_bagging',
    'extra_trees': 'tree_bagging',
    'hist_gb': 'boosting',
    'gradient_boosting': 'boosting',
    'xgboost': 'boosting',
    'lightgbm': 'boosting',
    'catboost': 'boosting',
    'svc_rbf': 'nonlinear_neural',
    'mlp': 'nonlinear_neural',
    'dynamic_direct': 'dynamic',
    'base_rate': 'benchmark',
}

PAIR_MAP = {
    'v1_exact': ('rf_v1', 'rf_v1'),
    'linear_ridge_logit': ('ridge', 'logit'),
    'spline': ('spline_ridge', 'spline_logit'),
    'random_forest': ('random_forest', 'random_forest'),
    'extra_trees': ('extra_trees', 'extra_trees'),
    'hist_gb': ('hist_gb', 'hist_gb'),
    'gradient_boosting': ('gradient_boosting', 'gradient_boosting'),
    'mlp': ('mlp', 'mlp'),
    'xgboost': ('xgboost', 'xgboost'),
    'lightgbm': ('lightgbm', 'lightgbm'),
    'catboost': ('catboost', 'catboost'),
    'dynamic_direct': ('dynamic_direct', 'dynamic_direct'),
}

COMPLEXITY_RANK = {
    'market_zero': 0,
    'base_rate': 0,
    'linear': 1,
    'ridge': 2,
    'logit': 2,
    'elastic_net': 3,
    'bayesian_ridge': 3,
    'huber': 4,
    'spline_ridge': 5,
    'spline_logit': 5,
    'random_forest': 6,
    'rf_v1': 6,
    'extra_trees': 7,
    'hist_gb': 8,
    'gradient_boosting': 8,
    'adaboost': 8,
    'xgboost': 9,
    'lightgbm': 9,
    'catboost': 9,
    'svr_rbf': 10,
    'svc_rbf': 10,
    'mlp': 11,
    'dynamic_direct': 4,
}


def _dense_preprocessor(
    nums: list[str],
    cats: list[str],
    *,
    spline: bool = False,
) -> ColumnTransformer:
    transformers = []
    if nums:
        num_steps: list[tuple[str, object]] = [
            ('imp', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]
        if spline:
            num_steps.append(
                (
                    'spline',
                    SplineTransformer(
                        n_knots=4,
                        degree=3,
                        include_bias=False,
                    ),
                )
            )
        transformers.append(('num', Pipeline(num_steps), nums))
    if cats:
        transformers.append(
            (
                'cat',
                Pipeline(
                    [
                        ('imp', SimpleImputer(strategy='most_frequent')),
                        (
                            'onehot',
                            OneHotEncoder(
                                handle_unknown='ignore',
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                cats,
            )
        )
    return ColumnTransformer(
        transformers,
        sparse_threshold=0.0,
    )


def _pipe(
    nums: list[str],
    cats: list[str],
    model: object,
    *,
    spline: bool = False,
) -> Pipeline:
    return Pipeline(
        [
            ('prep', _dense_preprocessor(nums, cats, spline=spline)),
            ('model', model),
        ]
    )


def _optional_models() -> tuple[dict[str, object], dict[str, object], list[str]]:
    regressors: dict[str, object] = {}
    classifiers: dict[str, object] = {}
    unavailable: list[str] = []

    try:
        from xgboost import XGBClassifier, XGBRegressor

        regressors['xgboost'] = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbosity=0,
        )
        classifiers['xgboost'] = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbosity=0,
        )
    except Exception as exc:
        unavailable.append(f'xgboost: {exc}')

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        regressors['lightgbm'] = LGBMRegressor(
            n_estimators=250,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=35,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbosity=-1,
        )
        classifiers['lightgbm'] = LGBMClassifier(
            n_estimators=250,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=35,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbosity=-1,
        )
    except Exception as exc:
        unavailable.append(f'lightgbm: {exc}')

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        regressors['catboost'] = CatBoostRegressor(
            iterations=250,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            loss_function='MAE',
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=2,
        )
        classifiers['catboost'] = CatBoostClassifier(
            iterations=250,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            loss_function='Logloss',
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=2,
        )
    except Exception as exc:
        unavailable.append(f'catboost: {exc}')

    return regressors, classifiers, unavailable


def _regressors(
    nums: list[str],
    cats: list[str],
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        'linear': _pipe(
            nums,
            cats,
            LinearRegression(),
        ),
        'ridge': _pipe(
            nums,
            cats,
            Ridge(alpha=25.0),
        ),
        'elastic_net': _pipe(
            nums,
            cats,
            ElasticNet(
                alpha=0.05,
                l1_ratio=0.15,
                max_iter=20000,
            ),
        ),
        'bayesian_ridge': _pipe(
            nums,
            cats,
            BayesianRidge(),
        ),
        'huber': _pipe(
            nums,
            cats,
            HuberRegressor(
                epsilon=1.35,
                alpha=0.0001,
                max_iter=1000,
            ),
        ),
        'spline_ridge': _pipe(
            nums,
            cats,
            Ridge(alpha=25.0),
            spline=True,
        ),
        'rf_v1': _v1_deep_models(nums, cats)['reg_rf_leaf25'][1],
        'random_forest': _pipe(
            nums,
            cats,
            RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=25,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
        ),
        'extra_trees': _pipe(
            nums,
            cats,
            ExtraTreesRegressor(
                n_estimators=250,
                min_samples_leaf=25,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
        ),
        'hist_gb': _pipe(
            nums,
            cats,
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.035,
                l2_regularization=1.0,
                max_leaf_nodes=15,
                min_samples_leaf=35,
                random_state=RANDOM_STATE,
            ),
        ),
        'gradient_boosting': _pipe(
            nums,
            cats,
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
                loss='huber',
            ),
        ),
        'adaboost': _pipe(
            nums,
            cats,
            AdaBoostRegressor(
                n_estimators=150,
                learning_rate=0.03,
                random_state=RANDOM_STATE,
                loss='linear',
            ),
        ),
        'svr_rbf': _pipe(
            nums,
            cats,
            SVR(
                C=1.0,
                gamma='scale',
                epsilon=0.1,
            ),
        ),
        'mlp': _pipe(
            nums,
            cats,
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=RANDOM_STATE,
            ),
        ),
    }

    opt_reg, _, _ = _optional_models()
    for name, model in opt_reg.items():
        models[name] = _pipe(nums, cats, model)
    return models


def _classifiers(
    nums: list[str],
    cats: list[str],
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        'logit': _pipe(
            nums,
            cats,
            LogisticRegression(
                C=1.0,
                max_iter=5000,
            ),
        ),
        'spline_logit': _pipe(
            nums,
            cats,
            LogisticRegression(
                C=1.0,
                max_iter=5000,
            ),
            spline=True,
        ),
        'rf_v1': _v1_deep_models(nums, cats)['cls_rf_leaf25'][1],
        'random_forest': _pipe(
            nums,
            cats,
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=25,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
        ),
        'extra_trees': _pipe(
            nums,
            cats,
            ExtraTreesClassifier(
                n_estimators=250,
                min_samples_leaf=25,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
        ),
        'hist_gb': _pipe(
            nums,
            cats,
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.035,
                l2_regularization=1.0,
                max_leaf_nodes=15,
                min_samples_leaf=35,
                random_state=RANDOM_STATE,
            ),
        ),
        'gradient_boosting': _pipe(
            nums,
            cats,
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            ),
        ),
        'svc_rbf': _pipe(
            nums,
            cats,
            SVC(
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=RANDOM_STATE,
            ),
        ),
        'mlp': _pipe(
            nums,
            cats,
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=RANDOM_STATE,
            ),
        ),
    }

    _, opt_cls, _ = _optional_models()
    for name, model in opt_cls.items():
        models[name] = _pipe(nums, cats, model)
    return models


def _feature_sets(
    frame: pd.DataFrame,
) -> dict[str, tuple[list[str], list[str]]]:
    baseline_nums, baseline_cats = _feature_columns(frame, LEAD, True)

    team_cols = sorted(
        c for c in frame.columns
        if (
            c.startswith('team_')
            or c.startswith('matchup_')
        )
        and c != 'team_context_eligible'
        and pd.to_numeric(frame[c], errors='coerce').notna().any()
    )
    dynamic_cols = sorted(
        c for c in frame.columns
        if c.startswith('dyn_')
        and pd.to_numeric(frame[c], errors='coerce').notna().any()
    )

    return {
        'v1_weather': (
            list(dict.fromkeys(baseline_nums)),
            baseline_cats,
        ),
        'v1_plus_team': (
            list(dict.fromkeys(baseline_nums + team_cols)),
            baseline_cats,
        ),
        'v1_plus_dynamic': (
            list(dict.fromkeys(baseline_nums + dynamic_cols)),
            baseline_cats,
        ),
        'all_context': (
            list(
                dict.fromkeys(
                    baseline_nums + team_cols + dynamic_cols
                )
            ),
            baseline_cats,
        ),
    }


def _prep_frame(
    df: pd.DataFrame,
    nums: list[str],
    cats: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for c in nums:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    for c in cats:
        out[c] = out[c].astype(str).fillna('missing')
    return out


def _meta_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c for c in [
            'game_id',
            'season',
            'week',
            'gameday',
            'away_team',
            'home_team',
            'closing_total',
            'actual_total_points',
            'market_residual',
        ]
        if c in frame.columns
    ]


def _dynamic_direct_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    edges = [
        c for c in [
            'dyn_latent_edge_k005',
            'dyn_latent_edge_k012',
        ]
        if c in train.columns and c in test.columns
    ]
    if not edges:
        raise RuntimeError('Dynamic direct edge columns are missing.')

    train_pred = train[edges].apply(
        pd.to_numeric, errors='coerce'
    ).mean(axis=1)
    test_pred = test[edges].apply(
        pd.to_numeric, errors='coerce'
    ).mean(axis=1)

    train_actual = pd.to_numeric(
        train['market_residual'], errors='coerce'
    )
    errors = train_actual - train_pred
    sigma = float(errors.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 10.0

    probability = norm.cdf(
        pd.to_numeric(test_pred, errors='coerce').fillna(0.0)
        / sigma
    )
    return (
        pd.to_numeric(test_pred, errors='coerce')
        .fillna(0.0)
        .to_numpy(),
        np.asarray(probability, dtype=float),
    )


def _fit_outer_predictions(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_sets = _feature_sets(frame)
    reg_parts: list[pd.DataFrame] = []
    cls_parts: list[pd.DataFrame] = []
    errors: list[dict] = []

    for season in OUTER_SEASONS:
        train = frame[frame['season'].lt(season)].copy()
        test = frame[frame['season'].eq(season)].copy()
        if len(train) < MIN_TRAIN or test.empty:
            continue

        base_meta = test[_meta_columns(test)].copy()

        market_zero = base_meta.copy()
        market_zero['feature_set'] = 'benchmark'
        market_zero['model'] = 'market_zero'
        market_zero['family'] = 'benchmark'
        market_zero['prediction'] = 0.0
        reg_parts.append(market_zero)

        nonpush_train = train[
            pd.to_numeric(train['market_residual'], errors='coerce').ne(0)
        ].copy()
        over_rate = float(
            pd.to_numeric(
                nonpush_train['market_residual'], errors='coerce'
            ).gt(0).mean()
        )
        base_rate = base_meta.copy()
        base_rate['feature_set'] = 'benchmark'
        base_rate['model'] = 'base_rate'
        base_rate['family'] = 'benchmark'
        base_rate['probability'] = np.clip(over_rate, 0.001, 0.999)
        cls_parts.append(base_rate)

        try:
            dyn_pred, dyn_prob = _dynamic_direct_predictions(train, test)
            dyn_reg = base_meta.copy()
            dyn_reg['feature_set'] = 'dynamic_direct'
            dyn_reg['model'] = 'dynamic_direct'
            dyn_reg['family'] = 'dynamic'
            dyn_reg['prediction'] = dyn_pred
            reg_parts.append(dyn_reg)

            dyn_cls = base_meta.copy()
            dyn_cls['feature_set'] = 'dynamic_direct'
            dyn_cls['model'] = 'dynamic_direct'
            dyn_cls['family'] = 'dynamic'
            dyn_cls['probability'] = dyn_prob
            cls_parts.append(dyn_cls)
        except Exception as exc:
            errors.append({
                'season': season,
                'task': 'dynamic',
                'feature_set': 'dynamic_direct',
                'model': 'dynamic_direct',
                'error': repr(exc),
            })

        for feature_set, (nums, cats) in feature_sets.items():
            prepared_train = _prep_frame(train, nums, cats)
            prepared_test = _prep_frame(test, nums, cats)

            for model_name, model in _regressors(nums, cats).items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        model.fit(
                            prepared_train[nums + cats],
                            prepared_train['market_residual'],
                        )
                        prediction = model.predict(
                            prepared_test[nums + cats]
                        )
                    out = base_meta.copy()
                    out['feature_set'] = feature_set
                    out['model'] = model_name
                    out['family'] = REG_FAMILY.get(
                        model_name, 'other'
                    )
                    out['prediction'] = prediction
                    reg_parts.append(out)
                except Exception as exc:
                    errors.append({
                        'season': season,
                        'task': 'regression',
                        'feature_set': feature_set,
                        'model': model_name,
                        'error': repr(exc),
                    })

            cls_train = prepared_train[
                pd.to_numeric(
                    prepared_train['market_residual'],
                    errors='coerce',
                ).ne(0)
            ].copy()
            y_cls = pd.to_numeric(
                cls_train['market_residual'], errors='coerce'
            ).gt(0).astype(int)

            for model_name, model in _classifiers(nums, cats).items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        model.fit(
                            cls_train[nums + cats],
                            y_cls,
                        )
                        probability = model.predict_proba(
                            prepared_test[nums + cats]
                        )[:, 1]
                    out = base_meta.copy()
                    out['feature_set'] = feature_set
                    out['model'] = model_name
                    out['family'] = CLS_FAMILY.get(
                        model_name, 'other'
                    )
                    out['probability'] = np.clip(
                        probability, 0.001, 0.999
                    )
                    cls_parts.append(out)
                except Exception as exc:
                    errors.append({
                        'season': season,
                        'task': 'classification',
                        'feature_set': feature_set,
                        'model': model_name,
                        'error': repr(exc),
                    })

    return (
        pd.concat(reg_parts, ignore_index=True),
        pd.concat(cls_parts, ignore_index=True),
        pd.DataFrame(errors),
    )


def _reg_metrics(frame: pd.DataFrame) -> dict:
    actual = pd.to_numeric(frame['market_residual'], errors='coerce')
    pred = pd.to_numeric(frame['prediction'], errors='coerce')
    valid = actual.notna() & pred.notna()
    if not valid.any():
        return {
            'games': 0,
            'mae': np.nan,
            'rmse': np.nan,
            'mean_prediction': np.nan,
        }
    return {
        'games': int(valid.sum()),
        'mae': mean_absolute_error(actual[valid], pred[valid]),
        'rmse': math.sqrt(
            mean_squared_error(actual[valid], pred[valid])
        ),
        'mean_prediction': float(pred[valid].mean()),
    }


def _calibration(
    actual_over: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    if (
        len(actual_over) < 20
        or len(np.unique(actual_over)) < 2
    ):
        return np.nan, np.nan
    p = np.clip(probability, 0.001, 0.999)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(
            C=1e6,
            max_iter=5000,
        )
        model.fit(logit, actual_over.astype(int))
        return (
            float(model.intercept_[0]),
            float(model.coef_[0][0]),
        )
    except Exception:
        return np.nan, np.nan


def _cls_metrics(frame: pd.DataFrame) -> dict:
    actual = pd.to_numeric(frame['market_residual'], errors='coerce')
    prob = pd.to_numeric(frame['probability'], errors='coerce')
    valid = actual.ne(0) & actual.notna() & prob.notna()
    if not valid.any():
        return {
            'games': 0,
            'brier': np.nan,
            'log_loss': np.nan,
            'calibration_intercept': np.nan,
            'calibration_slope': np.nan,
        }
    y = actual[valid].gt(0).astype(int).to_numpy()
    p = np.clip(prob[valid].to_numpy(), 0.001, 0.999)
    intercept, slope = _calibration(y, p)
    return {
        'games': int(valid.sum()),
        'brier': brier_score_loss(y, p),
        'log_loss': log_loss(y, p, labels=[0, 1]),
        'calibration_intercept': intercept,
        'calibration_slope': slope,
    }


def _candidate_summaries(
    reg: pd.DataFrame,
    cls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reg_rows = []
    reg_season_rows = []
    for (feature_set, model), group in reg.groupby(
        ['feature_set', 'model']
    ):
        reg_rows.append({
            'feature_set': feature_set,
            'model': model,
            'family': group['family'].iloc[0],
            **_reg_metrics(group),
        })
        for season, season_group in group.groupby('season'):
            reg_season_rows.append({
                'feature_set': feature_set,
                'model': model,
                'family': group['family'].iloc[0],
                'season': int(season),
                **_reg_metrics(season_group),
            })

    cls_rows = []
    cls_season_rows = []
    for (feature_set, model), group in cls.groupby(
        ['feature_set', 'model']
    ):
        cls_rows.append({
            'feature_set': feature_set,
            'model': model,
            'family': group['family'].iloc[0],
            **_cls_metrics(group),
        })
        for season, season_group in group.groupby('season'):
            cls_season_rows.append({
                'feature_set': feature_set,
                'model': model,
                'family': group['family'].iloc[0],
                'season': int(season),
                **_cls_metrics(season_group),
            })

    return (
        pd.DataFrame(reg_rows),
        pd.DataFrame(reg_season_rows),
        pd.DataFrame(cls_rows),
        pd.DataFrame(cls_season_rows),
    )


def _candidate_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame['feature_set'].astype(str)
        + '::'
        + frame['model'].astype(str)
    )


def _best_reg_candidate(
    prior: pd.DataFrame,
    family: str | None = None,
    allowed_models: list[str] | None = None,
) -> tuple[str, str] | None:
    work = prior.copy()
    if family is not None:
        work = work[work['family'].eq(family)]
    if allowed_models is not None:
        work = work[work['model'].isin(allowed_models)]
    if work.empty:
        return None

    rows = []
    for (feature_set, model), group in work.groupby(
        ['feature_set', 'model']
    ):
        overall = _reg_metrics(group)
        season_maes = [
            _reg_metrics(g)['mae']
            for _, g in group.groupby('season')
        ]
        rows.append({
            'feature_set': feature_set,
            'model': model,
            'mae': overall['mae'],
            'median_season_mae': float(np.nanmedian(season_maes)),
            'complexity': COMPLEXITY_RANK.get(model, 99),
        })
    table = pd.DataFrame(rows).sort_values(
        ['mae', 'median_season_mae', 'complexity', 'feature_set', 'model'],
        kind='stable',
    )
    if table.empty:
        return None
    top = table.iloc[0]
    return str(top['feature_set']), str(top['model'])


def _best_cls_candidate(
    prior: pd.DataFrame,
    family: str | None = None,
    allowed_models: list[str] | None = None,
) -> tuple[str, str] | None:
    work = prior.copy()
    if family is not None:
        work = work[work['family'].eq(family)]
    if allowed_models is not None:
        work = work[work['model'].isin(allowed_models)]
    if work.empty:
        return None

    rows = []
    for (feature_set, model), group in work.groupby(
        ['feature_set', 'model']
    ):
        overall = _cls_metrics(group)
        rows.append({
            'feature_set': feature_set,
            'model': model,
            'brier': overall['brier'],
            'log_loss': overall['log_loss'],
            'complexity': COMPLEXITY_RANK.get(model, 99),
        })
    table = pd.DataFrame(rows).sort_values(
        ['brier', 'log_loss', 'complexity', 'feature_set', 'model'],
        kind='stable',
    )
    if table.empty:
        return None
    top = table.iloc[0]
    return str(top['feature_set']), str(top['model'])


def _select_current(
    frame: pd.DataFrame,
    season: int,
    candidate: tuple[str, str],
    value_col: str,
) -> pd.DataFrame:
    feature_set, model = candidate
    out = frame[
        frame['season'].eq(season)
        & frame['feature_set'].eq(feature_set)
        & frame['model'].eq(model)
    ].copy()
    return out[_meta_columns(out) + [value_col]].copy()


def _fixed_systems(
    reg: pd.DataFrame,
    cls: pd.DataFrame,
) -> list[pd.DataFrame]:
    systems: list[pd.DataFrame] = []
    feature_sets = [
        'v1_weather',
        'v1_plus_team',
        'v1_plus_dynamic',
        'all_context',
    ]

    for system_name, (reg_model, cls_model) in PAIR_MAP.items():
        if system_name == 'dynamic_direct':
            reg_sets = ['dynamic_direct']
        elif system_name == 'v1_exact':
            reg_sets = ['v1_weather']
        else:
            reg_sets = feature_sets

        for feature_set in reg_sets:
            r = reg[
                reg['feature_set'].eq(feature_set)
                & reg['model'].eq(reg_model)
            ].copy()
            c = cls[
                cls['feature_set'].eq(feature_set)
                & cls['model'].eq(cls_model)
            ].copy()
            common = r.merge(
                c[['game_id', 'season', 'probability']],
                on=['game_id', 'season'],
                how='inner',
            )
            if common.empty:
                continue
            common['system'] = (
                system_name
                if system_name in {'v1_exact', 'dynamic_direct'}
                else f'{system_name}__{feature_set}'
            )
            common['reg_source'] = f'{feature_set}::{reg_model}'
            common['cls_source'] = f'{feature_set}::{cls_model}'
            systems.append(
                common[
                    _meta_columns(common)
                    + [
                        'prediction',
                        'probability',
                        'system',
                        'reg_source',
                        'cls_source',
                    ]
                ]
            )
    return systems


def _online_selected_pair(
    reg: pd.DataFrame,
    cls: pd.DataFrame,
    trace: list[dict],
) -> list[pd.DataFrame]:
    parts = []
    for season in OUTER_SEASONS:
        if season == min(OUTER_SEASONS):
            continue
        prior_reg = reg[reg['season'].lt(season)]
        prior_cls = cls[cls['season'].lt(season)]
        reg_pick = _best_reg_candidate(prior_reg)
        cls_pick = _best_cls_candidate(prior_cls)
        if reg_pick is None or cls_pick is None:
            continue
        current_reg = _select_current(
            reg, season, reg_pick, 'prediction'
        )
        current_cls = _select_current(
            cls, season, cls_pick, 'probability'
        )
        current = current_reg.merge(
            current_cls[
                ['game_id', 'season', 'probability']
            ],
            on=['game_id', 'season'],
            how='inner',
        )
        if current.empty:
            continue
        current['system'] = 'selected_pair'
        current['reg_source'] = '::'.join(reg_pick)
        current['cls_source'] = '::'.join(cls_pick)
        trace.append({
            'season': season,
            'system': 'selected_pair',
            'reg_source': '::'.join(reg_pick),
            'cls_source': '::'.join(cls_pick),
        })
        parts.append(current)
    return parts


def _online_diversified(
    reg: pd.DataFrame,
    cls: pd.DataFrame,
    trace: list[dict],
) -> list[pd.DataFrame]:
    parts = []
    families = [
        'linear_spline',
        'tree_bagging',
        'boosting',
        'nonlinear_neural',
    ]
    for season in OUTER_SEASONS:
        if season == min(OUTER_SEASONS):
            continue
        prior_reg = reg[reg['season'].lt(season)]
        prior_cls = cls[cls['season'].lt(season)]

        reg_picks = [
            pick
            for fam in families
            if (
                pick := _best_reg_candidate(
                    prior_reg, family=fam
                )
            ) is not None
        ]
        cls_picks = [
            pick
            for fam in families
            if (
                pick := _best_cls_candidate(
                    prior_cls, family=fam
                )
            ) is not None
        ]
        if not reg_picks or not cls_picks:
            continue

        reg_frames = []
        for i, pick in enumerate(reg_picks):
            cur = _select_current(reg, season, pick, 'prediction')
            reg_frames.append(
                cur[['game_id', 'season', 'prediction']].rename(
                    columns={'prediction': f'pred_{i}'}
                )
            )
        cls_frames = []
        for i, pick in enumerate(cls_picks):
            cur = _select_current(cls, season, pick, 'probability')
            cls_frames.append(
                cur[['game_id', 'season', 'probability']].rename(
                    columns={'probability': f'prob_{i}'}
                )
            )

        reg_wide = reg_frames[0]
        for other in reg_frames[1:]:
            reg_wide = reg_wide.merge(
                other, on=['game_id', 'season'], how='inner'
            )
        cls_wide = cls_frames[0]
        for other in cls_frames[1:]:
            cls_wide = cls_wide.merge(
                other, on=['game_id', 'season'], how='inner'
            )

        meta = reg[
            reg['season'].eq(season)
        ][_meta_columns(reg)].drop_duplicates('game_id')
        current = meta.merge(
            reg_wide, on=['game_id', 'season'], how='inner'
        ).merge(
            cls_wide, on=['game_id', 'season'], how='inner'
        )
        pred_cols = [c for c in current if c.startswith('pred_')]
        prob_cols = [c for c in current if c.startswith('prob_')]
        current['prediction'] = current[pred_cols].mean(axis=1)
        current['probability'] = current[prob_cols].mean(axis=1)
        current['system'] = 'diversified_average'
        current['reg_source'] = '|'.join(
            '::'.join(x) for x in reg_picks
        )
        current['cls_source'] = '|'.join(
            '::'.join(x) for x in cls_picks
        )
        trace.append({
            'season': season,
            'system': 'diversified_average',
            'reg_source': current['reg_source'].iloc[0],
            'cls_source': current['cls_source'].iloc[0],
        })
        parts.append(
            current[
                _meta_columns(current)
                + [
                    'prediction',
                    'probability',
                    'system',
                    'reg_source',
                    'cls_source',
                ]
            ]
        )
    return parts


def _best_feature_for_model_reg(
    prior: pd.DataFrame,
    model: str,
) -> tuple[str, str] | None:
    return _best_reg_candidate(
        prior[prior['model'].eq(model)],
        allowed_models=[model],
    )


def _best_feature_for_model_cls(
    prior: pd.DataFrame,
    model: str,
) -> tuple[str, str] | None:
    return _best_cls_candidate(
        prior[prior['model'].eq(model)],
        allowed_models=[model],
    )


def _stacked_meta(
    reg: pd.DataFrame,
    cls: pd.DataFrame,
    trace: list[dict],
) -> list[pd.DataFrame]:
    parts = []
    for season in OUTER_SEASONS:
        prior_seasons = sorted(
            set(
                int(x)
                for x in reg.loc[
                    reg['season'].lt(season), 'season'
                ].unique()
            )
        )
        if len(prior_seasons) < 2:
            continue

        prior_reg = reg[reg['season'].lt(season)]
        prior_cls = cls[cls['season'].lt(season)]

        reg_picks = [
            pick
            for model in COMPACT_STACK_REG
            if (
                pick := _best_feature_for_model_reg(
                    prior_reg, model
                )
            ) is not None
        ]
        cls_picks = [
            pick
            for model in COMPACT_STACK_CLS
            if (
                pick := _best_feature_for_model_cls(
                    prior_cls, model
                )
            ) is not None
        ]
        if len(reg_picks) < 4 or len(cls_picks) < 4:
            continue

        reg_train = None
        reg_current = None
        for i, pick in enumerate(reg_picks):
            fs, model = pick
            prior_piece = prior_reg[
                prior_reg['feature_set'].eq(fs)
                & prior_reg['model'].eq(model)
            ][
                ['game_id', 'season', 'market_residual', 'prediction']
            ].rename(columns={'prediction': f'p{i}'})
            current_piece = reg[
                reg['season'].eq(season)
                & reg['feature_set'].eq(fs)
                & reg['model'].eq(model)
            ][
                ['game_id', 'season', 'prediction']
            ].rename(columns={'prediction': f'p{i}'})
            reg_train = (
                prior_piece
                if reg_train is None
                else reg_train.merge(
                    prior_piece.drop(columns=['market_residual']),
                    on=['game_id', 'season'],
                    how='inner',
                )
            )
            reg_current = (
                current_piece
                if reg_current is None
                else reg_current.merge(
                    current_piece,
                    on=['game_id', 'season'],
                    how='inner',
                )
            )

        cls_train = None
        cls_current = None
        for i, pick in enumerate(cls_picks):
            fs, model = pick
            prior_piece = prior_cls[
                prior_cls['feature_set'].eq(fs)
                & prior_cls['model'].eq(model)
            ][
                ['game_id', 'season', 'market_residual', 'probability']
            ].rename(columns={'probability': f'q{i}'})
            current_piece = cls[
                cls['season'].eq(season)
                & cls['feature_set'].eq(fs)
                & cls['model'].eq(model)
            ][
                ['game_id', 'season', 'probability']
            ].rename(columns={'probability': f'q{i}'})
            cls_train = (
                prior_piece
                if cls_train is None
                else cls_train.merge(
                    prior_piece.drop(columns=['market_residual']),
                    on=['game_id', 'season'],
                    how='inner',
                )
            )
            cls_current = (
                current_piece
                if cls_current is None
                else cls_current.merge(
                    current_piece,
                    on=['game_id', 'season'],
                    how='inner',
                )
            )

        pred_cols = [c for c in reg_train if c.startswith('p')]
        prob_cols = [c for c in cls_train if c.startswith('q')]
        meta_reg = Ridge(alpha=10.0)
        meta_reg.fit(
            reg_train[pred_cols],
            reg_train['market_residual'],
        )
        reg_current['prediction'] = meta_reg.predict(
            reg_current[pred_cols]
        )

        cls_fit = cls_train[
            pd.to_numeric(
                cls_train['market_residual'], errors='coerce'
            ).ne(0)
        ].copy()
        y_cls = pd.to_numeric(
            cls_fit['market_residual'], errors='coerce'
        ).gt(0).astype(int)
        meta_cls = LogisticRegression(
            C=1.0,
            max_iter=5000,
        )
        meta_cls.fit(cls_fit[prob_cols], y_cls)
        cls_current['probability'] = meta_cls.predict_proba(
            cls_current[prob_cols]
        )[:, 1]

        meta = reg[
            reg['season'].eq(season)
        ][_meta_columns(reg)].drop_duplicates('game_id')
        current = meta.merge(
            reg_current[
                ['game_id', 'season', 'prediction']
            ],
            on=['game_id', 'season'],
            how='inner',
        ).merge(
            cls_current[
                ['game_id', 'season', 'probability']
            ],
            on=['game_id', 'season'],
            how='inner',
        )
        current['system'] = 'stacked_meta'
        current['reg_source'] = '|'.join(
            '::'.join(x) for x in reg_picks
        )
        current['cls_source'] = '|'.join(
            '::'.join(x) for x in cls_picks
        )
        trace.append({
            'season': season,
            'system': 'stacked_meta',
            'reg_source': current['reg_source'].iloc[0],
            'cls_source': current['cls_source'].iloc[0],
        })
        parts.append(current)
    return parts


def _build_system_predictions(
    reg: pd.DataFrame,
    cls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace: list[dict] = []
    parts = _fixed_systems(reg, cls)
    parts.extend(_online_selected_pair(reg, cls, trace))
    parts.extend(_online_diversified(reg, cls, trace))
    parts.extend(_stacked_meta(reg, cls, trace))

    if not parts:
        return pd.DataFrame(), pd.DataFrame(trace)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(trace)


def _wilson(wins: int, graded: int) -> tuple[float, float]:
    if graded <= 0:
        return np.nan, np.nan
    z = 1.96
    p = wins / graded
    denom = 1 + z * z / graded
    center = (p + z * z / (2 * graded)) / denom
    margin = (
        z
        * np.sqrt(
            (p * (1 - p) + z * z / (4 * graded))
            / graded
        )
        / denom
    )
    return center - margin, center + margin


def _qualifier_mask(
    frame: pd.DataFrame,
    threshold: tuple[float, float],
) -> pd.Series:
    edge, prob = threshold
    return (
        pd.to_numeric(frame['prediction'], errors='coerce').ge(edge)
        & pd.to_numeric(
            frame['probability'], errors='coerce'
        ).ge(prob)
    )


def _max_drawdown(units: np.ndarray) -> float:
    if len(units) == 0:
        return np.nan
    cumulative = np.cumsum(units)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    drawdown = cumulative - peak
    return float(abs(np.min(drawdown))) if len(drawdown) else 0.0


def _grade_plays(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            'plays': 0,
            'graded': 0,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
            'hit_rate': np.nan,
            'wilson95_low': np.nan,
            'wilson95_high': np.nan,
            'flat_units_minus110': 0.0,
            'flat_roi_minus110': np.nan,
            'max_drawdown_units': np.nan,
            'p_value_vs_minus110': np.nan,
            'max_single_season_profit_share': np.nan,
        }

    ordered = frame.copy()
    ordered['_date'] = pd.to_datetime(
        ordered.get('gameday'), errors='coerce'
    )
    ordered = ordered.sort_values(
        ['season', '_date', 'week', 'game_id'],
        kind='stable',
    )
    residual = pd.to_numeric(
        ordered['market_residual'], errors='coerce'
    )
    wins_mask = residual.gt(0)
    losses_mask = residual.lt(0)
    pushes_mask = residual.eq(0)
    wins = int(wins_mask.sum())
    losses = int(losses_mask.sum())
    pushes = int(pushes_mask.sum())
    graded = wins + losses
    units = np.where(
        wins_mask,
        100 / 110,
        np.where(losses_mask, -1.0, 0.0),
    )
    low, high = _wilson(wins, graded)

    season_profit = (
        pd.DataFrame({
            'season': ordered['season'].to_numpy(),
            'units': units,
        })
        .groupby('season')['units']
        .sum()
    )
    total_positive = float(season_profit.clip(lower=0).sum())
    max_share = (
        float(season_profit.clip(lower=0).max() / total_positive)
        if total_positive > 0
        else np.nan
    )

    return {
        'plays': int(len(ordered)),
        'graded': graded,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / graded if graded else np.nan,
        'wilson95_low': low,
        'wilson95_high': high,
        'flat_units_minus110': float(np.sum(units)),
        'flat_roi_minus110': (
            float(np.sum(units)) / graded
            if graded else np.nan
        ),
        'max_drawdown_units': _max_drawdown(
            np.asarray(units, dtype=float)
        ),
        'p_value_vs_minus110': (
            binomtest(
                wins,
                graded,
                p=BREAKEVEN,
                alternative='greater',
            ).pvalue
            if graded else np.nan
        ),
        'max_single_season_profit_share': max_share,
    }


def _system_prediction_metrics(frame: pd.DataFrame) -> dict:
    actual = pd.to_numeric(frame['market_residual'], errors='coerce')
    pred = pd.to_numeric(frame['prediction'], errors='coerce')
    prob = pd.to_numeric(frame['probability'], errors='coerce')
    valid_reg = actual.notna() & pred.notna()
    valid_cls = actual.ne(0) & prob.notna()
    y = actual[valid_cls].gt(0).astype(int).to_numpy()
    p = np.clip(prob[valid_cls].to_numpy(), 0.001, 0.999)
    intercept, slope = _calibration(y, p) if len(y) else (np.nan, np.nan)

    return {
        'games': int(valid_reg.sum()),
        'mae': (
            mean_absolute_error(actual[valid_reg], pred[valid_reg])
            if valid_reg.any() else np.nan
        ),
        'rmse': (
            math.sqrt(
                mean_squared_error(
                    actual[valid_reg], pred[valid_reg]
                )
            )
            if valid_reg.any() else np.nan
        ),
        'brier': (
            brier_score_loss(y, p)
            if len(y) else np.nan
        ),
        'log_loss': (
            log_loss(y, p, labels=[0, 1])
            if len(y) else np.nan
        ),
        'calibration_intercept': intercept,
        'calibration_slope': slope,
        'mean_prediction': (
            float(pred[valid_reg].mean())
            if valid_reg.any() else np.nan
        ),
    }


def _bh_qvalues(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    m = len(valid)
    if not m:
        return out
    running = 1.0
    for rank in range(m, 0, -1):
        idx = valid.index[rank - 1]
        q = min(running, float(valid.loc[idx]) * m / rank)
        out.loc[idx] = q
        running = q
    return out


def _system_summaries(
    systems: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    season_rows = []

    for system, group in systems.groupby('system'):
        base = {
            'system': system,
            **_system_prediction_metrics(group),
        }
        for rule_name, threshold in RULES.items():
            plays = group[_qualifier_mask(group, threshold)]
            grade = _grade_plays(plays)
            prefix = (
                'qualifies'
                if rule_name.startswith('qualifies')
                else 'strong'
            )
            for key, value in grade.items():
                base[f'{prefix}_{key}'] = value
        rows.append(base)

        for season, season_group in group.groupby('season'):
            row = {
                'system': system,
                'season': int(season),
                **_system_prediction_metrics(season_group),
            }
            for rule_name, threshold in RULES.items():
                plays = season_group[
                    _qualifier_mask(season_group, threshold)
                ]
                grade = _grade_plays(plays)
                prefix = (
                    'qualifies'
                    if rule_name.startswith('qualifies')
                    else 'strong'
                )
                row[f'{prefix}_plays'] = grade['plays']
                row[f'{prefix}_wins'] = grade['wins']
                row[f'{prefix}_losses'] = grade['losses']
                row[f'{prefix}_roi'] = grade['flat_roi_minus110']
            season_rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary['qualifies_q_value_bh'] = _bh_qvalues(
            summary['qualifies_p_value_vs_minus110']
        )
        summary['strong_q_value_bh'] = _bh_qvalues(
            summary['strong_p_value_vs_minus110']
        )
    return summary, pd.DataFrame(season_rows)


def _paired_bootstrap(
    systems: pd.DataFrame,
) -> pd.DataFrame:
    v1 = systems[
        systems['system'].eq('v1_exact')
    ].set_index(['game_id', 'season'])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []

    for system, group in systems.groupby('system'):
        if system == 'v1_exact':
            continue
        ch = group.set_index(['game_id', 'season'])
        common = v1.index.intersection(ch.index)
        if len(common) < 50:
            continue
        base = v1.loc[common]
        challenger = ch.loc[common]

        actual = pd.to_numeric(
            base['market_residual'], errors='coerce'
        ).to_numpy()
        bpred = pd.to_numeric(
            base['prediction'], errors='coerce'
        ).to_numpy()
        cpred = pd.to_numeric(
            challenger['prediction'], errors='coerce'
        ).to_numpy()
        bprob = pd.to_numeric(
            base['probability'], errors='coerce'
        ).to_numpy()
        cprob = pd.to_numeric(
            challenger['probability'], errors='coerce'
        ).to_numpy()

        valid = (
            np.isfinite(actual)
            & np.isfinite(bpred)
            & np.isfinite(cpred)
            & np.isfinite(bprob)
            & np.isfinite(cprob)
        )
        actual = actual[valid]
        bpred = bpred[valid]
        cpred = cpred[valid]
        bprob = bprob[valid]
        cprob = cprob[valid]
        n = len(actual)
        if n < 50:
            continue

        y = (actual > 0).astype(float)
        nonpush = actual != 0
        mae_diffs = np.empty(BOOTSTRAP_SAMPLES)
        brier_diffs = np.empty(BOOTSTRAP_SAMPLES)

        for i in range(BOOTSTRAP_SAMPLES):
            idx = rng.integers(0, n, n)
            mae_diffs[i] = (
                np.mean(np.abs(actual[idx] - cpred[idx]))
                - np.mean(np.abs(actual[idx] - bpred[idx]))
            )
            keep = nonpush[idx]
            if not keep.any():
                brier_diffs[i] = np.nan
            else:
                sampled = idx[keep]
                brier_diffs[i] = (
                    np.mean(
                        (y[sampled] - cprob[sampled]) ** 2
                    )
                    - np.mean(
                        (y[sampled] - bprob[sampled]) ** 2
                    )
                )

        bvalid = brier_diffs[np.isfinite(brier_diffs)]
        rows.append({
            'system': system,
            'paired_games': n,
            'mae_diff_minus_v1': (
                np.mean(np.abs(actual - cpred))
                - np.mean(np.abs(actual - bpred))
            ),
            'mae_ci95_low': float(
                np.quantile(mae_diffs, 0.025)
            ),
            'mae_ci95_high': float(
                np.quantile(mae_diffs, 0.975)
            ),
            'bootstrap_probability_mae_better': float(
                np.mean(mae_diffs < 0)
            ),
            'brier_diff_minus_v1': (
                np.mean(
                    (y[nonpush] - cprob[nonpush]) ** 2
                )
                - np.mean(
                    (y[nonpush] - bprob[nonpush]) ** 2
                )
            ),
            'brier_ci95_low': (
                float(np.quantile(bvalid, 0.025))
                if len(bvalid) else np.nan
            ),
            'brier_ci95_high': (
                float(np.quantile(bvalid, 0.975))
                if len(bvalid) else np.nan
            ),
            'bootstrap_probability_brier_better': (
                float(np.mean(bvalid < 0))
                if len(bvalid) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def _season_improvement_counts(
    by_season: pd.DataFrame,
) -> pd.DataFrame:
    v1 = by_season[
        by_season['system'].eq('v1_exact')
    ].set_index('season')
    rows = []
    for system, group in by_season.groupby('system'):
        if system == 'v1_exact':
            continue
        ch = group.set_index('season')
        common = v1.index.intersection(ch.index)
        improved = 0
        materially_bad_other = 0
        for season in common:
            mae_diff = float(ch.loc[season, 'mae'] - v1.loc[season, 'mae'])
            brier_diff = float(
                ch.loc[season, 'brier'] - v1.loc[season, 'brier']
            )
            if (
                (mae_diff < 0 and brier_diff <= 0.005)
                or (brier_diff < 0 and mae_diff <= 0.10)
            ):
                improved += 1
            if (
                (mae_diff > 0.10 and brier_diff >= 0)
                or (brier_diff > 0.005 and mae_diff >= 0)
            ):
                materially_bad_other += 1
        rows.append({
            'system': system,
            'common_seasons': int(len(common)),
            'seasons_improved_one_metric_without_material_other_degradation': improved,
            'seasons_materially_worse': materially_bad_other,
        })
    return pd.DataFrame(rows)


def _advancement_screen(
    summary: pd.DataFrame,
    by_season: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    v1 = summary[summary['system'].eq('v1_exact')]
    if v1.empty:
        return pd.DataFrame(), {
            'status': 'ERROR_NO_V1_BASELINE',
            'primary_candidate': None,
        }
    v1_row = v1.iloc[0]
    season_counts = _season_improvement_counts(by_season)
    boot = bootstrap.set_index('system') if not bootstrap.empty else pd.DataFrame()

    rows = []
    for _, row in summary.iterrows():
        system = str(row['system'])
        if system == 'v1_exact':
            continue
        season_row = season_counts[
            season_counts['system'].eq(system)
        ]
        common_seasons = (
            int(season_row.iloc[0]['common_seasons'])
            if not season_row.empty else 0
        )
        improved = (
            int(
                season_row.iloc[0][
                    'seasons_improved_one_metric_without_material_other_degradation'
                ]
            )
            if not season_row.empty else 0
        )

        if (
            isinstance(boot, pd.DataFrame)
            and not boot.empty
            and system in boot.index
        ):
            b = boot.loc[system]
            prob_mae = float(
                b['bootstrap_probability_mae_better']
            )
            prob_brier = float(
                b['bootstrap_probability_brier_better']
            )
        else:
            prob_mae = np.nan
            prob_brier = np.nan

        pass_quality = bool(
            row['mae'] <= v1_row['mae']
            and row['brier'] <= v1_row['brier']
        )
        pass_seasons = bool(
            common_seasons >= 5 and improved >= 3
        )
        pass_volume = bool(
            row['qualifies_graded'] >= 25
        )
        pass_roi = bool(
            pd.notna(row['qualifies_flat_roi_minus110'])
            and row['qualifies_flat_roi_minus110'] > 0
        )
        share = row['qualifies_max_single_season_profit_share']
        pass_concentration = bool(
            pd.isna(share) or share <= 0.60
        )
        pass_bootstrap = bool(
            (
                np.isfinite(prob_mae)
                and prob_mae >= 0.80
            )
            or (
                np.isfinite(prob_brier)
                and prob_brier >= 0.80
            )
        )
        historical_pass = all(
            [
                pass_quality,
                pass_seasons,
                pass_volume,
                pass_roi,
                pass_concentration,
                pass_bootstrap,
            ]
        )
        rows.append({
            'system': system,
            'pass_quality': pass_quality,
            'pass_seasons': pass_seasons,
            'pass_volume': pass_volume,
            'pass_positive_roi': pass_roi,
            'pass_profit_concentration': pass_concentration,
            'pass_bootstrap': pass_bootstrap,
            'historical_phase3_pass_before_decision_time': historical_pass,
            'common_seasons': common_seasons,
            'improved_seasons': improved,
            'mae': row['mae'],
            'brier': row['brier'],
            'qualifies_graded': row['qualifies_graded'],
            'qualifies_roi': row['qualifies_flat_roi_minus110'],
            'bootstrap_probability_mae_better': prob_mae,
            'bootstrap_probability_brier_better': prob_brier,
        })

    screen = pd.DataFrame(rows)
    passing = screen[
        screen['historical_phase3_pass_before_decision_time']
    ].copy()
    if passing.empty:
        result = {
            'status': 'NO_SYSTEM_PASSED_PRE_DECISION_TIME_GATES',
            'primary_candidate': None,
        }
    else:
        passing = passing.sort_values(
            [
                'brier',
                'mae',
                'qualifies_roi',
                'system',
            ],
            ascending=[True, True, False, True],
            kind='stable',
        )
        result = {
            'status': 'HISTORICAL_CANDIDATE_AWAITS_2025_DECISION_TIME',
            'primary_candidate': str(
                passing.iloc[0]['system']
            ),
        }
    return screen, result


def main() -> None:
    df = read_df(DATA_PATH).copy()
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df = df.dropna(
        subset=[
            'season',
            'closing_total',
            'actual_total_points',
            'market_residual',
        ]
    )
    df = _weather_features(df, LEAD)
    df = df[
        df[f'forecast_complete_{LEAD}h'].fillna(False)
    ].copy()
    # Do not filter the historical training sample on new-feature availability.
    # The frozen v1 benchmark must see exactly the same forecast-native rows it
    # originally used. Challenger pipelines impute the small number of missing
    # team/dynamic values instead.
    _, _, unavailable = _optional_models()
    reg, cls, errors = _fit_outer_predictions(df)

    (
        reg_summary,
        reg_by_season,
        cls_summary,
        cls_by_season,
    ) = _candidate_summaries(reg, cls)

    systems, trace = _build_system_predictions(reg, cls)
    if systems.empty:
        raise RuntimeError('No paired model systems were produced.')

    system_summary, system_by_season = _system_summaries(systems)
    bootstrap = _paired_bootstrap(systems)
    advancement, result = _advancement_screen(
        system_summary,
        system_by_season,
        bootstrap,
    )

    write_df(
        reg,
        'outputs/nfl/model_tournament_regression_predictions.csv',
    )
    write_df(
        cls,
        'outputs/nfl/model_tournament_classifier_predictions.csv',
    )
    write_df(
        reg_summary,
        'outputs/nfl/model_tournament_regression_summary.csv',
    )
    write_df(
        reg_by_season,
        'outputs/nfl/model_tournament_regression_by_season.csv',
    )
    write_df(
        cls_summary,
        'outputs/nfl/model_tournament_classifier_summary.csv',
    )
    write_df(
        cls_by_season,
        'outputs/nfl/model_tournament_classifier_by_season.csv',
    )
    write_df(
        systems,
        'outputs/nfl/model_tournament_system_predictions.csv',
    )
    write_df(
        system_summary,
        'outputs/nfl/model_tournament_system_summary.csv',
    )
    write_df(
        system_by_season,
        'outputs/nfl/model_tournament_system_by_season.csv',
    )
    write_df(
        trace,
        'outputs/nfl/model_tournament_online_selection_trace.csv',
    )
    write_df(
        bootstrap,
        'outputs/nfl/model_tournament_bootstrap.csv',
    )
    write_df(
        advancement,
        'outputs/nfl/model_tournament_advancement_screen.csv',
    )
    write_df(
        errors,
        'outputs/nfl/model_tournament_errors.csv',
    )

    out_dir = ensure_dir('outputs/nfl')
    (
        out_dir / 'model_tournament_advancement.json'
    ).write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding='utf-8',
    )

    top_reg = reg_summary.sort_values(
        ['mae', 'rmse', 'model', 'feature_set'],
        kind='stable',
    ).head(20)
    top_cls = cls_summary.sort_values(
        ['brier', 'log_loss', 'model', 'feature_set'],
        kind='stable',
    ).head(20)
    top_systems = system_summary.sort_values(
        ['brier', 'mae', 'system'],
        kind='stable',
    ).head(30)

    lines = [
        '# NFL Comprehensive Model Tournament — Phase 3',
        '',
        'This tournament was preregistered before results and leaves the '
        'frozen v1 protocol untouched.',
        '',
        f'- Eligible forecast-native rows: **{len(df):,}**',
        f'- Regression candidate rows: **{len(reg_summary)}**',
        f'- Classification candidate rows: **{len(cls_summary)}**',
        f'- Paired/ensemble systems: **{len(system_summary)}**',
        f'- Optional-library warnings: **{len(unavailable)}**',
        f'- Fit failures recorded: **{len(errors)}**',
        '',
        '## Advancement status',
        '',
        f"- **{result['status']}**",
        f"- Primary historical candidate: **{result.get('primary_candidate')}**",
        '',
        '## Best regression candidates',
        '',
        top_reg.to_markdown(index=False),
        '',
        '## Best probability candidates',
        '',
        top_cls.to_markdown(index=False),
        '',
        '## Best paired systems by Brier then MAE',
        '',
        top_systems.to_markdown(index=False),
        '',
        '## Preregistered advancement screen',
        '',
        advancement.to_markdown(index=False),
        '',
        '## Optional model availability',
        '',
        (
            '\n'.join(f'- {item}' for item in unavailable)
            if unavailable else '- All optional model families loaded.'
        ),
        '',
        'A Phase 3 historical pass is not a production promotion. '
        'Any passing candidate still requires the archived 2025 '
        'decision-time test and prospective 2026 tracking.',
    ]
    (
        out_dir / 'model_tournament_phase3.md'
    ).write_text(
        '\n'.join(lines),
        encoding='utf-8',
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    if unavailable:
        print('Optional model warnings:')
        for item in unavailable:
            print(f'- {item}')
    if not errors.empty:
        print(f'Fit failures: {len(errors)}')


if __name__ == '__main__':
    main()
