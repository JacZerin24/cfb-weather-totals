from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .predict_week import as_bool, classify_row
from .utils import read_df

FCS_QUALIFY_EDGE = 7.5
FCS_QUALIFY_TOTAL = 58.0
FCS_LEAN_EDGE = 5.0
FCS_LEAN_TOTAL = 56.0

FCS_NUMERIC_FEATURES = [
    'closing_total', 'wind_mph', 'temperature_f', 'humidity', 'precipitation',
    'snowfall', 'dewpoint_f', 'pressure',
]
FCS_CATEGORICAL_FEATURES = [
    'game_indoors_bool', 'neutral_site', 'conference_game', 'line_provider',
    'wind_bin', 'temp_bin', 'total_bin', 'home_conference', 'away_conference',
]


def division_track(frame: pd.DataFrame) -> pd.Series:
    home = frame.get('home_classification', pd.Series('', index=frame.index)).astype(str).str.lower()
    away = frame.get('away_classification', pd.Series('', index=frame.index)).astype(str).str.lower()
    return pd.Series(
        np.select(
            [home.eq('fbs') & away.eq('fbs'), home.eq('fcs') & away.eq('fcs')],
            ['FBS', 'FCS'],
            default='OTHER',
        ),
        index=frame.index,
        dtype='object',
    )


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in FCS_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors='coerce')

    if 'game_indoors_bool' not in out.columns:
        if 'game_indoors' in out.columns:
            out['game_indoors_bool'] = out['game_indoors'].map(as_bool)
        else:
            out['game_indoors_bool'] = False

    if 'wind_bin' not in out.columns:
        out['wind_bin'] = pd.cut(
            out['wind_mph'], [-1, 5, 10, 15, 20, 200],
            labels=['0-5', '5-10', '10-15', '15-20', '20+'],
        )
    if 'temp_bin' not in out.columns:
        out['temp_bin'] = pd.cut(
            out['temperature_f'], [-100, 35, 50, 70, 85, 200],
            labels=['<=35', '35-50', '50-70', '70-85', '85+'],
        )
    if 'total_bin' not in out.columns:
        out['total_bin'] = pd.cut(
            out['closing_total'], [0, 42, 49, 56, 63, 100],
            labels=['<=42', '42-49', '49-56', '56-63', '63+'],
        )

    for col in FCS_CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = 'missing'
        out[col] = out[col].astype(str).fillna('missing')
    return out


def build_fcs_model() -> Pipeline:
    prep = ColumnTransformer([
        (
            'num',
            Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
            ]),
            FCS_NUMERIC_FEATURES,
        ),
        (
            'cat',
            Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]),
            FCS_CATEGORICAL_FEATURES,
        ),
    ])
    return Pipeline([
        ('prep', prep),
        (
            'model',
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.04,
                l2_regularization=0.5,
                min_samples_leaf=35,
                random_state=42,
            ),
        ),
    ])


def historical_fcs_training() -> pd.DataFrame:
    raw = read_df('data/processed/modeling_dataset.csv')
    if raw.empty:
        return raw
    track = division_track(raw)
    out = raw[track.eq('FCS')].copy()
    out = out.dropna(subset=['closing_total', 'actual_total_points', 'season']).copy()
    out['market_residual'] = pd.to_numeric(out.get('market_residual'), errors='coerce')
    if out['market_residual'].isna().any():
        out['actual_total_points'] = pd.to_numeric(out['actual_total_points'], errors='coerce')
        out['closing_total'] = pd.to_numeric(out['closing_total'], errors='coerce')
        out['market_residual'] = out['market_residual'].fillna(out['actual_total_points'] - out['closing_total'])
    return _prepare_features(out.dropna(subset=['market_residual']))


def score_fcs_rows(live: pd.DataFrame) -> pd.DataFrame:
    out = live.copy()
    if 'division_track' not in out.columns:
        out['division_track'] = division_track(out)
    if 'model_track' not in out.columns:
        out['model_track'] = 'GENERAL HGB'

    mask = out['division_track'].eq('FCS') & out.get('closing_total', pd.Series(np.nan, index=out.index)).notna()
    if not mask.any():
        return out

    historical = historical_fcs_training()
    if len(historical) < 1000:
        out.loc[mask, 'model_track'] = 'FCS HGB unavailable'
        return out

    model = build_fcs_model()
    model.fit(
        historical[FCS_NUMERIC_FEATURES + FCS_CATEGORICAL_FEATURES],
        historical['market_residual'],
    )

    prepared = _prepare_features(out.loc[mask].copy())
    pred = model.predict(prepared[FCS_NUMERIC_FEATURES + FCS_CATEGORICAL_FEATURES])
    out.loc[mask, 'pred_market_residual'] = pred
    out.loc[mask, 'model_projected_total'] = pd.to_numeric(out.loc[mask, 'closing_total'], errors='coerce') + pred
    out.loc[mask, 'model_side'] = np.where(pred < 0, 'under', 'over')
    out.loc[mask, 'abs_pred_edge'] = np.abs(pred)
    out.loc[mask, 'model_track'] = 'FCS-only HGB'
    return out


def classify_division_row(row: pd.Series) -> tuple[str, str]:
    if str(row.get('division_track') or '').upper() != 'FCS':
        return classify_row(row)

    if pd.isna(row.get('closing_total')):
        return 'NO LINE', 'No current FCS market total is available.'
    if pd.isna(row.get('pred_market_residual')):
        return 'NO PLAY', 'The dedicated FCS model could not score this game.'

    forecast_ready = str(row.get('nws_status') or '').lower() in {'ok', 'indoor'}
    if not forecast_ready:
        return 'WATCH', 'FCS game does not yet have a usable NWS kickoff forecast.'
    if as_bool(row.get('start_time_tbd', False)):
        return 'WATCH', 'Kickoff time is still TBD, so the FCS weather match is not stable yet.'

    pred = float(row['pred_market_residual'])
    total = float(row['closing_total'])
    if pred <= -FCS_QUALIFY_EDGE and total >= FCS_QUALIFY_TOTAL:
        return (
            'QUALIFIES',
            'FCS-only HGB under edge ≥7.5 with a 58+ total; the conservative 2023-2025 walk-forward candidate screen.',
        )
    if pred <= -FCS_LEAN_EDGE and total >= FCS_LEAN_TOTAL:
        return (
            'LEAN',
            'FCS-only HGB points under with a 5.0+ edge and 56+ total, but it does not meet the conservative 7.5/58 qualifying screen.',
        )
    if pred <= -FCS_QUALIFY_EDGE:
        return (
            'LEAN',
            'FCS-only HGB has a strong under edge, but the market total is below the preferred 58+ qualifying screen.',
        )
    if pred >= FCS_LEAN_EDGE:
        return 'NO PLAY', 'FCS-only model points over; the FCS research track has not validated an over production strategy.'
    return 'NO PLAY', 'FCS-only model does not meet the conservative under-edge screen.'


def fcs_research_tags(row: pd.Series) -> str:
    if str(row.get('division_track') or '').upper() != 'FCS':
        return ''
    tags = ['FCS-only HGB']
    edge = row.get('abs_pred_edge')
    total = row.get('closing_total')
    pred = row.get('pred_market_residual')
    if pd.notna(pred) and pd.notna(edge) and float(pred) < 0 and float(edge) >= FCS_QUALIFY_EDGE:
        tags.append('FCS 7.5+ under edge')
    if pd.notna(total) and float(total) >= FCS_QUALIFY_TOTAL:
        tags.append('FCS 58+ total')
    return '; '.join(tags)
