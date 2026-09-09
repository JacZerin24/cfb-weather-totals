from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fcs_model import (
    FCS_CATEGORICAL_FEATURES,
    FCS_NUMERIC_FEATURES,
    _prepare_features,
    build_fcs_model,
    historical_fcs_training,
)
from .utils import ROOT, ensure_dir, read_df, write_df

OUTDIR = ROOT / 'outputs' / 'diagnostics' / 'fcs_live_attribution'
BOARD_PATH = ROOT / 'outputs' / 'weekly_board.csv'


def _predict(model, raw_row: pd.DataFrame) -> float:
    prepared = _prepare_features(raw_row.copy())
    return float(model.predict(prepared[FCS_NUMERIC_FEATURES + FCS_CATEGORICAL_FEATURES])[0])


def _mode(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return 'missing'
    mode = values.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else values.iloc[0]


def _screen(rows: pd.DataFrame, label: str, mask: pd.Series) -> dict:
    s = rows.loc[mask].copy()
    residual = pd.to_numeric(s.get('market_residual'), errors='coerce').dropna()
    return {
        'screen': label,
        'games': int(len(residual)),
        'mean_market_residual': float(residual.mean()) if len(residual) else np.nan,
        'median_market_residual': float(residual.median()) if len(residual) else np.nan,
        'under_rate': float((residual < 0).mean()) if len(residual) else np.nan,
        'mean_actual_total': float(pd.to_numeric(s.loc[residual.index, 'actual_total_points'], errors='coerce').mean()) if len(residual) else np.nan,
    }


def main() -> None:
    ensure_dir('outputs/diagnostics/fcs_live_attribution')
    board = read_df(BOARD_PATH)
    qualifies = board[board['status'].astype(str).eq('QUALIFIES')].copy()
    if len(qualifies) != 1:
        raise RuntimeError(f'Expected exactly one QUALIFIES row; found {len(qualifies)}')
    live = qualifies.iloc[0].copy()
    if str(live.get('model_track')) != 'FCS-only HGB':
        raise RuntimeError('The qualifying row is not using FCS-only HGB.')

    hist = historical_fcs_training()
    model = build_fcs_model()
    features = FCS_NUMERIC_FEATURES + FCS_CATEGORICAL_FEATURES
    model.fit(hist[features], hist['market_residual'])

    # weekly_board intentionally omits pressure, neutral_site and conference_game.
    # Pressure is absent from the live NWS forecast and is therefore missing/imputed.
    # For the two static booleans, select the combination that exactly reproduces the
    # committed live prediction. This is a verification step, not a tuning step.
    base = pd.DataFrame([live.to_dict()])
    base['pressure'] = np.nan
    target_pred = float(live['pred_market_residual'])
    candidates = []
    for neutral_site in [False, True]:
        for conference_game in [False, True]:
            candidate = base.copy()
            candidate['neutral_site'] = neutral_site
            candidate['conference_game'] = conference_game
            pred = _predict(model, candidate)
            candidates.append({
                'neutral_site': neutral_site,
                'conference_game': conference_game,
                'prediction': pred,
                'abs_error_vs_committed': abs(pred - target_pred),
            })
    candidate_df = pd.DataFrame(candidates).sort_values('abs_error_vs_committed').reset_index(drop=True)
    chosen = candidate_df.iloc[0]
    if float(chosen['abs_error_vs_committed']) > 1e-6:
        raise RuntimeError(
            'Could not exactly reproduce committed FCS prediction; closest error=' +
            str(chosen['abs_error_vs_committed'])
        )
    base['neutral_site'] = bool(chosen['neutral_site'])
    base['conference_game'] = bool(chosen['conference_game'])
    base_pred = _predict(model, base)

    # Typical FCS values are computed from the exact training sample used by the model.
    medians = {c: float(pd.to_numeric(hist[c], errors='coerce').median()) for c in FCS_NUMERIC_FEATURES}
    modes = {c: _mode(hist[c].astype(str)) for c in FCS_CATEGORICAL_FEATURES}

    groups = {
        'market_total': ['closing_total'],
        'wind': ['wind_mph'],
        'temperature': ['temperature_f'],
        'humidity': ['humidity'],
        'precipitation': ['precipitation'],
        'snowfall': ['snowfall'],
        'dewpoint': ['dewpoint_f'],
        'pressure_missing_vs_typical': ['pressure'],
        'indoor_outdoor': ['game_indoors_bool'],
        'neutral_site': ['neutral_site'],
        'conference_game': ['conference_game'],
        'line_provider': ['line_provider'],
        'home_conference': ['home_conference'],
        'away_conference': ['away_conference'],
    }

    rows = []
    base_prepared = _prepare_features(base.copy())
    for name, cols in groups.items():
        scenario = base.copy()
        for col in cols:
            if col in FCS_NUMERIC_FEATURES:
                scenario[col] = medians[col]
            elif col == 'game_indoors_bool':
                scenario['game_indoors_bool'] = modes[col]
            else:
                scenario[col] = modes[col]
        # Force category bins to be recomputed when the underlying continuous
        # variable is changed; otherwise the counterfactual would be internally inconsistent.
        if name == 'market_total' and 'total_bin' in scenario.columns:
            scenario = scenario.drop(columns=['total_bin'])
        if name == 'wind' and 'wind_bin' in scenario.columns:
            scenario = scenario.drop(columns=['wind_bin'])
        if name == 'temperature' and 'temp_bin' in scenario.columns:
            scenario = scenario.drop(columns=['temp_bin'])
        cf_pred = _predict(model, scenario)
        actual_cols = {}
        typical_cols = {}
        for col in cols:
            if col == 'game_indoors_bool':
                actual_cols[col] = str(base_prepared.iloc[0][col])
                typical_cols[col] = str(modes[col])
            else:
                actual_cols[col] = None if pd.isna(base_prepared.iloc[0].get(col)) else base_prepared.iloc[0].get(col)
                typical_cols[col] = medians[col] if col in FCS_NUMERIC_FEATURES else str(modes[col])
        rows.append({
            'factor': name,
            'actual_value': json.dumps(actual_cols, default=str),
            'typical_fcs_value': json.dumps(typical_cols, default=str),
            'base_pred_residual': base_pred,
            'counterfactual_pred_residual': cf_pred,
            'projection_change_if_typical': cf_pred - base_pred,
            'interpretation': (
                'current value pushes UNDER' if cf_pred - base_pred > 0.05 else
                'current value pushes OVER / offsets under' if cf_pred - base_pred < -0.05 else
                'little local effect'
            ),
        })
    factor_df = pd.DataFrame(rows).sort_values('projection_change_if_typical', ascending=False)

    # Joint counterfactuals capture interactions; they are more meaningful than
    # summing individual HGB effects, which is not valid for a nonlinear tree model.
    joint_rows = []
    for label, numeric_cols, categorical_cols in [
        ('all_weather_typical', ['wind_mph','temperature_f','humidity','precipitation','snowfall','dewpoint_f','pressure'], []),
        ('all_context_typical', [], ['game_indoors_bool','neutral_site','conference_game','line_provider','home_conference','away_conference']),
        ('market_plus_weather_typical', ['closing_total','wind_mph','temperature_f','humidity','precipitation','snowfall','dewpoint_f','pressure'], []),
        ('all_inputs_typical', FCS_NUMERIC_FEATURES, FCS_CATEGORICAL_FEATURES),
    ]:
        scenario = base.copy()
        for col in numeric_cols:
            scenario[col] = medians[col]
        for col in categorical_cols:
            if col == 'game_indoors_bool':
                scenario['game_indoors_bool'] = modes[col]
            else:
                scenario[col] = modes[col]
        for derived in ['wind_bin','temp_bin','total_bin']:
            if derived in scenario.columns and (
                (derived == 'wind_bin' and 'wind_mph' in numeric_cols) or
                (derived == 'temp_bin' and 'temperature_f' in numeric_cols) or
                (derived == 'total_bin' and 'closing_total' in numeric_cols)
            ):
                scenario = scenario.drop(columns=[derived])
        pred = _predict(model, scenario)
        joint_rows.append({
            'scenario': label,
            'pred_market_residual': pred,
            'projection_change_vs_live': pred - base_pred,
            'projected_total_at_current_market_63_5': float(live['closing_total']) + pred,
        })
    joint_df = pd.DataFrame(joint_rows)

    # Local sweeps show nonlinear response while keeping all other live inputs fixed.
    sweep_specs = {
        'closing_total': [50, 55, 60, 63.5, 65, 70],
        'wind_mph': [0, 5, 10, 15, 20],
        'temperature_f': [50, 70, 85, 91, 100],
        'humidity': [30, 50, 56, 70, 85, 95],
        'precipitation': [0, 0.05, 0.10, 0.25, 0.50],
    }
    sweep_rows = []
    for feature, values in sweep_specs.items():
        for value in values:
            scenario = base.copy()
            scenario[feature] = value
            derived = {'closing_total':'total_bin','wind_mph':'wind_bin','temperature_f':'temp_bin'}.get(feature)
            if derived and derived in scenario.columns:
                scenario = scenario.drop(columns=[derived])
            pred = _predict(model, scenario)
            sweep_rows.append({
                'feature': feature,
                'value': value,
                'pred_market_residual': pred,
                'projected_total_using_current_market': float(live['closing_total']) + pred,
            })
    sweep_df = pd.DataFrame(sweep_rows)

    # Descriptive historical screens: these are context, not causal attribution.
    h = hist.copy()
    ct = pd.to_numeric(h['closing_total'], errors='coerce')
    temp = pd.to_numeric(h['temperature_f'], errors='coerce')
    wind = pd.to_numeric(h['wind_mph'], errors='coerce')
    precip = pd.to_numeric(h['precipitation'], errors='coerce')
    screens = [
        _screen(h, 'all historical FCS', pd.Series(True, index=h.index)),
        _screen(h, 'closing total >= 63', ct >= 63),
        _screen(h, 'closing total >= 63 and temp >= 85F', (ct >= 63) & (temp >= 85)),
        _screen(h, 'closing total >= 63, temp >= 85F, wind <= 5 mph', (ct >= 63) & (temp >= 85) & (wind <= 5)),
        _screen(h, 'closing total >= 63, temp >= 85F, wind <= 5 mph, precip <= .01in', (ct >= 63) & (temp >= 85) & (wind <= 5) & (precip.fillna(0) <= 0.01)),
        _screen(h, 'home UAC', h['home_conference'].astype(str).eq('UAC')),
        _screen(h, 'away Pioneer', h['away_conference'].astype(str).eq('Pioneer')),
        _screen(h, 'home UAC vs away Pioneer', h['home_conference'].astype(str).eq('UAC') & h['away_conference'].astype(str).eq('Pioneer')),
    ]
    screen_df = pd.DataFrame(screens)

    metadata = pd.DataFrame([{
        'game_id': live['game_id'],
        'away_team': live['away_team'],
        'home_team': live['home_team'],
        'market_total': float(live['closing_total']),
        'committed_pred_residual': target_pred,
        'reproduced_pred_residual': base_pred,
        'committed_projected_total': float(live['model_projected_total']),
        'reproduced_projected_total': float(live['closing_total']) + base_pred,
        'neutral_site': bool(chosen['neutral_site']),
        'conference_game': bool(chosen['conference_game']),
        'training_games': int(len(hist)),
        'training_first_season': int(pd.to_numeric(hist['season']).min()),
        'training_last_season': int(pd.to_numeric(hist['season']).max()),
        'training_mean_residual': float(hist['market_residual'].mean()),
        'training_median_total': medians['closing_total'],
    }])

    write_df(metadata, OUTDIR / 'metadata.csv')
    write_df(candidate_df, OUTDIR / 'static_boolean_reproduction.csv')
    write_df(factor_df, OUTDIR / 'factor_counterfactuals.csv')
    write_df(joint_df, OUTDIR / 'joint_counterfactuals.csv')
    write_df(sweep_df, OUTDIR / 'feature_sweeps.csv')
    write_df(screen_df, OUTDIR / 'historical_context_screens.csv')

    print('--- METADATA ---')
    print(metadata.to_string(index=False))
    print('\n--- FACTOR COUNTERFACTUALS ---')
    print(factor_df.to_string(index=False))
    print('\n--- JOINT COUNTERFACTUALS ---')
    print(joint_df.to_string(index=False))
    print('\n--- HISTORICAL CONTEXT SCREENS ---')
    print(screen_df.to_string(index=False))
    print('\n--- FEATURE SWEEPS ---')
    print(sweep_df.to_string(index=False))


if __name__ == '__main__':
    main()
