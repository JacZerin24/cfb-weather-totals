from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ..utils import ensure_dir, read_df, write_df
from .model_bakeoff import BREAKEVEN, _prep
from .roi_search import _deep_models


DATA_PATH = 'data/nfl/processed/forecast_native_dataset.csv'
LEADS = (24, 48, 72)
MIN_TRAIN = 700
MIN_TEST = 200
FIXED_CANDIDATES = {
    'ensemble_over_3pt_60pct': (3.0, 0.60),
    'ensemble_over_4pt_60pct': (4.0, 0.60),
}
CONTEXT_NUMS = ['closing_total', 'week', 'home_rest', 'away_rest']
CONTEXT_CATS = [
    'div_game', 'location', 'weekday', 'month',
    'decision_roof', 'surface', 'total_bin',
]


def _available_numeric(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        c for c in columns
        if c in df.columns and pd.to_numeric(df[c], errors='coerce').notna().any()
    ]


def _available_categorical(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns and df[c].notna().any()]


def _weather_features(df: pd.DataFrame, lead: int) -> pd.DataFrame:
    out = df.copy()
    out['forecast_temperature_f'] = pd.to_numeric(
        out[f'forecast_temp_{lead}h'], errors='coerce'
    )
    out['forecast_wind_mph'] = pd.to_numeric(
        out[f'forecast_wind_{lead}h'], errors='coerce'
    )
    out['forecast_wind_excess_10'] = (
        out['forecast_wind_mph'] - 10
    ).clip(lower=0)
    out['forecast_wind_excess_15'] = (
        out['forecast_wind_mph'] - 15
    ).clip(lower=0)
    out['forecast_cold_below_32'] = (
        32 - out['forecast_temperature_f']
    ).clip(lower=0)
    out['forecast_cold_below_20'] = (
        20 - out['forecast_temperature_f']
    ).clip(lower=0)
    out['forecast_wind_10plus'] = (
        out['forecast_wind_mph'] >= 10
    ).astype(int)
    out['forecast_wind_15plus'] = (
        out['forecast_wind_mph'] >= 15
    ).astype(int)
    out['forecast_cold_32_or_less'] = (
        out['forecast_temperature_f'] <= 32
    ).astype(int)
    out['forecast_cold_windy'] = (
        (out['forecast_temperature_f'] <= 40)
        & (out['forecast_wind_mph'] >= 12)
    ).astype(int)

    # Only use forecast evolution that would already be known at the decision time.
    if lead <= 48:
        out['temp_change_vs_72h'] = (
            pd.to_numeric(out[f'forecast_temp_{lead}h'], errors='coerce')
            - pd.to_numeric(out['forecast_temp_72h'], errors='coerce')
        )
        out['wind_change_vs_72h'] = (
            pd.to_numeric(out[f'forecast_wind_{lead}h'], errors='coerce')
            - pd.to_numeric(out['forecast_wind_72h'], errors='coerce')
        )
    if lead == 24:
        out['temp_change_vs_48h'] = (
            pd.to_numeric(out['forecast_temp_24h'], errors='coerce')
            - pd.to_numeric(out['forecast_temp_48h'], errors='coerce')
        )
        out['wind_change_vs_48h'] = (
            pd.to_numeric(out['forecast_wind_24h'], errors='coerce')
            - pd.to_numeric(out['forecast_wind_48h'], errors='coerce')
        )
        wind_matrix = out[
            ['forecast_wind_24h', 'forecast_wind_48h', 'forecast_wind_72h']
        ].apply(pd.to_numeric, errors='coerce')
        temp_matrix = out[
            ['forecast_temp_24h', 'forecast_temp_48h', 'forecast_temp_72h']
        ].apply(pd.to_numeric, errors='coerce')
        out['wind_forecast_spread'] = wind_matrix.max(axis=1) - wind_matrix.min(axis=1)
        out['temp_forecast_spread'] = temp_matrix.max(axis=1) - temp_matrix.min(axis=1)

    return out


def _feature_columns(df: pd.DataFrame, lead: int, weather: bool) -> tuple[list[str], list[str]]:
    nums = list(CONTEXT_NUMS)
    cats = list(CONTEXT_CATS)
    if weather:
        nums.extend([
            'forecast_temperature_f',
            'forecast_wind_mph',
            'forecast_wind_excess_10',
            'forecast_wind_excess_15',
            'forecast_cold_below_32',
            'forecast_cold_below_20',
            'forecast_wind_10plus',
            'forecast_wind_15plus',
            'forecast_cold_32_or_less',
            'forecast_cold_windy',
        ])
        if lead <= 48:
            nums.extend(['temp_change_vs_72h', 'wind_change_vs_72h'])
        if lead == 24:
            nums.extend([
                'temp_change_vs_48h',
                'wind_change_vs_48h',
                'wind_forecast_spread',
                'temp_forecast_spread',
            ])
    return _available_numeric(df, nums), _available_categorical(df, cats)


def _american_profit(odds: float) -> float:
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    return odds / 100 if odds > 0 else 100 / abs(odds)


def _grade_over(frame: pd.DataFrame) -> dict:
    residual = pd.to_numeric(frame['market_residual'], errors='coerce')
    wins = int((residual > 0).sum())
    losses = int((residual < 0).sum())
    pushes = int((residual == 0).sum())
    graded = wins + losses
    flat_units = wins * (100 / 110) - losses

    odds = pd.to_numeric(frame.get('over_odds'), errors='coerce')
    valid = odds.notna() & (residual != 0)
    actual_units = 0.0
    actual_graded = int(valid.sum())
    for idx in frame.index[valid]:
        actual_units += (
            _american_profit(float(odds.loc[idx]))
            if residual.loc[idx] > 0 else -1.0
        )

    return {
        'games': int(len(frame)),
        'graded': graded,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / graded if graded else np.nan,
        'flat_roi_minus110': flat_units / graded if graded else np.nan,
        'actual_odds_graded': actual_graded,
        'actual_odds_roi': (
            actual_units / actual_graded if actual_graded else np.nan
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
    }


def _walk_forward(
    frame: pd.DataFrame,
    lead: int,
    weather: bool,
) -> pd.DataFrame:
    nums, cats = _feature_columns(frame, lead, weather)
    work = _prep(frame, nums, cats)
    parts = []

    for season in sorted(work['season'].dropna().astype(int).unique()):
        train = work[work['season'] < season].copy()
        test = work[work['season'] == season].copy()
        if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
            continue

        reg = _deep_models(nums, cats)['reg_rf_leaf25'][1]
        cls = _deep_models(nums, cats)['cls_rf_leaf25'][1]

        reg.fit(train[nums + cats], train['market_residual'])
        cls_train = train[train['market_residual'] != 0].copy()
        cls.fit(
            cls_train[nums + cats],
            (cls_train['market_residual'] > 0).astype(int),
        )

        out = test.copy()
        out['reg_signal'] = reg.predict(test[nums + cats])
        out['over_probability'] = cls.predict_proba(test[nums + cats])[:, 1]
        out['test_season'] = season
        parts.append(out)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _summaries(pred: pd.DataFrame, lead: int, feature_set: str) -> tuple[list[dict], list[pd.DataFrame]]:
    rows = []
    play_parts = []
    for name, (reg_threshold, prob_threshold) in FIXED_CANDIDATES.items():
        plays = pred[
            (pred['reg_signal'] >= reg_threshold)
            & (pred['over_probability'] >= prob_threshold)
        ].copy()
        metrics = _grade_over(plays)
        rows.append({
            'lead_hours': lead,
            'feature_set': feature_set,
            'candidate': name,
            'subset': 'all',
            **metrics,
        })
        for season, group in plays.groupby('season'):
            rows.append({
                'lead_hours': lead,
                'feature_set': feature_set,
                'candidate': name,
                'subset': f'season_{int(season)}',
                **_grade_over(group),
            })
        if not plays.empty:
            plays['lead_hours'] = lead
            plays['feature_set'] = feature_set
            plays['candidate'] = name
            play_parts.append(plays)
    return rows, play_parts


def main() -> None:
    df = read_df(DATA_PATH).copy()
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df = df.dropna(
        subset=['season', 'closing_total', 'actual_total_points', 'market_residual']
    )

    rows: list[dict] = []
    play_parts: list[pd.DataFrame] = []

    for lead in LEADS:
        work = _weather_features(df, lead)
        complete = work[f'forecast_complete_{lead}h'].fillna(False)
        work = work[complete].copy()

        for feature_set, use_weather in [
            ('context_only', False),
            ('context_plus_forecast_weather', True),
        ]:
            pred = _walk_forward(work, lead, use_weather)
            if pred.empty:
                continue
            new_rows, new_parts = _summaries(
                pred,
                lead,
                feature_set,
            )
            rows.extend(new_rows)
            play_parts.extend(new_parts)

    summary = pd.DataFrame(rows)
    write_df(summary, 'outputs/nfl/forecast_native_bakeoff.csv')
    if play_parts:
        plays = pd.concat(play_parts, ignore_index=True)
        keep = [
            c for c in [
                'game_id', 'season', 'week', 'gameday',
                'away_team', 'home_team', 'closing_total', 'over_odds',
                'actual_total_points', 'market_residual',
                'decision_roof', 'reg_signal', 'over_probability',
                'lead_hours', 'feature_set', 'candidate',
            ]
            if c in plays.columns
        ]
        write_df(
            plays[keep],
            'outputs/nfl/forecast_native_qualifying_plays.csv',
        )

    headline = summary[summary['subset'].eq('all')].copy()
    lines = [
        '# NFL Forecast-Native Walk-Forward Bakeoff',
        '',
        'This stage trains the model on archived JMA GSM forecasts themselves rather than observed game weather. The RF regression/classifier architecture and the two OVER ensemble thresholds are frozen from the earlier research; they are not re-optimized here.',
        '',
        '## Design',
        '',
        '- Forecast archive: JMA GSM fixed 24h/48h/72h previous-run forecasts.',
        '- Seasons in source table: 2018-2025.',
        f'- Walk-forward minimum training sample: {MIN_TRAIN} games.',
        f'- Minimum test-season sample: {MIN_TEST} games.',
        '- Retractable-roof games are represented as retractable and receive no weather input.',
        '- Forecast evolution features only use older forecasts already available at that decision time.',
        '',
        '## Headline fixed-rule results',
        '',
        headline.to_markdown(index=False) if not headline.empty else '_No results._',
        '',
        '## Remaining limitation',
        '',
        '- Closing total is still the market anchor because the current Odds API key does not have historical snapshot access. Therefore this tests whether forecast-native weather can identify residuals versus the final market, not yet whether a bettor could have achieved the same ROI at a 24/48/72-hour market line.',
        '- The context-only rows are the key benchmark: forecast weather should add value beyond the same market/context model before it earns a live role.',
    ]
    out = ensure_dir('outputs/nfl') / 'forecast_native_bakeoff.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote forecast-native bakeoff outputs under {out.parent}')


if __name__ == '__main__':
    main()
