from __future__ import annotations

from io import StringIO
from time import sleep
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from ..utils import ensure_dir, load_yaml, read_df, write_df
from .model_bakeoff import (
    CONTEXT_CATS,
    CONTEXT_NUMS,
    WEATHER_CATS,
    WEATHER_NUMS,
    _available_categorical,
    _available_numeric,
    _prep,
)
from .roi_search import _deep_models


STADIUMS_URL = (
    'https://raw.githubusercontent.com/greerreNFL/Stadiums/'
    'main/data/stadiums.csv'
)
PREVIOUS_RUNS_URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'
FORECAST_MODEL = 'gfs_seamless'
TEST_SEASONS = (2024, 2025)
LEAD_DAYS = (1, 2, 3)
REG_MODEL = 'reg_rf_leaf25'
CLS_MODEL = 'cls_rf_leaf25'
CANDIDATES = {
    'ensemble_over_3pt_60pct': {
        'reg_threshold': 3.0,
        'cls_probability': 0.60,
    },
    'ensemble_over_4pt_60pct': {
        'reg_threshold': 4.0,
        'cls_probability': 0.60,
    },
}
EASTERN = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')


def _download_csv(url: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(
                url,
                timeout=45,
                headers={'User-Agent': 'nfl-weather-totals-research/1.0'},
            )
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except Exception as exc:
            last_error = exc
            sleep(2 ** attempt)
    raise RuntimeError(f'Failed to download {url}: {last_error}')


def _stadiums() -> pd.DataFrame:
    frame = _download_csv(STADIUMS_URL)
    required = {'stadium_id', 'lat', 'lon'}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f'Stadium coordinate source missing columns: {sorted(missing)}'
        )
    frame = frame.copy()
    frame['stadium_id'] = frame['stadium_id'].astype(str)
    frame['lat'] = pd.to_numeric(frame['lat'], errors='coerce')
    frame['lon'] = pd.to_numeric(frame['lon'], errors='coerce')
    return frame.dropna(subset=['stadium_id', 'lat', 'lon'])


def _kickoff_utc(gameday: object, gametime: object) -> pd.Timestamp:
    day = pd.to_datetime(gameday, errors='coerce')
    if pd.isna(day) or pd.isna(gametime):
        return pd.NaT
    text = str(gametime).strip()
    try:
        local = pd.Timestamp(
            f'{day.date().isoformat()} {text}'
        ).tz_localize(EASTERN)
    except Exception:
        return pd.NaT
    return local.tz_convert(UTC)


def _hourly_key(hourly: dict, base: str) -> str | None:
    if base in hourly:
        return base
    candidates = [key for key in hourly if key.startswith(base)]
    return candidates[0] if candidates else None


def _request_previous_runs(
    date_text: str,
    stadium_rows: pd.DataFrame,
) -> list[dict]:
    if stadium_rows.empty:
        return []

    hourly = []
    for lead in LEAD_DAYS:
        hourly.extend([
            f'temperature_2m_previous_day{lead}',
            f'wind_speed_10m_previous_day{lead}',
        ])

    start = pd.Timestamp(date_text)
    end = start + pd.Timedelta(days=1)
    params = {
        'latitude': ','.join(f'{value:.6f}' for value in stadium_rows['lat']),
        'longitude': ','.join(f'{value:.6f}' for value in stadium_rows['lon']),
        'start_date': start.date().isoformat(),
        'end_date': end.date().isoformat(),
        'hourly': ','.join(hourly),
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'UTC',
        'models': FORECAST_MODEL,
    }

    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(
                PREVIOUS_RUNS_URL,
                params=params,
                timeout=90,
                headers={'User-Agent': 'nfl-weather-totals-research/1.0'},
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                raise RuntimeError(
                    f'Unexpected Open-Meteo response type: {type(payload)}'
                )
            if len(payload) != len(stadium_rows):
                raise RuntimeError(
                    'Open-Meteo multi-location response length '
                    f'{len(payload)} != requested {len(stadium_rows)}'
                )
            return payload
        except Exception as exc:
            last_error = exc
            sleep(min(20, 2 ** attempt))

    raise RuntimeError(
        f'Open-Meteo previous-runs request failed for {date_text}: '
        f'{last_error}'
    )


def _nearest_hour_value(
    hourly: dict,
    variable: str,
    kickoff: pd.Timestamp,
) -> float:
    key = _hourly_key(hourly, variable)
    if key is None or 'time' not in hourly:
        return np.nan

    times = pd.to_datetime(hourly['time'], utc=True, errors='coerce')
    values = pd.to_numeric(pd.Series(hourly.get(key, [])), errors='coerce')
    if len(times) == 0 or len(values) != len(times) or pd.isna(kickoff):
        return np.nan

    diffs = np.abs(
        (pd.Series(times) - kickoff).dt.total_seconds().to_numpy()
    )
    if not np.isfinite(diffs).any():
        return np.nan
    idx = int(np.nanargmin(diffs))
    if diffs[idx] > 90 * 60:
        return np.nan
    return float(values.iloc[idx]) if pd.notna(values.iloc[idx]) else np.nan


def _fetch_forecasts(games: pd.DataFrame) -> pd.DataFrame:
    exposed = games[
        games['weather_exposed'].fillna(False)
        & games['stadium_id'].notna()
        & games['kickoff_utc'].notna()
        & games['lat'].notna()
        & games['lon'].notna()
    ].copy()

    rows: list[dict] = []
    for gameday, group in exposed.groupby('gameday', sort=True):
        locs = (
            group[['stadium_id', 'lat', 'lon']]
            .drop_duplicates('stadium_id')
            .reset_index(drop=True)
        )
        payload = _request_previous_runs(str(gameday), locs)
        by_stadium = {
            str(locs.iloc[idx]['stadium_id']): payload[idx]
            for idx in range(len(locs))
        }

        for _, game in group.iterrows():
            stadium_id = str(game['stadium_id'])
            item = by_stadium.get(stadium_id, {})
            hourly = item.get('hourly', {}) if isinstance(item, dict) else {}
            base = {
                'game_id': game['game_id'],
                'season': int(game['season']),
                'gameday': game['gameday'],
                'stadium_id': stadium_id,
                'kickoff_utc': game['kickoff_utc'],
            }
            for lead in LEAD_DAYS:
                temp = _nearest_hour_value(
                    hourly,
                    f'temperature_2m_previous_day{lead}',
                    game['kickoff_utc'],
                )
                wind = _nearest_hour_value(
                    hourly,
                    f'wind_speed_10m_previous_day{lead}',
                    game['kickoff_utc'],
                )
                rows.append({
                    **base,
                    'lead_hours': lead * 24,
                    'forecast_temperature_f': temp,
                    'forecast_wind_mph': wind,
                    'forecast_complete': bool(
                        np.isfinite(temp) and np.isfinite(wind)
                    ),
                    'forecast_model': FORECAST_MODEL,
                })

        sleep(0.15)

    return pd.DataFrame(rows)


def _recompute_weather_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out['temperature_f'] = pd.to_numeric(
        out['temperature_f'], errors='coerce'
    )
    out['wind_mph'] = pd.to_numeric(out['wind_mph'], errors='coerce')

    out['wind_bin'] = pd.cut(
        out['wind_mph'],
        bins=[-1, 5, 10, 15, 20, 200],
        labels=['0-5', '5-10', '10-15', '15-20', '20+'],
    )
    out['temp_bin'] = pd.cut(
        out['temperature_f'],
        bins=[-100, 20, 32, 45, 60, 75, 90, 200],
        labels=['<=20', '20-32', '32-45', '45-60', '60-75', '75-90', '90+'],
    )
    out['wind_excess_10'] = (out['wind_mph'] - 10).clip(lower=0)
    out['wind_excess_15'] = (out['wind_mph'] - 15).clip(lower=0)
    out['cold_excess_below_32'] = (
        32 - out['temperature_f']
    ).clip(lower=0)
    out['extreme_cold_below_20'] = (
        20 - out['temperature_f']
    ).clip(lower=0)
    out['wind_10plus'] = (out['wind_mph'] >= 10).astype(int)
    out['wind_15plus'] = (out['wind_mph'] >= 15).astype(int)
    out['cold_32_or_less'] = (out['temperature_f'] <= 32).astype(int)
    out['cold_windy'] = (
        (out['temperature_f'] <= 40)
        & (out['wind_mph'] >= 12)
    ).astype(int)
    return out


def _american_profit(odds: float) -> float:
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    return odds / 100 if odds > 0 else 100 / abs(odds)


def _grade(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            'games': 0,
            'graded': 0,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
            'hit_rate': np.nan,
            'flat_roi_minus110': np.nan,
            'actual_odds_graded': 0,
            'actual_odds_roi': np.nan,
        }

    residual = pd.to_numeric(frame['market_residual'], errors='coerce')
    wins = int((residual > 0).sum())
    losses = int((residual < 0).sum())
    pushes = int((residual == 0).sum())
    graded = wins + losses
    flat_units = wins * (100 / 110) - losses

    odds = pd.to_numeric(frame.get('over_odds'), errors='coerce')
    valid_odds = odds.notna() & residual.notna() & (residual != 0)
    actual_units = 0.0
    actual_graded = int(valid_odds.sum())
    if actual_graded:
        for idx in frame.index[valid_odds]:
            if residual.loc[idx] > 0:
                actual_units += _american_profit(float(odds.loc[idx]))
            else:
                actual_units -= 1.0

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
    }


def _predict_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    nums: list[str],
    cats: list[str],
) -> pd.DataFrame:
    train_work = _prep(train, nums, cats)
    test_work = _prep(test, nums, cats)

    models = _deep_models(nums, cats)
    reg = models[REG_MODEL][1]
    cls = models[CLS_MODEL][1]

    reg.fit(train_work[nums + cats], train_work['market_residual'])
    cls_train = train_work[train_work['market_residual'] != 0].copy()
    cls_target = (cls_train['market_residual'] > 0).astype(int)
    cls.fit(cls_train[nums + cats], cls_target)

    out = test.copy()
    out['reg_signal'] = reg.predict(test_work[nums + cats])
    out['over_probability'] = cls.predict_proba(
        test_work[nums + cats]
    )[:, 1]
    return out


def _candidate_mask(
    frame: pd.DataFrame,
    reg_threshold: float,
    cls_probability: float,
) -> pd.Series:
    return (
        (frame['reg_signal'] >= reg_threshold)
        & (frame['over_probability'] >= cls_probability)
    )


def _summary_rows(
    predicted: pd.DataFrame,
    weather_mode: str,
    lead_hours: int | None,
) -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    play_parts: list[pd.DataFrame] = []

    for candidate_name, cfg in CANDIDATES.items():
        base = predicted[
            _candidate_mask(
                predicted,
                float(cfg['reg_threshold']),
                float(cfg['cls_probability']),
            )
        ].copy()

        subsets = {
            'all_eligible': base,
            'fixed_roof_only': base[
                base['roof'].astype(str).str.lower().isin(
                    ['outdoors', 'dome']
                )
            ],
            'strict_outdoor_only': base[
                base['roof'].astype(str).str.lower().eq('outdoors')
            ],
        }
        for subset_name, plays in subsets.items():
            if weather_mode == 'forecast':
                plays = plays[plays['decision_weather_eligible']].copy()

            metrics = _grade(plays)
            rows.append({
                'weather_mode': weather_mode,
                'lead_hours': lead_hours,
                'candidate': candidate_name,
                'subset': subset_name,
                'reg_threshold': cfg['reg_threshold'],
                'classifier_probability': cfg['cls_probability'],
                **metrics,
            })

            if not plays.empty:
                tagged = plays.copy()
                tagged['weather_mode'] = weather_mode
                tagged['lead_hours'] = lead_hours
                tagged['candidate'] = candidate_name
                tagged['subset'] = subset_name
                play_parts.append(tagged)

            for season, season_plays in plays.groupby('season'):
                season_metrics = _grade(season_plays)
                rows.append({
                    'weather_mode': weather_mode,
                    'lead_hours': lead_hours,
                    'candidate': candidate_name,
                    'subset': f'{subset_name}_season_{int(season)}',
                    'reg_threshold': cfg['reg_threshold'],
                    'classifier_probability': cfg['cls_probability'],
                    **season_metrics,
                })

    return rows, play_parts


def main() -> None:
    settings = load_yaml('config/nfl_settings.yml')
    model_cfg = settings['modeling']
    df = read_df(settings['data']['processed_path']).copy()
    df = df.dropna(
        subset=[
            'season',
            'closing_total',
            'actual_total_points',
            'market_residual',
        ]
    )
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['kickoff_utc'] = [
        _kickoff_utc(day, time)
        for day, time in zip(df['gameday'], df['gametime'])
    ]

    stadiums = _stadiums()[
        ['stadium_id', 'lat', 'lon', 'stadium_name', 'tz']
    ].copy()
    df['stadium_id'] = df['stadium_id'].astype(str)
    df = df.merge(
        stadiums,
        on='stadium_id',
        how='left',
        suffixes=('', '_coordinate_source'),
    )

    test_games = df[df['season'].isin(TEST_SEASONS)].copy()
    forecasts = _fetch_forecasts(test_games)
    write_df(
        forecasts,
        'outputs/nfl/forecast_realistic_weather_rows.csv',
    )

    nums = _available_numeric(df, CONTEXT_NUMS + WEATHER_NUMS)
    cats = _available_categorical(df, CONTEXT_CATS + WEATHER_CATS)
    min_train = int(model_cfg.get('min_train_games', 1000))

    summary_rows: list[dict] = []
    play_parts: list[pd.DataFrame] = []

    for season in TEST_SEASONS:
        train = df[df['season'] < season].copy()
        observed_test = df[df['season'] == season].copy()
        if len(train) < min_train or observed_test.empty:
            continue

        observed_test['decision_weather_eligible'] = True
        observed_pred = _predict_models(
            train,
            observed_test,
            nums,
            cats,
        )
        rows, parts = _summary_rows(
            observed_pred,
            weather_mode='observed_reference',
            lead_hours=None,
        )
        summary_rows.extend(rows)
        play_parts.extend(parts)

        for lead in LEAD_DAYS:
            lead_forecast = forecasts[
                (forecasts['season'] == season)
                & (forecasts['lead_hours'] == lead * 24)
            ][
                [
                    'game_id',
                    'forecast_temperature_f',
                    'forecast_wind_mph',
                    'forecast_complete',
                    'forecast_model',
                ]
            ].copy()

            forecast_test = observed_test.merge(
                lead_forecast,
                on='game_id',
                how='left',
            )
            exposed = forecast_test['weather_exposed'].fillna(False)
            complete = forecast_test['forecast_complete'].fillna(False)

            forecast_test.loc[
                exposed & complete,
                'temperature_f',
            ] = forecast_test.loc[
                exposed & complete,
                'forecast_temperature_f',
            ]
            forecast_test.loc[
                exposed & complete,
                'wind_mph',
            ] = forecast_test.loc[
                exposed & complete,
                'forecast_wind_mph',
            ]

            forecast_test.loc[
                exposed & ~complete,
                ['temperature_f', 'wind_mph'],
            ] = np.nan
            forecast_test['decision_weather_eligible'] = (
                (~exposed) | complete
            )
            forecast_test = _recompute_weather_features(forecast_test)

            forecast_pred = _predict_models(
                train,
                forecast_test,
                nums,
                cats,
            )
            rows, parts = _summary_rows(
                forecast_pred,
                weather_mode='forecast',
                lead_hours=lead * 24,
            )
            summary_rows.extend(rows)
            play_parts.extend(parts)

    summary = pd.DataFrame(summary_rows)
    write_df(summary, 'outputs/nfl/forecast_realism_summary.csv')

    if play_parts:
        plays = pd.concat(play_parts, ignore_index=True)
        keep = [
            c for c in [
                'game_id', 'season', 'week', 'gameday', 'gametime',
                'away_team', 'home_team', 'stadium_id', 'stadium',
                'roof', 'surface', 'closing_total', 'over_odds',
                'actual_total_points', 'market_residual',
                'temperature_f', 'wind_mph',
                'forecast_temperature_f', 'forecast_wind_mph',
                'forecast_complete', 'forecast_model',
                'reg_signal', 'over_probability',
                'decision_weather_eligible',
                'weather_mode', 'lead_hours', 'candidate', 'subset',
            ]
            if c in plays.columns
        ]
        write_df(
            plays[keep],
            'outputs/nfl/forecast_realism_qualifying_plays.csv',
        )

    exposed_test = test_games[test_games['weather_exposed'].fillna(False)]
    coord_coverage = (
        exposed_test['lat'].notna() & exposed_test['lon'].notna()
    ).mean() if len(exposed_test) else np.nan

    coverage_rows = []
    for lead in LEAD_DAYS:
        lead_frame = forecasts[forecasts['lead_hours'] == lead * 24]
        coverage_rows.append({
            'lead_hours': lead * 24,
            'forecast_model': FORECAST_MODEL,
            'weather_exposed_games': int(len(exposed_test)),
            'weather_exposed_with_coordinates': int(
                (
                    exposed_test['lat'].notna()
                    & exposed_test['lon'].notna()
                ).sum()
            ),
            'forecast_rows': int(len(lead_frame)),
            'complete_forecasts': int(
                lead_frame['forecast_complete'].fillna(False).sum()
            ),
            'complete_forecast_rate_vs_exposed': (
                float(
                    lead_frame['forecast_complete']
                    .fillna(False)
                    .sum()
                    / len(exposed_test)
                )
                if len(exposed_test) else np.nan
            ),
        })
    coverage = pd.DataFrame(coverage_rows)
    write_df(coverage, 'outputs/nfl/forecast_realism_coverage.csv')

    headline = summary[
        summary['subset'].isin(
            ['all_eligible', 'fixed_roof_only', 'strict_outdoor_only']
        )
    ].copy()

    lines = [
        '# NFL Forecast-Realistic Weather Validation',
        '',
        'This test freezes the previously discovered RF ensemble before looking at archived forecast results. It replaces observed outdoor weather in the 2024-2025 test seasons with GFS forecasts archived at fixed 24-, 48-, and 72-hour lead times.',
        '',
        '## Frozen ensemble candidates',
        '',
        '- Volume candidate: RF residual regression >= 3.0 OVER points AND RF classifier OVER probability >= 60%.',
        '- Selective candidate: RF residual regression >= 4.0 OVER points AND RF classifier OVER probability >= 60%.',
        '- Both models use the leaf-25 variants selected in the prior deep research pass.',
        '',
        '## Weather reconstruction',
        '',
        f'- Archived forecast source: Open-Meteo Previous Runs API, model={FORECAST_MODEL}.',
        f'- Stadium-coordinate source: {STADIUMS_URL}',
        f'- Weather-exposed games with coordinate coverage: {coord_coverage:.1%}.',
        '- nflverse gametime is treated as Eastern Time and converted to UTC before matching the nearest hourly forecast.',
        '- Exposed games with missing archived forecasts are excluded rather than silently falling back to observed weather.',
        '- fixed_roof_only excludes retractable open/closed outcomes; strict_outdoor_only is the cleanest weather-sensitive subset.',
        '',
        '## Headline results',
        '',
        headline.to_markdown(index=False) if not headline.empty else '_No headline rows._',
        '',
        '## Forecast coverage',
        '',
        coverage.to_markdown(index=False),
        '',
        '## Important limitation',
        '',
        '- This is forecast-realistic for weather, not yet fully decision-time-realistic for betting inputs. The model is still conditioned on nflverse/PFR closing total and spread. A true 24/48/72-hour betting backtest also requires archived market snapshots from the same decision time.',
        '- The purpose of this stage is to measure how much of the discovered ensemble edge survives when observed weather is replaced by information that was actually forecast before kickoff.',
        '- No thresholds are re-optimized on these 2024-2025 archived forecasts.',
    ]

    out = ensure_dir('outputs/nfl') / 'forecast_realism.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote NFL forecast-realism outputs under {out.parent}')


if __name__ == '__main__':
    main()
