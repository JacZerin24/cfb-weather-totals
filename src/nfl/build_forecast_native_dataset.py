from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from ..utils import load_yaml, read_df, write_df


STADIUMS_URL = (
    'https://raw.githubusercontent.com/greerreNFL/Stadiums/'
    'main/data/stadiums.csv'
)
PREVIOUS_RUNS_URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'
FORECAST_MODEL = 'jma_gsm'
START_SEASON = 2018
END_SEASON = 2025
LEAD_DAYS = (1, 2, 3)
EASTERN = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')
OUT_PATH = 'data/nfl/processed/forecast_native_dataset.csv'


def _download_csv(url: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(
                url,
                timeout=60,
                headers={'User-Agent': 'nfl-weather-totals-research/1.0'},
            )
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except Exception as exc:
            last_error = exc
            sleep(min(20, 2 ** attempt))
    raise RuntimeError(f'Failed to download {url}: {last_error}')


def _stadiums() -> pd.DataFrame:
    frame = _download_csv(STADIUMS_URL).copy()
    frame['stadium_id'] = frame['stadium_id'].astype(str)
    frame['lat'] = pd.to_numeric(frame['lat'], errors='coerce')
    frame['lon'] = pd.to_numeric(frame['lon'], errors='coerce')
    return frame.dropna(subset=['stadium_id', 'lat', 'lon'])


def _kickoff_utc(gameday: object, gametime: object) -> pd.Timestamp:
    day = pd.to_datetime(gameday, errors='coerce')
    if pd.isna(day) or pd.isna(gametime):
        return pd.NaT
    try:
        local = pd.Timestamp(
            f'{day.date().isoformat()} {str(gametime).strip()}'
        ).tz_localize(EASTERN)
    except Exception:
        return pd.NaT
    return local.tz_convert(UTC)


def _request_range(
    start_date: str,
    end_date: str,
    locations: pd.DataFrame,
) -> list[dict]:
    hourly = []
    for lead in LEAD_DAYS:
        hourly.extend([
            f'temperature_2m_previous_day{lead}',
            f'wind_speed_10m_previous_day{lead}',
        ])
    params = {
        'latitude': ','.join(f'{value:.6f}' for value in locations['lat']),
        'longitude': ','.join(f'{value:.6f}' for value in locations['lon']),
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(hourly),
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'UTC',
        'models': FORECAST_MODEL,
    }
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.get(
                PREVIOUS_RUNS_URL,
                params=params,
                timeout=120,
                headers={'User-Agent': 'nfl-weather-totals-research/1.0'},
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payload = [payload]
            if len(payload) != len(locations):
                raise RuntimeError(
                    f'Location response mismatch {len(payload)} != {len(locations)}'
                )
            return payload
        except Exception as exc:
            last_error = exc
            sleep(min(30, 2 ** attempt))
    raise RuntimeError(
        f'JMA archive request failed for {start_date} to {end_date}: '
        f'{last_error}'
    )


def _value_at_kickoff(
    hourly: dict,
    variable: str,
    kickoff: pd.Timestamp,
) -> float:
    values = hourly.get(variable)
    times = hourly.get('time')
    if values is None or times is None or pd.isna(kickoff):
        return np.nan
    ts = pd.to_datetime(times, utc=True, errors='coerce')
    vals = pd.to_numeric(pd.Series(values), errors='coerce')
    if len(ts) != len(vals) or len(vals) == 0:
        return np.nan
    diff = np.abs((pd.Series(ts) - kickoff).dt.total_seconds().to_numpy())
    if not np.isfinite(diff).any():
        return np.nan
    idx = int(np.nanargmin(diff))
    if diff[idx] > 90 * 60:
        return np.nan
    value = vals.iloc[idx]
    return float(value) if pd.notna(value) else np.nan


def _decision_roof(value: object) -> str:
    roof = str(value or '').strip().lower()
    if roof == 'outdoors':
        return 'outdoors'
    if roof == 'dome':
        return 'dome'
    if roof in {'open', 'closed'}:
        return 'retractable'
    return 'unknown'


def main() -> None:
    settings = load_yaml('config/nfl_settings.yml')
    df = read_df(settings['data']['processed_path']).copy()
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df = df[df['season'].between(START_SEASON, END_SEASON)].copy()
    df['stadium_id'] = df['stadium_id'].astype(str)
    df['kickoff_utc'] = [
        _kickoff_utc(day, time)
        for day, time in zip(df['gameday'], df['gametime'])
    ]
    df['decision_roof'] = df['roof'].map(_decision_roof)

    stadiums = _stadiums()[
        ['stadium_id', 'lat', 'lon', 'stadium_name', 'tz']
    ].drop_duplicates('stadium_id')
    df = df.merge(stadiums, on='stadium_id', how='left')

    outdoor = df[
        df['decision_roof'].eq('outdoors')
        & df['lat'].notna()
        & df['lon'].notna()
        & df['kickoff_utc'].notna()
    ].copy()

    forecast_rows: list[dict] = []
    outdoor['archive_month'] = (
        pd.to_datetime(outdoor['gameday'], errors='coerce')
        .dt.to_period('M')
        .astype(str)
    )
    grouped = list(outdoor.groupby('archive_month', sort=True))
    total_blocks = len(grouped)

    def fetch_block(item: tuple[str, pd.DataFrame]) -> tuple[str, list[dict], int, int]:
        archive_month, group = item
        group = group.copy()
        game_dates = pd.to_datetime(group['gameday'], errors='coerce')
        start_date = game_dates.min().date().isoformat()
        # Include the following UTC day for prime-time games that kick off
        # after midnight UTC.
        end_date = (
            game_dates.max() + pd.Timedelta(days=1)
        ).date().isoformat()
        locations = (
            group[['stadium_id', 'lat', 'lon']]
            .drop_duplicates('stadium_id')
            .reset_index(drop=True)
        )
        payload = _request_range(start_date, end_date, locations)
        by_stadium = {
            str(locations.iloc[i]['stadium_id']): payload[i]
            for i in range(len(locations))
        }

        block_rows: list[dict] = []
        for _, game in group.iterrows():
            item_payload = by_stadium.get(str(game['stadium_id']), {})
            hourly = item_payload.get('hourly') or {}
            row = {
                'game_id': game['game_id'],
                'season': int(game['season']),
                'gameday': game['gameday'],
                'stadium_id': game['stadium_id'],
                'forecast_model': FORECAST_MODEL,
            }
            for lead in LEAD_DAYS:
                row[f'forecast_temp_{lead * 24}h'] = _value_at_kickoff(
                    hourly,
                    f'temperature_2m_previous_day{lead}',
                    game['kickoff_utc'],
                )
                row[f'forecast_wind_{lead * 24}h'] = _value_at_kickoff(
                    hourly,
                    f'wind_speed_10m_previous_day{lead}',
                    game['kickoff_utc'],
                )
            block_rows.append(row)
        return archive_month, block_rows, len(locations), len(group)

    completed = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fetch_block, item): item[0]
            for item in grouped
        }
        for future in as_completed(futures):
            archive_month, block_rows, location_count, game_count = future.result()
            forecast_rows.extend(block_rows)
            completed += 1
            print(
                f'Fetched archive block {completed}/{total_blocks}: '
                f'{archive_month} ({location_count} stadiums, {game_count} games).'
            )

    forecasts = pd.DataFrame(forecast_rows)
    merged = df.merge(forecasts, on=['game_id', 'season', 'gameday', 'stadium_id'], how='left')

    for lead in (24, 48, 72):
        complete = (
            pd.to_numeric(merged[f'forecast_temp_{lead}h'], errors='coerce').notna()
            & pd.to_numeric(merged[f'forecast_wind_{lead}h'], errors='coerce').notna()
        )
        merged[f'forecast_complete_{lead}h'] = np.where(
            merged['decision_roof'].eq('outdoors'),
            complete,
            True,
        )

    # Decision-time roof handling: actual open/closed status is not a model input.
    # Weather is intentionally withheld for retractable roofs.
    for lead in (24, 48, 72):
        non_outdoor = ~merged['decision_roof'].eq('outdoors')
        merged.loc[non_outdoor, f'forecast_temp_{lead}h'] = np.nan
        merged.loc[non_outdoor, f'forecast_wind_{lead}h'] = np.nan

    write_df(merged, OUT_PATH)

    quality = []
    for season, group in merged.groupby('season'):
        outdoor_group = group[group['decision_roof'].eq('outdoors')]
        quality.append({
            'season': int(season),
            'games': int(len(group)),
            'fixed_outdoor_games': int(len(outdoor_group)),
            'coordinate_coverage': float(
                (outdoor_group['lat'].notna() & outdoor_group['lon'].notna()).mean()
            ) if len(outdoor_group) else np.nan,
            'complete_24h': float(
                outdoor_group['forecast_complete_24h'].mean()
            ) if len(outdoor_group) else np.nan,
            'complete_48h': float(
                outdoor_group['forecast_complete_48h'].mean()
            ) if len(outdoor_group) else np.nan,
            'complete_72h': float(
                outdoor_group['forecast_complete_72h'].mean()
            ) if len(outdoor_group) else np.nan,
        })
    write_df(pd.DataFrame(quality), 'outputs/nfl/forecast_native_data_quality.csv')
    print(f'Wrote {len(merged):,} games to {OUT_PATH}')


if __name__ == '__main__':
    main()
