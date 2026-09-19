from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'
OUT = Path('outputs/nfl/jma_archive_probe.json')


def main() -> None:
    params = {
        'latitude': 44.501477,
        'longitude': -88.062162,
        'start_date': '2019-12-01',
        'end_date': '2019-12-01',
        'hourly': ','.join([
            'temperature_2m_previous_day1',
            'temperature_2m_previous_day2',
            'temperature_2m_previous_day3',
            'wind_speed_10m_previous_day1',
            'wind_speed_10m_previous_day2',
            'wind_speed_10m_previous_day3',
        ]),
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'UTC',
        'models': 'jma_gsm',
    }
    response = requests.get(URL, params=params, timeout=90)
    summary = {
        'http_status': response.status_code,
        'url_without_query': URL,
        'model': 'jma_gsm',
        'test_date': '2019-12-01',
        'location': 'Lambeau Field',
    }

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code == 200 and isinstance(payload, dict):
        hourly = payload.get('hourly') or {}
        summary['hourly_keys'] = sorted(hourly.keys())
        counts = {}
        samples = {}
        for key, values in hourly.items():
            if key == 'time':
                continue
            series = pd.to_numeric(pd.Series(values), errors='coerce')
            counts[key] = int(series.notna().sum())
            valid = series.dropna()
            samples[key] = valid.head(3).round(3).tolist()
        summary['nonnull_counts'] = counts
        summary['sample_values'] = samples
        summary['usable_24_48_72'] = all(
            counts.get(name, 0) > 0
            for name in [
                'temperature_2m_previous_day1',
                'temperature_2m_previous_day2',
                'temperature_2m_previous_day3',
                'wind_speed_10m_previous_day1',
                'wind_speed_10m_previous_day2',
                'wind_speed_10m_previous_day3',
            ]
        )
    else:
        summary['usable_24_48_72'] = False
        summary['error'] = payload if payload else response.text[:500]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
