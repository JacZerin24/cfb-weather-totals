from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import ROOT, ensure_dir

NWS_BASE = 'https://api.weather.gov'
CACHE_PATH = ROOT / 'outputs' / 'nws_grid_cache.json'
NWS_CONNECT_TIMEOUT_SECONDS = 15
NWS_READ_TIMEOUT_SECONDS = 45
NWS_RETRY_TOTAL = 3
NWS_RETRY_BACKOFF_SECONDS = 1.0
NWS_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
MAX_NEAREST_DISTANCE = timedelta(hours=3)


def build_nws_retry_policy() -> Retry:
    return Retry(
        total=NWS_RETRY_TOTAL,
        connect=NWS_RETRY_TOTAL,
        read=NWS_RETRY_TOTAL,
        status=NWS_RETRY_TOTAL,
        backoff_factor=NWS_RETRY_BACKOFF_SECONDS,
        status_forcelist=NWS_RETRY_STATUS_CODES,
        allowed_methods=frozenset({'GET'}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def _duration(value: str) -> timedelta:
    match = re.fullmatch(r'P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?', value)
    if not match:
        return timedelta(hours=1)
    parts = {k: int(v or 0) for k, v in match.groupdict().items()}
    return timedelta(days=parts['days'], hours=parts['hours'], minutes=parts['minutes'], seconds=parts['seconds'])


def _interval(valid_time: str) -> tuple[datetime, datetime]:
    start_raw, _, duration_raw = valid_time.partition('/')
    start = _dt(start_raw)
    end = start + (_duration(duration_raw) if duration_raw else timedelta(hours=1))
    return start, end


def _value_at(
    series: dict[str, Any] | None,
    when: datetime,
    max_nearest_distance: timedelta = MAX_NEAREST_DISTANCE,
) -> Any:
    """Return a value valid at ``when`` or a nearby value within a tight guard.

    Older behavior accepted the nearest non-null grid value regardless of how
    far it was from kickoff. That could silently turn an out-of-horizon grid
    into a seemingly valid kickoff forecast. Containing intervals still win;
    nearest-value fallback is now bounded to three hours by default.
    """
    if not series:
        return None
    values = series.get('values') or []
    best: tuple[float, Any] | None = None
    for item in values:
        if not item.get('validTime'):
            continue
        start, end = _interval(item['validTime'])
        value = item.get('value')
        if start <= when < end:
            return value
        if value is None:
            continue
        distance = min(abs((start - when).total_seconds()), abs((end - when).total_seconds()))
        if best is None or distance < best[0]:
            best = (distance, value)
    if best is None or best[0] > max_nearest_distance.total_seconds():
        return None
    return best[1]


def _c_to_f(value: Any) -> float | None:
    try:
        return float(value) * 9 / 5 + 32
    except (TypeError, ValueError):
        return None


def _kph_to_mph(value: Any) -> float | None:
    try:
        return float(value) * 0.621371
    except (TypeError, ValueError):
        return None


def _mm_to_inches(value: Any) -> float | None:
    try:
        return float(value) / 25.4
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _weather_label(pop: float | None, qpf: float | None, snow: float | None) -> str:
    if snow is not None and snow > 0.01:
        return 'Snow/wintry precipitation possible'
    if qpf is not None and qpf > 0.01:
        return 'Measurable precipitation forecast'
    if pop is not None and pop >= 50:
        return 'Precipitation possible'
    if pop is not None and pop >= 25:
        return 'Low precipitation chance'
    return 'No meaningful precipitation signal'


class NWSClient:
    def __init__(self, cache_path: Path = CACHE_PATH) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'cfb-weather-totals/1.0 (github.com/JacZerin24/cfb-weather-totals)',
            'Accept': 'application/geo+json',
        })
        adapter = HTTPAdapter(max_retries=build_nws_retry_policy())
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.cache_dirty = False

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def save_cache(self) -> None:
        if not self.cache_dirty:
            return
        ensure_dir('outputs')
        self.cache_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True), encoding='utf-8')
        self.cache_dirty = False

    def _get(self, url: str) -> dict[str, Any]:
        response = self.session.get(
            url,
            timeout=(NWS_CONNECT_TIMEOUT_SECONDS, NWS_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _cache_key(latitude: float, longitude: float) -> str:
        return f'{latitude:.4f},{longitude:.4f}'

    def grid_for_point(self, latitude: float, longitude: float) -> dict[str, Any]:
        key = self._cache_key(latitude, longitude)
        cached = self.cache.get(key)
        if cached and cached.get('grid_data_url'):
            try:
                checked = _dt(cached.get('checked_at', '1970-01-01T00:00:00Z'))
            except Exception:
                checked = datetime(1970, 1, 1, tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - checked < timedelta(days=30):
                return cached

        points = self._get(f'{NWS_BASE}/points/{latitude:.4f},{longitude:.4f}')
        props = points.get('properties') or {}
        grid = {
            'office': props.get('gridId'),
            'grid_x': props.get('gridX'),
            'grid_y': props.get('gridY'),
            'grid_data_url': props.get('forecastGridData'),
            'forecast_url': props.get('forecast'),
            'forecast_hourly_url': props.get('forecastHourly'),
            'checked_at': datetime.now(timezone.utc).isoformat(),
        }
        if not grid['grid_data_url']:
            raise RuntimeError('NWS point lookup did not return forecastGridData.')
        self.cache[key] = grid
        self.cache_dirty = True
        return grid

    def kickoff_forecast(self, latitude: float, longitude: float, kickoff: datetime) -> dict[str, Any]:
        kickoff = kickoff.astimezone(timezone.utc)
        grid = self.grid_for_point(latitude, longitude)
        payload = self._get(grid['grid_data_url'])
        props = payload.get('properties') or {}

        temperature_f = _c_to_f(_value_at(props.get('temperature'), kickoff))
        dewpoint_f = _c_to_f(_value_at(props.get('dewpoint'), kickoff))
        humidity = _number(_value_at(props.get('relativeHumidity'), kickoff))
        wind_mph = _kph_to_mph(_value_at(props.get('windSpeed'), kickoff))
        gust_mph = _kph_to_mph(_value_at(props.get('windGust'), kickoff))
        pop = _number(_value_at(props.get('probabilityOfPrecipitation'), kickoff))
        precipitation = _mm_to_inches(_value_at(props.get('quantitativePrecipitation'), kickoff))
        snowfall = _mm_to_inches(_value_at(props.get('snowfallAmount'), kickoff))
        wind_direction = _number(_value_at(props.get('windDirection'), kickoff))

        # The live model must not treat an out-of-horizon or badly incomplete
        # grid as forecast-ready merely because a request returned HTTP 200.
        forecast_ready = temperature_f is not None and wind_mph is not None
        status = 'ok' if forecast_ready else 'incomplete_forecast'
        summary = (
            _weather_label(pop, precipitation, snowfall)
            if forecast_ready
            else 'NWS grid is available, but kickoff temperature/wind are not yet within the usable forecast horizon.'
        )

        return {
            'nws_status': status,
            'nws_office': grid.get('office'),
            'nws_grid_x': grid.get('grid_x'),
            'nws_grid_y': grid.get('grid_y'),
            'nws_grid_data_url': grid.get('grid_data_url'),
            'temperature_f': temperature_f,
            'dewpoint_f': dewpoint_f,
            'humidity': humidity,
            'wind_mph': wind_mph,
            'wind_gust_mph': gust_mph,
            'precip_probability_pct': pop,
            'precipitation': precipitation,
            'snowfall': snowfall,
            'wind_direction_degrees': wind_direction,
            'weather_summary': summary,
        }
