from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml

from ..utils import ensure_dir, read_df, write_df
from .forecast_native_bakeoff import (
    DATA_PATH,
    _feature_columns,
    _weather_features,
)
from .model_bakeoff import _prep
from .roi_search import _deep_models


SCHEDULE_URL = (
    'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
)
STADIUMS_URL = (
    'https://raw.githubusercontent.com/greerreNFL/Stadiums/'
    'main/data/stadiums.csv'
)
PREVIOUS_RUNS_URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'
ODDS_URL = (
    'https://api.the-odds-api.com/v4/sports/'
    'americanfootball_nfl/odds'
)
PROTOCOL_PATH = Path('config/nfl_frozen_protocol.yml')
OUTPUT_DIR = Path('outputs/nfl/live')
EASTERN = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')
FORECAST_MODEL = 'jma_gsm'
LEAD_HOURS = 24
DECISION_WINDOW_TOLERANCE_HOURS = 1.6

# These venues have retractable roofs. The public stadium reference describes
# the underlying stadium shell as Outdoors, so the decision-time policy must
# explicitly prevent the eventual open/closed roof decision from leaking into
# the official model.
RETRACTABLE_STADIUM_IDS = {
    'ATL97',  # Mercedes-Benz Stadium
    'DAL00',  # AT&T Stadium
    'HOU00',  # NRG / Reliant Stadium
    'IND00',  # Lucas Oil Stadium
    'PHO00',  # State Farm Stadium
}

TEAM_NAMES = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs',
    'LA': 'Los Angeles Rams',
    'LAC': 'Los Angeles Chargers',
    'LV': 'Las Vegas Raiders',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NO': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SEA': 'Seattle Seahawks',
    'SF': 'San Francisco 49ers',
    'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders',
}

BOARD_COLUMNS = [
    'status', 'model_signal', 'decision_state', 'decision_due',
    'hours_to_kickoff', 'hours_from_24h_target',
    'season', 'week', 'game_id', 'kickoff_utc', 'gameday', 'gametime',
    'away_team', 'home_team', 'stadium', 'stadium_id',
    'decision_roof', 'surface', 'lat', 'lon',
    'closing_total', 'consensus_over_price', 'best_over_price_same_line',
    'sportsbooks_at_selected_line', 'sportsbooks_in_snapshot',
    'market_total_min', 'market_total_max', 'market_total_range',
    'odds_event_id', 'odds_market_last_update',
    'odds_request_remaining', 'odds_request_used', 'odds_request_last',
    'forecast_model', 'forecast_temp_24h', 'forecast_wind_24h',
    'forecast_temp_48h', 'forecast_wind_48h',
    'forecast_temp_72h', 'forecast_wind_72h',
    'forecast_complete_24h',
    'pred_market_residual', 'over_probability', 'model_projected_total',
    'qualifier_threshold', 'strong_threshold', 'probability_threshold',
]


def _download_csv(url: str) -> pd.DataFrame:
    response = requests.get(
        url,
        timeout=60,
        headers={'User-Agent': 'nfl-weather-totals-live/1.0'},
    )
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def _load_protocol() -> dict[str, Any]:
    data = yaml.safe_load(PROTOCOL_PATH.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise RuntimeError('NFL frozen protocol is not a YAML mapping.')
    if data.get('protocol', {}).get('id') != 'nfl_totals_paper_v1':
        raise RuntimeError('Unexpected NFL frozen protocol id.')
    if data.get('protocol', {}).get('status') != (
        'frozen_for_prospective_paper_tracking'
    ):
        raise RuntimeError('NFL protocol is not frozen for paper tracking.')
    return data


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


def _active_regular_season(
    schedule: pd.DataFrame,
    now: pd.Timestamp,
) -> pd.DataFrame:
    work = schedule.copy()
    work['season'] = pd.to_numeric(work['season'], errors='coerce')
    work['week'] = pd.to_numeric(work['week'], errors='coerce')
    work = work[work['game_type'].astype(str).str.upper().eq('REG')].copy()
    work['kickoff_utc'] = [
        _kickoff_utc(day, time)
        for day, time in zip(work['gameday'], work['gametime'])
    ]
    work = work[work['kickoff_utc'].notna()].copy()

    current_year = int(now.year)
    candidates = work[work['season'].isin([current_year - 1, current_year])].copy()
    future = candidates[candidates['kickoff_utc'] > now].copy()
    if future.empty:
        return future

    season = int(future.sort_values('kickoff_utc').iloc[0]['season'])
    future = future[future['season'].eq(season)].copy()
    target_week = int(
        pd.to_numeric(future['week'], errors='coerce').dropna().min()
    )
    return (
        future[future['week'].eq(target_week)]
        .sort_values('kickoff_utc')
        .reset_index(drop=True)
    )


def _stadiums() -> pd.DataFrame:
    stadiums = _download_csv(STADIUMS_URL).copy()
    stadiums['stadium_id'] = stadiums['stadium_id'].astype(str)
    stadiums['lat'] = pd.to_numeric(stadiums['lat'], errors='coerce')
    stadiums['lon'] = pd.to_numeric(stadiums['lon'], errors='coerce')
    keep = [
        'stadium_id', 'stadium_name', 'lat', 'lon',
        'surface_type', 'roof_type', 'tz',
    ]
    return stadiums[keep].drop_duplicates('stadium_id')


def _decision_roof(row: pd.Series) -> str:
    stadium_id = str(row.get('stadium_id') or '')
    if stadium_id in RETRACTABLE_STADIUM_IDS:
        return 'retractable'

    schedule_roof = str(row.get('roof') or '').strip().lower()
    reference_roof = str(row.get('roof_type') or '').strip().lower()
    if schedule_roof == 'dome' or reference_roof == 'dome':
        return 'dome'
    if schedule_roof in {'outdoors', 'outdoor'}:
        return 'outdoors'
    if reference_roof in {'outdoors', 'outdoor', 'open'}:
        return 'outdoors'
    if schedule_roof in {'open', 'closed'}:
        # Unknown retractable venue not in the explicit list: withhold weather
        # rather than treating the eventual roof position as a model feature.
        return 'retractable'
    return 'unknown'


def _attach_venue_context(board: pd.DataFrame) -> pd.DataFrame:
    out = board.copy()
    out['stadium_id'] = out['stadium_id'].astype(str)
    out = out.merge(
        _stadiums(),
        on='stadium_id',
        how='left',
        suffixes=('', '_reference'),
    )
    out['decision_roof'] = out.apply(_decision_roof, axis=1)
    out['surface'] = out['surface'].where(
        out['surface'].notna(),
        out.get('surface_type'),
    )
    return out


def _odds_events() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = os.getenv('ODDS_API_KEY', '').strip()
    if not key:
        raise RuntimeError('ODDS_API_KEY is required for the live NFL pipeline.')

    response = requests.get(
        ODDS_URL,
        params={
            'apiKey': key,
            'regions': 'us',
            'markets': 'totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso',
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError('Unexpected The Odds API response type.')

    quota = {
        'odds_request_remaining': response.headers.get('x-requests-remaining'),
        'odds_request_used': response.headers.get('x-requests-used'),
        'odds_request_last': response.headers.get('x-requests-last'),
    }
    return payload, quota


def _select_total_market(event: dict[str, Any]) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for book in event.get('bookmakers') or []:
        book_name = str(book.get('title') or book.get('key') or 'unknown')
        for market in book.get('markets') or []:
            if market.get('key') != 'totals':
                continue
            market_update = market.get('last_update')
            for outcome in market.get('outcomes') or []:
                if str(outcome.get('name')).lower() != 'over':
                    continue
                try:
                    point = float(outcome.get('point'))
                    price = float(outcome.get('price'))
                except (TypeError, ValueError):
                    continue
                rows.append({
                    'sportsbook': book_name,
                    'line': point,
                    'price': price,
                    'last_update': market_update,
                })

    if not rows:
        return None

    frame = pd.DataFrame(rows)
    counts = frame['line'].value_counts()
    max_count = int(counts.max())
    candidates = [float(v) for v in counts[counts.eq(max_count)].index]
    market_median = float(frame['line'].median())
    selected_line = min(
        candidates,
        key=lambda value: (abs(value - market_median), value),
    )
    at_line = frame[np.isclose(frame['line'], selected_line)].copy()
    if at_line.empty:
        return None

    median_price = float(at_line['price'].median())
    representative_idx = (
        at_line['price'] - median_price
    ).abs().idxmin()
    representative_price = float(at_line.loc[representative_idx, 'price'])
    latest_update = pd.to_datetime(
        at_line['last_update'], utc=True, errors='coerce'
    ).max()

    return {
        'closing_total': selected_line,
        'consensus_over_price': representative_price,
        'best_over_price_same_line': float(at_line['price'].max()),
        'sportsbooks_at_selected_line': int(at_line['sportsbook'].nunique()),
        'sportsbooks_in_snapshot': int(frame['sportsbook'].nunique()),
        'market_total_min': float(frame['line'].min()),
        'market_total_max': float(frame['line'].max()),
        'market_total_range': float(frame['line'].max() - frame['line'].min()),
        'odds_market_last_update': (
            latest_update.isoformat()
            if pd.notna(latest_update) else None
        ),
    }


def _attach_market(
    board: pd.DataFrame,
    events: list[dict[str, Any]],
    quota: dict[str, Any],
) -> pd.DataFrame:
    event_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        key = (
            str(event.get('home_team') or ''),
            str(event.get('away_team') or ''),
        )
        event_lookup.setdefault(key, []).append(event)

    rows = []
    for _, game in board.iterrows():
        home = TEAM_NAMES.get(str(game['home_team']))
        away = TEAM_NAMES.get(str(game['away_team']))
        matches = event_lookup.get((home or '', away or ''), [])
        best_event = None
        best_distance = np.inf
        kickoff = pd.to_datetime(game['kickoff_utc'], utc=True, errors='coerce')

        for event in matches:
            event_time = pd.to_datetime(
                event.get('commence_time'), utc=True, errors='coerce'
            )
            if pd.isna(kickoff) or pd.isna(event_time):
                continue
            distance = abs((event_time - kickoff).total_seconds()) / 3600.0
            if distance < best_distance:
                best_event = event
                best_distance = distance

        market = _select_total_market(best_event) if (
            best_event is not None and best_distance <= 8
        ) else None

        row = {'game_id': game['game_id'], **quota}
        if best_event is not None:
            row['odds_event_id'] = best_event.get('id')
        if market:
            row.update(market)
        rows.append(row)

    return board.merge(pd.DataFrame(rows), on='game_id', how='left')


def _hourly_value(
    hourly: dict[str, Any],
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
    diffs = np.abs(
        (pd.Series(ts) - kickoff).dt.total_seconds().to_numpy()
    )
    if not np.isfinite(diffs).any():
        return np.nan
    idx = int(np.nanargmin(diffs))
    if diffs[idx] > 90 * 60:
        return np.nan
    value = vals.iloc[idx]
    return float(value) if pd.notna(value) else np.nan


def _fetch_fixed_lead_weather(board: pd.DataFrame) -> pd.DataFrame:
    outdoor = board[
        board['decision_roof'].eq('outdoors')
        & board['lat'].notna()
        & board['lon'].notna()
    ].copy()
    if outdoor.empty:
        return pd.DataFrame(columns=['game_id'])

    locations = (
        outdoor[['stadium_id', 'lat', 'lon']]
        .drop_duplicates('stadium_id')
        .reset_index(drop=True)
    )
    min_date = pd.to_datetime(outdoor['gameday']).min().date().isoformat()
    max_date = (
        pd.to_datetime(outdoor['gameday']).max() + pd.Timedelta(days=1)
    ).date().isoformat()

    variables = []
    for day in (1, 2, 3):
        variables.extend([
            f'temperature_2m_previous_day{day}',
            f'wind_speed_10m_previous_day{day}',
        ])

    response = requests.get(
        PREVIOUS_RUNS_URL,
        params={
            'latitude': ','.join(
                f'{value:.6f}' for value in locations['lat']
            ),
            'longitude': ','.join(
                f'{value:.6f}' for value in locations['lon']
            ),
            'start_date': min_date,
            'end_date': max_date,
            'hourly': ','.join(variables),
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'timezone': 'UTC',
            'models': FORECAST_MODEL,
        },
        timeout=120,
        headers={'User-Agent': 'nfl-weather-totals-live/1.0'},
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        payload = [payload]
    if len(payload) != len(locations):
        raise RuntimeError(
            'Open-Meteo multi-location response does not match locations.'
        )

    by_stadium = {
        str(locations.iloc[i]['stadium_id']): payload[i]
        for i in range(len(locations))
    }

    rows = []
    for _, game in outdoor.iterrows():
        item = by_stadium.get(str(game['stadium_id']), {})
        hourly = item.get('hourly') or {}
        row = {
            'game_id': game['game_id'],
            'forecast_model': FORECAST_MODEL,
        }
        for day in (1, 2, 3):
            lead = day * 24
            row[f'forecast_temp_{lead}h'] = _hourly_value(
                hourly,
                f'temperature_2m_previous_day{day}',
                game['kickoff_utc'],
            )
            row[f'forecast_wind_{lead}h'] = _hourly_value(
                hourly,
                f'wind_speed_10m_previous_day{day}',
                game['kickoff_utc'],
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_weather(board: pd.DataFrame) -> pd.DataFrame:
    out = board.copy()
    weather = _fetch_fixed_lead_weather(out)
    if not weather.empty:
        out = out.merge(weather, on='game_id', how='left')
    else:
        out['forecast_model'] = np.nan

    for lead in (24, 48, 72):
        temp = f'forecast_temp_{lead}h'
        wind = f'forecast_wind_{lead}h'
        if temp not in out.columns:
            out[temp] = np.nan
        if wind not in out.columns:
            out[wind] = np.nan

    outdoor = out['decision_roof'].eq('outdoors')
    complete = (
        pd.to_numeric(out['forecast_temp_24h'], errors='coerce').notna()
        & pd.to_numeric(out['forecast_wind_24h'], errors='coerce').notna()
    )
    out['forecast_complete_24h'] = np.where(
        outdoor,
        complete,
        True,
    )
    out.loc[~outdoor, 'forecast_model'] = FORECAST_MODEL
    return out


def _live_categories(board: pd.DataFrame) -> pd.DataFrame:
    out = board.copy()
    out['closing_total'] = pd.to_numeric(
        out['closing_total'], errors='coerce'
    )
    out['month'] = pd.to_datetime(
        out['gameday'], errors='coerce'
    ).dt.month
    out['total_bin'] = pd.cut(
        out['closing_total'],
        bins=[0, 38, 42, 46, 50, 54, 100],
        labels=['<=38', '38-42', '42-46', '46-50', '50-54', '54+'],
    )
    return out


def _score(board: pd.DataFrame) -> pd.DataFrame:
    historical = read_df(DATA_PATH).copy()
    historical['season'] = pd.to_numeric(
        historical['season'], errors='coerce'
    )
    historical = historical[
        historical['forecast_complete_24h'].fillna(False)
        & historical['market_residual'].notna()
        & historical['season'].le(2025)
    ].copy()
    if historical.empty:
        raise RuntimeError('NFL live training dataset is empty.')

    train = _weather_features(historical, LEAD_HOURS)
    live = _weather_features(board, LEAD_HOURS)
    combined = pd.concat([train, live], ignore_index=True, sort=False)
    nums, cats = _feature_columns(
        combined,
        LEAD_HOURS,
        weather=True,
    )

    reg = _deep_models(nums, cats)['reg_rf_leaf25'][1]
    cls = _deep_models(nums, cats)['cls_rf_leaf25'][1]

    train_work = _prep(train, nums, cats)
    reg.fit(train_work[nums + cats], train_work['market_residual'])

    cls_train = train_work[train_work['market_residual'] != 0].copy()
    cls.fit(
        cls_train[nums + cats],
        (cls_train['market_residual'] > 0).astype(int),
    )

    out = live.copy()
    out['pred_market_residual'] = np.nan
    out['over_probability'] = np.nan
    scoreable = (
        out['closing_total'].notna()
        & out['forecast_complete_24h'].fillna(False)
    )
    if scoreable.any():
        score_work = _prep(out.loc[scoreable], nums, cats)
        out.loc[scoreable, 'pred_market_residual'] = reg.predict(
            score_work[nums + cats]
        )
        out.loc[scoreable, 'over_probability'] = cls.predict_proba(
            score_work[nums + cats]
        )[:, 1]

    out['model_projected_total'] = (
        out['closing_total'] + out['pred_market_residual']
    )
    return out


def _classify(
    board: pd.DataFrame,
    protocol: dict[str, Any],
    now: pd.Timestamp,
) -> pd.DataFrame:
    out = board.copy()
    qualifier_edge = float(
        protocol['qualifier']['minimum_regression_edge_points']
    )
    strong_edge = float(
        protocol['strong']['minimum_regression_edge_points']
    )
    probability = float(
        protocol['qualifier']['minimum_over_probability']
    )

    out['qualifier_threshold'] = qualifier_edge
    out['strong_threshold'] = strong_edge
    out['probability_threshold'] = probability
    out['hours_to_kickoff'] = (
        pd.to_datetime(out['kickoff_utc'], utc=True) - now
    ).dt.total_seconds() / 3600.0
    out['hours_from_24h_target'] = (
        out['hours_to_kickoff'] - LEAD_HOURS
    )
    out['decision_due'] = (
        out['hours_from_24h_target'].abs()
        <= DECISION_WINDOW_TOLERANCE_HOURS
    )

    model_signal = np.full(len(out), 'NO PLAY', dtype=object)
    valid_model = (
        out['pred_market_residual'].notna()
        & out['over_probability'].notna()
    )
    qualifies = (
        valid_model
        & out['pred_market_residual'].ge(qualifier_edge)
        & out['over_probability'].ge(probability)
    )
    strong = (
        qualifies
        & out['pred_market_residual'].ge(strong_edge)
    )
    model_signal[qualifies.to_numpy()] = 'QUALIFIES'
    model_signal[strong.to_numpy()] = 'STRONG'
    out['model_signal'] = model_signal

    states = []
    statuses = []
    for _, row in out.iterrows():
        hours = float(row['hours_to_kickoff'])
        if pd.isna(row.get('closing_total')):
            state = 'WAITING_FOR_MARKET'
            status = 'NO LINE'
        elif not bool(row.get('forecast_complete_24h')):
            state = 'WAITING_FOR_24H_FORECAST'
            status = 'WAITING'
        elif bool(row.get('decision_due')):
            state = '24H_DECISION_WINDOW'
            status = str(row.get('model_signal') or 'NO PLAY')
        elif hours > LEAD_HOURS + DECISION_WINDOW_TOLERANCE_HOURS:
            state = 'BEFORE_24H_WINDOW'
            status = 'EARLY LOOK'
        else:
            state = 'PAST_24H_WINDOW'
            status = 'PAST WINDOW'
        states.append(state)
        statuses.append(status)

    out['decision_state'] = states
    out['status'] = statuses
    return out


def _write_outputs(
    board: pd.DataFrame,
    now: pd.Timestamp,
    protocol: dict[str, Any],
) -> None:
    ensure_dir(OUTPUT_DIR)
    output = board[[c for c in BOARD_COLUMNS if c in board.columns]].copy()
    write_df(output, OUTPUT_DIR / 'weekly_board.csv')

    snapshot = {
        'generated_at_utc': now.isoformat(),
        'protocol_id': protocol['protocol']['id'],
        'protocol_version': protocol['protocol']['version'],
        'official_lead_hours': protocol['decision']['official_lead_hours'],
        'paper_tracking_only': True,
        'production_enabled': False,
        'season': (
            int(output['season'].dropna().iloc[0])
            if 'season' in output and output['season'].notna().any()
            else None
        ),
        'week': (
            int(output['week'].dropna().iloc[0])
            if 'week' in output and output['week'].notna().any()
            else None
        ),
        'game_count': int(len(output)),
        'decision_due_count': int(
            output.get('decision_due', pd.Series(dtype=bool))
            .fillna(False)
            .sum()
        ),
        'qualifies_count': int(
            output.get('status', pd.Series(dtype=str))
            .eq('QUALIFIES')
            .sum()
        ),
        'strong_count': int(
            output.get('status', pd.Series(dtype=str))
            .eq('STRONG')
            .sum()
        ),
        'games': json.loads(
            output.to_json(orient='records', date_format='iso')
        ),
    }
    (OUTPUT_DIR / 'weekly_snapshot.json').write_text(
        json.dumps(snapshot, indent=2),
        encoding='utf-8',
    )

    visible = [
        c for c in [
            'status', 'decision_state', 'week',
            'away_team', 'home_team', 'kickoff_utc',
            'closing_total', 'consensus_over_price',
            'forecast_temp_24h', 'forecast_wind_24h',
            'pred_market_residual', 'over_probability',
            'model_projected_total',
        ]
        if c in output.columns
    ]
    lines = [
        '# NFL Live Weekly Paper Board',
        '',
        f'Generated: {now.isoformat()}',
        '',
        f'Frozen protocol: {protocol["protocol"]["id"]} v{protocol["protocol"]["version"]}',
        '',
        'This board is operational paper-mode output only. It does not create the immutable prospective ledger; that is the next build stage.',
        '',
        output[visible].to_markdown(index=False)
        if not output.empty else '_No upcoming regular-season games._',
    ]
    (OUTPUT_DIR / 'weekly_report.md').write_text(
        '\n'.join(lines),
        encoding='utf-8',
    )


def main() -> None:
    protocol = _load_protocol()
    now = pd.Timestamp.now(tz='UTC')

    schedule = _download_csv(SCHEDULE_URL)
    board = _active_regular_season(schedule, now)
    if board.empty:
        ensure_dir(OUTPUT_DIR)
        empty = pd.DataFrame(columns=BOARD_COLUMNS)
        _write_outputs(empty, now, protocol)
        print('No upcoming NFL regular-season games.')
        return

    board = _attach_venue_context(board)

    events, quota = _odds_events()
    board = _attach_market(board, events, quota)

    board = _attach_weather(board)
    board = _live_categories(board)
    board = _score(board)
    board = _classify(board, protocol, now)
    _write_outputs(board, now, protocol)

    print(
        f'Wrote NFL live week {int(board["week"].iloc[0])}: '
        f'{len(board)} games, '
        f'{int(board["decision_due"].sum())} in 24h decision window, '
        f'{int(board["status"].eq("QUALIFIES").sum())} QUALIFIES, '
        f'{int(board["status"].eq("STRONG").sum())} STRONG.'
    )


if __name__ == '__main__':
    main()
