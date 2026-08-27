from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import requests

from .odds_api_fallback import _kickoff_distance_hours, _provider_rank, _team_score

BASE_URL = 'https://api.oddspapi.io/v4'
AMERICAN_FOOTBALL_SPORT_ID = 14
NCAA_TOURNAMENT_ID = 27653


def _api_get(path: str, api_key: str, **params: Any) -> tuple[Any | None, str]:
    payload_params = {**params, 'apiKey': api_key}
    for attempt in range(3):
        try:
            response = requests.get(
                f'{BASE_URL}/{path.lstrip("/")}',
                params=payload_params,
                timeout=90,
            )
        except requests.RequestException:
            return None, 'request_error'

        if response.status_code == 429 and attempt < 2:
            retry_ms = 1200
            try:
                retry_ms = int(response.json().get('error', {}).get('retryMs') or retry_ms)
            except (ValueError, TypeError, AttributeError):
                pass
            time.sleep(min(max(retry_ms / 1000.0 + 0.25, 1.1), 5.0))
            continue
        if response.status_code != 200:
            return None, f'http_{response.status_code}'
        try:
            return response.json(), 'ok'
        except ValueError:
            return None, 'invalid_json'
    return None, 'rate_limited'


def _active_subscription(account: Any) -> dict[str, Any]:
    if not isinstance(account, dict):
        return {}
    subscriptions = account.get('subscriptions') or []
    if not isinstance(subscriptions, list):
        return {}
    active = [s for s in subscriptions if isinstance(s, dict) and s.get('is_active')]
    if active:
        return active[-1]
    current_id = account.get('current_subscription_id')
    for sub in subscriptions:
        if isinstance(sub, dict) and sub.get('subscription_id') == current_id:
            return sub
    return subscriptions[-1] if subscriptions and isinstance(subscriptions[-1], dict) else {}


def _as_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    if payload.get('fixtureId'):
        return [payload]
    for key in ('fixtures', 'data', 'results', 'odds'):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    values = [value for value in payload.values() if isinstance(value, dict) and value.get('fixtureId')]
    return values


def _date_window(board: pd.DataFrame) -> tuple[str | None, str | None]:
    if 'start_date' not in board.columns:
        return None, None
    dates = pd.to_datetime(board['start_date'], utc=True, errors='coerce').dropna()
    if dates.empty:
        return None, None
    start = (dates.min() - pd.Timedelta(hours=18)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end = (dates.max() + pd.Timedelta(hours=18)).strftime('%Y-%m-%dT%H:%M:%SZ')
    return start, end


def _market_lookup(catalog: Any) -> dict[str, dict[str, Any]]:
    rows = catalog if isinstance(catalog, list) else []
    return {
        str(row.get('marketId')): row
        for row in rows
        if isinstance(row, dict) and row.get('marketId') is not None
    }


def _is_full_game_total(info: dict[str, Any]) -> bool:
    name = str(info.get('marketName') or '').strip().lower()
    market_type = str(info.get('marketType') or '').strip().lower()
    period = str(info.get('period') or '').strip().lower()
    player_prop = bool(info.get('playerProp', False))
    handicap = pd.to_numeric(pd.Series([info.get('handicap')]), errors='coerce').iloc[0]
    if player_prop or pd.isna(handicap) or float(handicap) < 20 or float(handicap) > 100:
        return False
    if market_type and market_type != 'totals':
        return False
    if period and period not in {'fulltime', 'full_time', 'game', 'match'}:
        return False
    # OddsPapi's documented NCAA market name is "Total (incl. overtime)".
    return 'total' in name and 'team total' not in name and ('overtime' in name or 'incl' in name)


def _player_node(outcome: Any) -> dict[str, Any] | None:
    if not isinstance(outcome, dict):
        return None
    players = outcome.get('players')
    if isinstance(players, dict):
        node = players.get('0')
        if isinstance(node, dict):
            return node
        for value in players.values():
            if isinstance(value, dict):
                return value
    if isinstance(players, list):
        for value in players:
            if isinstance(value, dict):
                return value
    return None


def _market_has_two_live_sides(market: Any) -> bool:
    if not isinstance(market, dict) or market.get('marketActive') is False:
        return False
    outcomes = market.get('outcomes') or {}
    if not isinstance(outcomes, dict):
        return False
    active = 0
    for outcome in outcomes.values():
        node = _player_node(outcome)
        if node and node.get('active') is True and pd.notna(node.get('price')):
            active += 1
    return active >= 2


def select_oddspapi_total(
    odds_fixture: dict[str, Any],
    market_catalog: Any,
    preferred_providers: list[str] | None = None,
) -> dict[str, Any] | None:
    lookup = _market_lookup(market_catalog)
    books = odds_fixture.get('bookmakerOdds') or {}
    if not isinstance(books, dict):
        return None

    book_lines: dict[str, set[float]] = {}
    for slug, book in books.items():
        if not isinstance(book, dict):
            continue
        if book.get('bookmakerIsActive') is False or book.get('suspended') is True:
            continue
        markets = book.get('markets') or {}
        if not isinstance(markets, dict):
            continue
        for market_id, market in markets.items():
            info = lookup.get(str(market_id), {})
            if not _is_full_game_total(info) or not _market_has_two_live_sides(market):
                continue
            line = pd.to_numeric(pd.Series([info.get('handicap')]), errors='coerce').iloc[0]
            if pd.isna(line):
                continue
            book_lines.setdefault(str(slug), set()).add(float(line))

    if not book_lines:
        return None

    line_counts: Counter[float] = Counter()
    for lines in book_lines.values():
        for line in lines:
            line_counts[line] += 1
    if not line_counts:
        return None

    all_quotes = [line for lines in book_lines.values() for line in lines]
    center = float(np.median(all_quotes))
    consensus = sorted(
        line_counts,
        key=lambda line: (-line_counts[line], abs(line - center), line),
    )[0]

    # Give every bookmaker one representative live line nearest the market mode.
    representatives = {
        slug: min(lines, key=lambda line: (abs(line - consensus), line))
        for slug, lines in book_lines.items()
    }
    rep_values = list(representatives.values())
    market_median = float(np.median(rep_values))

    preferred = list(preferred_providers or [])
    consensus_books = [slug for slug, lines in book_lines.items() if consensus in lines]
    provider = 'OddsPapi consensus'
    if consensus_books:
        ranked = sorted(consensus_books, key=lambda slug: (_provider_rank(slug, preferred), slug))
        provider = f'{ranked[0]} (OddsPapi)'

    return {
        'closing_total': float(consensus),
        'line_provider': provider,
        'line_provider_count': int(len(representatives)),
        'line_total_min': float(min(rep_values)),
        'line_total_max': float(max(rep_values)),
        'line_total_median': market_median,
        'line_total_range': float(max(rep_values) - min(rep_values)),
        'selected_vs_market_median': float(consensus - market_median),
        'line_source': 'OddsPapi',
        'oddspapi_consensus_book_count': int(line_counts[consensus]),
    }


def match_oddspapi_fixture(
    game: pd.Series,
    fixtures: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    best: dict[str, Any] | None = None
    best_score = 0.0
    for fixture in fixtures:
        if int(fixture.get('tournamentId') or 0) != NCAA_TOURNAMENT_ID:
            continue
        hours = _kickoff_distance_hours(game.get('start_date'), fixture.get('startTime'))
        if hours > 36:
            continue
        p1 = fixture.get('participant1Name') or fixture.get('participant1') or ''
        p2 = fixture.get('participant2Name') or fixture.get('participant2') or ''
        if not p1 or not p2:
            continue

        direct_home = _team_score(game.get('home_team'), p1)
        direct_away = _team_score(game.get('away_team'), p2)
        reverse_home = _team_score(game.get('home_team'), p2)
        reverse_away = _team_score(game.get('away_team'), p1)
        direct = (direct_home + direct_away) / 2.0 if min(direct_home, direct_away) >= 0.72 else 0.0
        reverse = (reverse_home + reverse_away) / 2.0 if min(reverse_home, reverse_away) >= 0.72 else 0.0
        score = max(direct, reverse)
        if score <= 0:
            continue
        score -= min(hours, 24.0) / 2400.0
        if score > best_score:
            best = fixture
            best_score = score

    if best_score < 0.82:
        return None, best_score
    return best, best_score


def fetch_oddspapi_ncaa(
    board: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, dict[str, Any]]:
    key = os.getenv('ODDSPAPI_API_KEY', '').strip()
    stats: dict[str, Any] = {
        'oddspapi_status': 'missing_key' if not key else 'starting',
        'oddspapi_request_limit': None,
        'oddspapi_request_count': None,
        'oddspapi_fixtures': 0,
        'oddspapi_odds_fixtures': 0,
    }
    if not key:
        return [], [], [], stats

    account, account_status = _api_get('account', key)
    stats['oddspapi_account_status'] = account_status
    if account_status != 'ok':
        stats['oddspapi_status'] = account_status
        return [], [], [], stats
    subscription = _active_subscription(account)
    stats['oddspapi_request_limit'] = subscription.get('request_limit')
    stats['oddspapi_request_count'] = subscription.get('request_count')

    start, end = _date_window(board)
    fixture_params: dict[str, Any] = {
        'tournamentId': NCAA_TOURNAMENT_ID,
        'statusId': 0,
        'language': 'en',
    }
    if start:
        fixture_params['from'] = start
    if end:
        fixture_params['to'] = end
    fixtures_payload, fixtures_status = _api_get('fixtures', key, **fixture_params)
    stats['oddspapi_fixtures_status'] = fixtures_status
    fixtures = _as_records(fixtures_payload)
    stats['oddspapi_fixtures'] = len(fixtures)
    if fixtures_status != 'ok':
        stats['oddspapi_status'] = fixtures_status
        return [], [], [], stats

    catalog, catalog_status = _api_get('markets', key, language='en')
    stats['oddspapi_markets_status'] = catalog_status
    if catalog_status != 'ok':
        stats['oddspapi_status'] = catalog_status
        return fixtures, [], [], stats

    odds_payload, odds_status = _api_get(
        'odds-by-tournaments',
        key,
        tournamentIds=str(NCAA_TOURNAMENT_ID),
        language='en',
        verbosity=3,
        oddsFormat='american',
    )
    stats['oddspapi_odds_status'] = odds_status
    odds_records = _as_records(odds_payload)
    stats['oddspapi_odds_fixtures'] = len(odds_records)
    stats['oddspapi_status'] = odds_status
    return fixtures, odds_records, catalog, stats


def apply_fcs_oddspapi_fallback(
    board: pd.DataFrame,
    preferred_providers: list[str] | None = None,
    fixtures: list[dict[str, Any]] | None = None,
    odds_records: list[dict[str, Any]] | None = None,
    market_catalog: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = board.copy()
    if 'line_source' not in out.columns:
        has_line = out.get('closing_total', pd.Series(np.nan, index=out.index)).notna()
        out['line_source'] = np.where(has_line, 'CFBD', '')

    division = out.get('division_track', pd.Series('', index=out.index)).astype(str).str.upper()
    missing = division.eq('FCS') & out.get('closing_total', pd.Series(np.nan, index=out.index)).isna()
    stats: dict[str, Any] = {
        'fcs_missing_before': int(missing.sum()),
        'fcs_oddspapi_filled': 0,
        'oddspapi_status': 'not_needed' if not missing.any() else 'not_called',
    }
    if not missing.any():
        return out, stats

    if fixtures is None or odds_records is None or market_catalog is None:
        fixtures, odds_records, market_catalog, fetched = fetch_oddspapi_ncaa(out.loc[missing].copy())
        stats.update(fetched)
    else:
        stats['oddspapi_status'] = 'fixture'

    if not fixtures or not odds_records or not market_catalog:
        stats['fcs_missing_after'] = int(missing.sum())
        return out, stats

    odds_by_fixture = {
        str(row.get('fixtureId')): row
        for row in odds_records
        if isinstance(row, dict) and row.get('fixtureId')
    }

    for idx, game in out.loc[missing].iterrows():
        fixture, confidence = match_oddspapi_fixture(game, fixtures)
        if fixture is None:
            continue
        fixture_id = str(fixture.get('fixtureId') or '')
        odds_fixture = odds_by_fixture.get(fixture_id)
        if not odds_fixture:
            continue
        market = select_oddspapi_total(odds_fixture, market_catalog, preferred_providers)
        if not market:
            continue
        for key, value in market.items():
            out.at[idx, key] = value
        out.at[idx, 'odds_match_confidence'] = round(float(confidence), 4)
        out.at[idx, 'odds_event_id'] = fixture_id
        stats['fcs_oddspapi_filled'] += 1

    stats['fcs_missing_after'] = int(
        (division.eq('FCS') & out.get('closing_total', pd.Series(np.nan, index=out.index)).isna()).sum()
    )
    return out, stats
