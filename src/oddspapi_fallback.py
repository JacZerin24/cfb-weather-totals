from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .odds_api_fallback import _kickoff_distance_hours, _provider_rank, _team_score

BASE_URL = 'https://api.oddspapi.io/v4'
AMERICAN_FOOTBALL_SPORT_ID = 14
NCAA_TOURNAMENT_ID = 27653
BULK_BOOKMAKER_CHUNK_SIZE = 1
COVERAGE_PATH = Path('outputs/oddspapi_bookmaker_coverage.csv')

# Live FCS coverage audit on 2026-08-27 found bet365 and Hard Rock Bet on
# all 30 FCS fixtures that had an active two-sided full-game total, while
# DraftKings and BetRivers each covered 28/30. Querying these four preserves
# the observed union while providing independent main-line corroboration and
# reducing a normal refresh from 19 OddsPapi requests to 6 total requests
# (fixtures + markets + four one-book odds calls).
NCAA_BOOKMAKER_PRIORITY = [
    'bet365', 'draftkings', 'betrivers', 'hardrockbet',
]


def _safe_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return str(response.text or '').strip()[:240]
    if isinstance(payload, dict):
        error = payload.get('error')
        if isinstance(error, dict):
            for key in ('message', 'detail', 'code'):
                if error.get(key):
                    return str(error[key])[:240]
        if error:
            return str(error)[:240]
        for key in ('message', 'detail', 'code'):
            if payload.get(key):
                return str(payload[key])[:240]
    return str(payload)[:240]


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
            return {'_http_error': _safe_error_detail(response)}, f'http_{response.status_code}'
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


def _subscription_bookmakers(subscription: dict[str, Any]) -> list[str]:
    books = subscription.get('bookmakers') or {}
    if isinstance(books, dict):
        return [str(slug) for slug in books if str(slug).strip()]
    if isinstance(books, list):
        return [str(slug) for slug in books if str(slug).strip()]
    return []


def _bulk_bookmakers(subscription: dict[str, Any]) -> list[str]:
    allowed = set(_subscription_bookmakers(subscription))
    if not allowed:
        return NCAA_BOOKMAKER_PRIORITY[:]
    selected = [slug for slug in NCAA_BOOKMAKER_PRIORITY if slug in allowed]
    if selected:
        return selected
    return sorted(allowed)[:20]


def _chunks(values: list[str], size: int = BULK_BOOKMAKER_CHUNK_SIZE) -> list[list[str]]:
    if size < 1:
        raise ValueError('Chunk size must be at least 1.')
    return [values[i:i + size] for i in range(0, len(values), size)]


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
    return [
        value for value in payload.values()
        if isinstance(value, dict) and value.get('fixtureId')
    ]


def _merge_odds_records(
    merged: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    """Merge disjoint one-book responses into one record per fixture."""
    for row in records:
        fixture_id = str(row.get('fixtureId') or '')
        if not fixture_id:
            continue
        if fixture_id not in merged:
            base = dict(row)
            base['bookmakerOdds'] = dict(row.get('bookmakerOdds') or {})
            merged[fixture_id] = base
            continue
        existing = merged[fixture_id]
        existing_books = existing.setdefault('bookmakerOdds', {})
        new_books = row.get('bookmakerOdds') or {}
        if isinstance(existing_books, dict) and isinstance(new_books, dict):
            existing_books.update(new_books)
        for key, value in row.items():
            if key != 'bookmakerOdds' and key not in existing:
                existing[key] = value


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
    """Identify a regulation/full-game total while excluding team/period totals.

    The live NCAA catalog labels the game-length period as ``result`` (for
    example, ``Total (incl. overtime)``), while other OddsPapi sports use names
    such as ``Over Under Full Time`` with period ``fulltime``. Rely primarily on
    market type + full-game period + plausible football total, not one exact
    display-name string.
    """
    name = str(info.get('marketName') or '').strip().lower()
    market_type = str(info.get('marketType') or '').strip().lower()
    period = str(info.get('period') or '').strip().lower()
    player_prop = bool(info.get('playerProp', False))
    handicap = pd.to_numeric(pd.Series([info.get('handicap')]), errors='coerce').iloc[0]

    if player_prop or pd.isna(handicap) or not 20 <= float(handicap) <= 100:
        return False
    if market_type != 'totals':
        return False
    if period not in {'result', 'fulltime', 'full_time', 'game', 'match'}:
        return False
    if 'team' in name:
        return False
    return 'total' in name or ('over' in name and 'under' in name)


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


def _valid_total_lines_by_book(
    odds_fixture: dict[str, Any],
    market_catalog: Any,
) -> dict[str, set[float]]:
    lookup = _market_lookup(market_catalog)
    books = odds_fixture.get('bookmakerOdds') or {}
    if not isinstance(books, dict):
        return {}

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
            if pd.notna(line):
                book_lines.setdefault(str(slug), set()).add(float(line))
    return book_lines


def select_oddspapi_total(
    odds_fixture: dict[str, Any],
    market_catalog: Any,
    preferred_providers: list[str] | None = None,
) -> dict[str, Any] | None:
    book_lines = _valid_total_lines_by_book(odds_fixture, market_catalog)
    if not book_lines:
        return None

    line_counts: Counter[float] = Counter()
    for lines in book_lines.values():
        for line in lines:
            line_counts[line] += 1
    if not line_counts:
        return None

    # Give each sportsbook one equal vote when locating the center. This keeps a
    # book that exposes a large alternate-total ladder (notably Hard Rock Bet)
    # from overpowering books that expose only their main or near-main total.
    book_centers = [float(np.median(sorted(lines))) for lines in book_lines.values() if lines]
    center = float(np.median(book_centers))
    consensus = sorted(
        line_counts,
        key=lambda line: (-line_counts[line], abs(line - center), line),
    )[0]

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
        'oddspapi_request_count_before': None,
        'oddspapi_request_count': None,
        'oddspapi_fixtures': 0,
        'oddspapi_fixtures_with_any_odds': 0,
        'oddspapi_odds_fixtures': 0,
        'oddspapi_bulk_bookmaker_count': 0,
        'oddspapi_bulk_calls': 0,
        'oddspapi_bulk_successes': 0,
        'oddspapi_bulk_failures': 0,
        'oddspapi_error_detail': '',
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
    stats['oddspapi_request_count_before'] = subscription.get('request_count')
    stats['oddspapi_request_count'] = subscription.get('request_count')
    bulk_books = _bulk_bookmakers(subscription)

    # Keep a reserve of at least five requests so scheduled runs cannot exhaust
    # the free account. Fixtures + markets cost two calls; each book costs one.
    try:
        request_limit = int(stats['oddspapi_request_limit'])
        request_count = int(stats['oddspapi_request_count_before'])
        max_bulk_calls = max(0, request_limit - request_count - 7)
        if max_bulk_calls < len(_chunks(bulk_books)):
            bulk_books = bulk_books[:max_bulk_calls * BULK_BOOKMAKER_CHUNK_SIZE]
    except (TypeError, ValueError):
        pass

    stats['oddspapi_bulk_bookmaker_count'] = len(bulk_books)
    bulk_chunks = _chunks(bulk_books)

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
    stats['oddspapi_fixtures_with_any_odds'] = sum(bool(row.get('hasOdds')) for row in fixtures)
    if fixtures_status != 'ok':
        stats['oddspapi_status'] = fixtures_status
        return [], [], [], stats

    catalog, catalog_status = _api_get('markets', key, language='en')
    stats['oddspapi_markets_status'] = catalog_status
    if catalog_status != 'ok':
        stats['oddspapi_status'] = catalog_status
        return fixtures, [], [], stats

    merged_odds: dict[str, dict[str, Any]] = {}
    bulk_errors: list[str] = []
    bulk_statuses: list[str] = []

    for chunk_index, bookmaker_chunk in enumerate(bulk_chunks):
        if chunk_index:
            time.sleep(1.05)
        stats['oddspapi_bulk_calls'] += 1
        odds_payload, odds_status = _api_get(
            'odds-by-tournaments',
            key,
            tournamentIds=str(NCAA_TOURNAMENT_ID),
            bookmakers=','.join(bookmaker_chunk),
            language='en',
            verbosity=3,
            oddsFormat='american',
        )
        bulk_statuses.append(odds_status)
        if odds_status == 'ok':
            stats['oddspapi_bulk_successes'] += 1
            _merge_odds_records(merged_odds, _as_records(odds_payload))
        else:
            stats['oddspapi_bulk_failures'] += 1
            detail = ''
            if isinstance(odds_payload, dict) and odds_payload.get('_http_error'):
                detail = str(odds_payload['_http_error'])[:160]
            books = ','.join(bookmaker_chunk)
            bulk_errors.append(f'{books}: {odds_status}{" - " + detail if detail else ""}')

    odds_records = list(merged_odds.values())
    stats['oddspapi_odds_fixtures'] = len(odds_records)
    if stats['oddspapi_bulk_successes'] and not stats['oddspapi_bulk_failures']:
        stats['oddspapi_status'] = 'ok'
        stats['oddspapi_odds_status'] = 'ok'
    elif stats['oddspapi_bulk_successes']:
        stats['oddspapi_status'] = 'partial'
        stats['oddspapi_odds_status'] = 'partial'
    elif not bulk_chunks:
        stats['oddspapi_status'] = 'quota_guard'
        stats['oddspapi_odds_status'] = 'quota_guard'
    else:
        stats['oddspapi_status'] = bulk_statuses[0] if bulk_statuses else 'no_bulk_calls'
        stats['oddspapi_odds_status'] = stats['oddspapi_status']
    if bulk_errors:
        stats['oddspapi_error_detail'] = '; '.join(bulk_errors)[:600]

    # /account is unmetered; read it again to record the true run cost.
    account_after, account_after_status = _api_get('account', key)
    if account_after_status == 'ok':
        after_sub = _active_subscription(account_after)
        stats['oddspapi_request_count'] = after_sub.get('request_count')
        stats['oddspapi_request_limit'] = after_sub.get('request_limit') or stats['oddspapi_request_limit']

    return fixtures, odds_records, catalog, stats


def _write_bookmaker_coverage(
    board: pd.DataFrame,
    matched: dict[Any, tuple[dict[str, Any], float]],
    odds_by_fixture: dict[str, dict[str, Any]],
    market_catalog: Any,
    bookmakers: list[str],
) -> dict[str, int]:
    """Persist per-game/per-book live FCS total coverage for source tuning."""
    rows: list[dict[str, Any]] = []
    coverage: Counter[str] = Counter()

    for idx, (fixture, _confidence) in matched.items():
        fixture_id = str(fixture.get('fixtureId') or '')
        odds_fixture = odds_by_fixture.get(fixture_id, {})
        books_node = odds_fixture.get('bookmakerOdds') or {}
        valid = _valid_total_lines_by_book(odds_fixture, market_catalog) if odds_fixture else {}
        game = board.loc[idx]

        for slug in bookmakers:
            lines = sorted(valid.get(slug, set()))
            has_valid = bool(lines)
            if has_valid:
                coverage[slug] += 1
            rows.append({
                'game_id': game.get('game_id'),
                'away_team': game.get('away_team'),
                'home_team': game.get('home_team'),
                'fixture_id': fixture_id,
                'bookmaker': slug,
                'book_present': bool(isinstance(books_node, dict) and slug in books_node),
                'valid_full_game_total': has_valid,
                'valid_total_lines': '|'.join(f'{line:g}' for line in lines),
            })

    COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'game_id', 'away_team', 'home_team', 'fixture_id', 'bookmaker',
        'book_present', 'valid_full_game_total', 'valid_total_lines',
    ]
    pd.DataFrame(rows, columns=columns).to_csv(COVERAGE_PATH, index=False)
    return {slug: int(coverage.get(slug, 0)) for slug in bookmakers}


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
        'fcs_oddspapi_matched_fixtures': 0,
        'fcs_oddspapi_matched_with_any_odds': 0,
        'oddspapi_status': 'not_needed' if not missing.any() else 'not_called',
    }
    if not missing.any():
        return out, stats

    live_fetch = fixtures is None or odds_records is None or market_catalog is None
    if live_fetch:
        fixtures, odds_records, market_catalog, fetched = fetch_oddspapi_ncaa(out.loc[missing].copy())
        stats.update(fetched)
    else:
        stats['oddspapi_status'] = 'fixture'

    matched: dict[Any, tuple[dict[str, Any], float]] = {}
    for idx, game in out.loc[missing].iterrows():
        fixture, confidence = match_oddspapi_fixture(game, fixtures)
        if fixture is None:
            continue
        matched[idx] = (fixture, confidence)
        stats['fcs_oddspapi_matched_fixtures'] += 1
        if bool(fixture.get('hasOdds')):
            stats['fcs_oddspapi_matched_with_any_odds'] += 1

    if not fixtures or not odds_records or not market_catalog:
        stats['fcs_missing_after'] = int(missing.sum())
        return out, stats

    odds_by_fixture = {
        str(row.get('fixtureId')): row
        for row in odds_records
        if isinstance(row, dict) and row.get('fixtureId')
    }

    if live_fetch:
        queried_books = [
            slug for slug in NCAA_BOOKMAKER_PRIORITY
            if any(
                isinstance(row.get('bookmakerOdds'), dict) and slug in row.get('bookmakerOdds', {})
                for row in odds_records
            )
        ]
        if not queried_books:
            queried_books = NCAA_BOOKMAKER_PRIORITY[: int(stats.get('oddspapi_bulk_bookmaker_count') or 0)]
        stats['oddspapi_book_total_coverage'] = _write_bookmaker_coverage(
            out, matched, odds_by_fixture, market_catalog, queried_books,
        )

    for idx, (fixture, confidence) in matched.items():
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
