from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .oddspapi_fallback import (
    NCAA_TOURNAMENT_ID,
    _api_get,
    _as_records,
    _date_window,
    match_oddspapi_fixture,
)

OUTPUT_PATH = Path('outputs/oddspapi_schema_probe.json')
BOOKMAKER = 'draftkings'


def _primitive_fields(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)] = item
    return out


def _catalog_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ('markets', 'data', 'results'):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            rows = [row for row in value.values() if isinstance(row, dict)]
            if rows:
                return rows
    return [row for row in payload.values() if isinstance(row, dict)]


def _outcome_summary(outcomes: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'type': type(outcomes).__name__,
        'keys': [],
        'sample': None,
    }
    if isinstance(outcomes, dict):
        summary['keys'] = [str(k) for k in list(outcomes.keys())[:8]]
        values = [v for v in outcomes.values() if isinstance(v, dict)]
    elif isinstance(outcomes, list):
        values = [v for v in outcomes if isinstance(v, dict)]
        summary['keys'] = [str(i) for i in range(min(len(outcomes), 8))]
    else:
        values = []
    if not values:
        return summary

    outcome = values[0]
    sample: dict[str, Any] = {
        'primitive_fields': _primitive_fields(outcome),
        'keys': [str(k) for k in outcome.keys()],
    }
    players = outcome.get('players')
    sample['players_type'] = type(players).__name__
    if isinstance(players, dict):
        sample['players_keys'] = [str(k) for k in list(players.keys())[:8]]
        player_values = [v for v in players.values() if isinstance(v, dict)]
    elif isinstance(players, list):
        sample['players_keys'] = [str(i) for i in range(min(len(players), 8))]
        player_values = [v for v in players if isinstance(v, dict)]
    else:
        sample['players_keys'] = []
        player_values = []
    if player_values:
        sample['player_primitive_fields'] = _primitive_fields(player_values[0])
        sample['player_keys'] = [str(k) for k in player_values[0].keys()]
    summary['sample'] = sample
    return summary


def main() -> None:
    key = os.getenv('ODDSPAPI_API_KEY', '').strip()
    if not key:
        raise SystemExit('ODDSPAPI_API_KEY is missing; schema probe cannot run.')

    board_path = Path('outputs/weekly_board.csv')
    if not board_path.exists():
        raise SystemExit('outputs/weekly_board.csv is missing; schema probe needs the previous live board.')
    board = pd.read_csv(board_path)
    fcs = board.loc[board.get('division_track', '').astype(str).str.upper().eq('FCS')].copy()
    if fcs.empty:
        raise SystemExit('No FCS rows are available in the previous live board.')

    start, end = _date_window(fcs)
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
    catalog_payload, catalog_status = _api_get('markets', key, language='en')
    odds_payload, odds_status = _api_get(
        'odds-by-tournaments',
        key,
        tournamentIds=str(NCAA_TOURNAMENT_ID),
        bookmakers=BOOKMAKER,
        language='en',
        verbosity=3,
        oddsFormat='american',
    )

    fixtures = _as_records(fixtures_payload)
    odds_records = _as_records(odds_payload)
    odds_by_fixture = {
        str(row.get('fixtureId')): row
        for row in odds_records
        if isinstance(row, dict) and row.get('fixtureId')
    }

    selected_game: pd.Series | None = None
    selected_fixture: dict[str, Any] | None = None
    selected_confidence = 0.0
    selected_odds: dict[str, Any] | None = None
    for _, game in fcs.iterrows():
        fixture, confidence = match_oddspapi_fixture(game, fixtures)
        if fixture is None:
            continue
        odds = odds_by_fixture.get(str(fixture.get('fixtureId') or ''))
        if odds is None:
            continue
        selected_game = game
        selected_fixture = fixture
        selected_confidence = confidence
        selected_odds = odds
        break

    catalog_records = _catalog_records(catalog_payload)
    catalog_by_id = {
        str(row.get('marketId')): row
        for row in catalog_records
        if row.get('marketId') is not None
    }

    summary: dict[str, Any] = {
        'probe_version': 1,
        'bookmaker': BOOKMAKER,
        'statuses': {
            'fixtures': fixtures_status,
            'markets': catalog_status,
            'odds': odds_status,
        },
        'counts': {
            'fcs_board_games': int(len(fcs)),
            'fixtures': int(len(fixtures)),
            'odds_fixtures': int(len(odds_records)),
            'catalog_records_detected': int(len(catalog_records)),
        },
        'catalog_shape': {
            'type': type(catalog_payload).__name__,
            'top_keys': list(catalog_payload.keys())[:20] if isinstance(catalog_payload, dict) else [],
            'sample_records': [_primitive_fields(row) for row in catalog_records[:5]],
        },
    }

    if selected_game is None or selected_fixture is None or selected_odds is None:
        summary['selected'] = None
    else:
        books = selected_odds.get('bookmakerOdds') or {}
        book = books.get(BOOKMAKER) if isinstance(books, dict) else None
        markets = book.get('markets') if isinstance(book, dict) else None
        market_summaries: list[dict[str, Any]] = []
        if isinstance(markets, dict):
            market_items = list(markets.items())
        elif isinstance(markets, list):
            market_items = [(str(i), market) for i, market in enumerate(markets)]
        else:
            market_items = []

        for market_key, market in market_items[:40]:
            if not isinstance(market, dict):
                continue
            catalog_row = catalog_by_id.get(str(market_key), {})
            market_summaries.append({
                'market_key': str(market_key),
                'market_primitive_fields': _primitive_fields(market),
                'market_keys': [str(k) for k in market.keys()],
                'catalog_match': _primitive_fields(catalog_row),
                'outcomes': _outcome_summary(market.get('outcomes')),
            })

        summary['selected'] = {
            'game': {
                'away_team': str(selected_game.get('away_team') or ''),
                'home_team': str(selected_game.get('home_team') or ''),
                'start_date': str(selected_game.get('start_date') or ''),
            },
            'match_confidence': round(float(selected_confidence), 4),
            'fixture': {
                key_name: selected_fixture.get(key_name)
                for key_name in (
                    'fixtureId', 'tournamentId', 'startTime', 'participant1Name',
                    'participant2Name', 'hasOdds',
                )
            },
            'odds_fixture_keys': [str(k) for k in selected_odds.keys()],
            'bookmaker_keys': [str(k) for k in book.keys()] if isinstance(book, dict) else [],
            'bookmaker_primitive_fields': _primitive_fields(book),
            'markets_type': type(markets).__name__,
            'market_count': len(market_items),
            'markets': market_summaries,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    print(
        f"OddsPapi schema probe: fixtures={fixtures_status}, markets={catalog_status}, "
        f"odds={odds_status}; wrote {OUTPUT_PATH}."
    )


if __name__ == '__main__':
    main()
