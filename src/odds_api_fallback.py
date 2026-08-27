from __future__ import annotations

import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
import requests

ODDS_URL = 'https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds'

# Common naming differences between CFBD and sportsbook feeds. Keep this list
# intentionally conservative; fuzzy matching is only used after kickoff/date
# filtering and still requires both teams to match well.
TEAM_ALIASES = {
    'se louisiana': 'southeastern louisiana',
    'southeastern louisiana lions': 'southeastern louisiana',
    'ut martin': 'tennessee martin',
    'tennessee martin skyhawks': 'tennessee martin',
    'ualbany': 'albany',
    'albany great danes': 'albany',
    'mcneese state': 'mcneese',
    'mcneese cowboys': 'mcneese',
    'nicholls state': 'nicholls',
    'nicholls colonels': 'nicholls',
    'grambling state': 'grambling',
    'grambling tigers': 'grambling',
    'tarleton state': 'tarleton',
    'tarleton state texans': 'tarleton',
    'prairie view': 'prairie view am',
    'prairie view a m': 'prairie view am',
    'prairie view am panthers': 'prairie view am',
    'texas am commerce': 'east texas am',
    'east texas a m': 'east texas am',
    'east texas am lions': 'east texas am',
    'central connecticut state': 'central connecticut',
    'central connecticut blue devils': 'central connecticut',
}


def _normalize_team(value: Any) -> str:
    text = unicodedata.normalize('NFKD', str(value or '')).encode('ascii', 'ignore').decode('ascii').lower()
    text = text.replace('&', ' and ')
    text = re.sub(r"['’`.-]", ' ', text)
    text = re.sub(r'\ba\s+and\s+m\b', 'am', text)
    text = re.sub(r'\ba\s+m\b', 'am', text)
    text = re.sub(r'\bst\b', 'state', text)
    text = re.sub(r'\buniversity\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return TEAM_ALIASES.get(text, text)


def _team_score(left: Any, right: Any) -> float:
    a = _normalize_team(left)
    b = _normalize_team(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Sportsbook feeds commonly append nicknames ("Stony Brook Seawolves")
    # to the school name supplied by CFBD.
    shorter, longer = sorted([a, b], key=len)
    if len(shorter) >= 5 and (longer.startswith(shorter + ' ') or longer.endswith(' ' + shorter)):
        return 0.97
    seq = SequenceMatcher(None, a, b).ratio()
    aset, bset = set(a.split()), set(b.split())
    dice = (2 * len(aset & bset) / (len(aset) + len(bset))) if aset and bset else 0.0
    return max(seq, dice)


def _kickoff_distance_hours(game_time: Any, event_time: Any) -> float:
    game_ts = pd.to_datetime(game_time, utc=True, errors='coerce')
    event_ts = pd.to_datetime(event_time, utc=True, errors='coerce')
    if pd.isna(game_ts) or pd.isna(event_ts):
        return 9999.0
    return abs((game_ts - event_ts).total_seconds()) / 3600.0


def match_odds_event(game: pd.Series, events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best: dict[str, Any] | None = None
    best_score = 0.0
    for event in events:
        hours = _kickoff_distance_hours(game.get('start_date'), event.get('commence_time'))
        if hours > 36:
            continue
        home_score = _team_score(game.get('home_team'), event.get('home_team'))
        away_score = _team_score(game.get('away_team'), event.get('away_team'))
        if min(home_score, away_score) < 0.72:
            continue
        score = (home_score + away_score) / 2.0
        # Prefer closer kickoff times when team-name scores are otherwise similar.
        score -= min(hours, 24.0) / 2400.0
        if score > best_score:
            best = event
            best_score = score
    if best_score < 0.82:
        return None, best_score
    return best, best_score


def _provider_rank(name: str, preferred: list[str]) -> int:
    norm = re.sub(r'[^a-z0-9]+', '', str(name).lower())
    for idx, provider in enumerate(preferred):
        if norm == re.sub(r'[^a-z0-9]+', '', str(provider).lower()):
            return idx
    return len(preferred) + 50


def select_total_market(event: dict[str, Any], preferred: list[str]) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for book in event.get('bookmakers') or []:
        title = str(book.get('title') or book.get('key') or 'unknown')
        for market in book.get('markets') or []:
            if market.get('key') != 'totals':
                continue
            points = []
            for outcome in market.get('outcomes') or []:
                point = pd.to_numeric(pd.Series([outcome.get('point')]), errors='coerce').iloc[0]
                if pd.notna(point):
                    points.append(float(point))
            if not points:
                continue
            # Over and under normally carry the same point. Median protects
            # against malformed/alternate outcomes if a provider returns more.
            rows.append({'provider': title, 'total': float(np.median(points))})
            break

    if not rows:
        return None
    market = pd.DataFrame(rows).drop_duplicates('provider')
    median_total = float(market['total'].median())
    market['distance_to_median'] = (market['total'] - median_total).abs()
    market['provider_rank'] = market['provider'].map(lambda value: _provider_rank(value, preferred))
    chosen = market.sort_values(['distance_to_median', 'provider_rank', 'provider']).iloc[0]
    return {
        'closing_total': float(chosen['total']),
        'line_provider': f"{chosen['provider']} (Odds API)",
        'line_provider_count': int(market['provider'].nunique()),
        'line_total_min': float(market['total'].min()),
        'line_total_max': float(market['total'].max()),
        'line_total_median': median_total,
        'line_total_range': float(market['total'].max() - market['total'].min()),
        'selected_vs_market_median': float(chosen['total'] - median_total),
        'line_source': 'The Odds API',
    }


def fetch_current_ncaaf_odds() -> tuple[list[dict[str, Any]], str]:
    key = os.getenv('ODDS_API_KEY', '').strip()
    if not key:
        return [], 'missing_key'
    try:
        response = requests.get(
            ODDS_URL,
            params={
                'regions': 'us',
                'markets': 'totals',
                'oddsFormat': 'american',
                'dateFormat': 'iso',
                'apiKey': key,
            },
            timeout=30,
        )
    except requests.RequestException:
        return [], 'request_error'
    if response.status_code != 200:
        return [], f'http_{response.status_code}'
    try:
        payload = response.json()
    except ValueError:
        return [], 'invalid_json'
    return payload if isinstance(payload, list) else [], 'ok'


def apply_fcs_odds_fallback(
    board: pd.DataFrame,
    preferred_providers: list[str] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = board.copy()
    preferred = list(preferred_providers or [])
    if 'line_source' not in out.columns:
        out['line_source'] = np.where(out.get('closing_total', pd.Series(np.nan, index=out.index)).notna(), 'CFBD', '')
    else:
        out.loc[out.get('closing_total', pd.Series(np.nan, index=out.index)).notna() & out['line_source'].astype(str).eq(''), 'line_source'] = 'CFBD'

    division = out.get('division_track', pd.Series('', index=out.index)).astype(str).str.upper()
    missing = division.eq('FCS') & out.get('closing_total', pd.Series(np.nan, index=out.index)).isna()
    stats = {
        'fcs_missing_before': int(missing.sum()),
        'fcs_fallback_filled': 0,
        'odds_api_status': 'not_needed' if not missing.any() else 'not_called',
    }
    if not missing.any():
        return out, stats

    if events is None:
        events, api_status = fetch_current_ncaaf_odds()
        stats['odds_api_status'] = api_status
    else:
        api_status = 'fixture'
        stats['odds_api_status'] = api_status
    if not events:
        return out, stats

    for idx, game in out.loc[missing].iterrows():
        event, confidence = match_odds_event(game, events)
        if event is None:
            continue
        market = select_total_market(event, preferred)
        if market is None:
            continue
        for key, value in market.items():
            out.at[idx, key] = value
        out.at[idx, 'odds_match_confidence'] = round(float(confidence), 4)
        out.at[idx, 'odds_event_id'] = str(event.get('id') or '')
        stats['fcs_fallback_filled'] += 1

    stats['fcs_missing_after'] = int(
        (division.eq('FCS') & out.get('closing_total', pd.Series(np.nan, index=out.index)).isna()).sum()
    )
    return out, stats
