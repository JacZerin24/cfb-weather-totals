from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ROOT

CACHE_PATH = ROOT / 'outputs/market_total_cache.csv'

LINE_FIELDS = [
    'closing_total',
    'line_provider',
    'line_source',
    'line_provider_count',
    'line_total_min',
    'line_total_max',
    'line_total_range',
    'line_total_median',
    'selected_vs_market_median',
    'odds_match_confidence',
]

CACHE_COLUMNS = [
    'game_id', 'season', 'week', 'start_date', 'away_team', 'home_team',
    'division_track', *LINE_FIELDS, 'line_first_seen_utc', 'line_last_seen_utc',
]


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors='coerce')
    if pd.isna(ts):
        return pd.NaT
    return ts


def _now_utc(now: Any | None = None) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp.now(tz='UTC')
    ts = _as_utc_timestamp(now)
    if pd.isna(ts):
        raise ValueError('now must be parseable as a UTC timestamp')
    return ts


def _normalize_game_id(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _empty_cache() -> pd.DataFrame:
    return pd.DataFrame(columns=CACHE_COLUMNS)


def load_market_cache(path: str | Path = CACHE_PATH) -> pd.DataFrame:
    cache_path = Path(path)
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return _empty_cache()
    try:
        cache = pd.read_csv(cache_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return _empty_cache()

    for col in CACHE_COLUMNS:
        if col not in cache.columns:
            cache[col] = np.nan
    cache = cache[CACHE_COLUMNS].copy()
    cache['game_id'] = cache['game_id'].map(_normalize_game_id)
    cache = cache[cache['game_id'].notna()].copy()
    if cache.empty:
        return _empty_cache()
    cache['game_id'] = cache['game_id'].astype(int)
    cache['line_last_seen_utc'] = pd.to_datetime(cache['line_last_seen_utc'], utc=True, errors='coerce')
    cache = cache.sort_values('line_last_seen_utc').drop_duplicates('game_id', keep='last')
    return cache.reset_index(drop=True)


def _cache_row_from_live(row: pd.Series, now: pd.Timestamp, first_seen: Any | None) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for col in ['game_id', 'season', 'week', 'start_date', 'away_team', 'home_team', 'division_track', *LINE_FIELDS]:
        record[col] = row.get(col, np.nan)
    record['game_id'] = _normalize_game_id(record['game_id'])
    first = _as_utc_timestamp(first_seen)
    if pd.isna(first):
        first = now
    record['line_first_seen_utc'] = first.isoformat()
    record['line_last_seen_utc'] = now.isoformat()
    return record


def _prune_cache(cache: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    if cache.empty:
        return cache
    starts = pd.to_datetime(cache.get('start_date'), utc=True, errors='coerce')
    keep = starts.isna() | (starts >= now - pd.Timedelta(days=14))
    return cache.loc[keep].copy()


def apply_market_line_cache(
    board: pd.DataFrame,
    *,
    now: Any | None = None,
    path: str | Path = CACHE_PATH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Preserve the last successfully observed total for every live-site game.

    Rows with a line on the current run are stamped LIVE and update the cache.
    Rows missing a current line may be restored from the cache and are stamped
    CACHED. Cached lines are intentionally distinguishable so downstream
    decision logic can prevent them from becoming official qualifiers.
    """
    out = board.copy()
    timestamp = _now_utc(now)
    cache_path = Path(path)
    cache = load_market_cache(cache_path)

    if 'closing_total' not in out.columns:
        out['closing_total'] = np.nan
    out['closing_total'] = pd.to_numeric(out['closing_total'], errors='coerce')
    if 'line_source' not in out.columns:
        out['line_source'] = ''
    if 'line_provider' not in out.columns:
        out['line_provider'] = ''

    fresh_mask = out['closing_total'].notna()
    out['line_cache_status'] = np.where(fresh_mask, 'LIVE', 'NO LINE')
    out['line_is_cached'] = False
    out['line_last_seen_utc'] = ''
    out['line_age_hours'] = np.nan
    out.loc[fresh_mask, 'line_last_seen_utc'] = timestamp.isoformat()
    out.loc[fresh_mask, 'line_age_hours'] = 0.0

    cache_lookup: dict[int, pd.Series] = {}
    if not cache.empty:
        cache_lookup = {int(row['game_id']): row for _, row in cache.iterrows()}

    restored = 0
    for idx, row in out.loc[~fresh_mask].iterrows():
        game_id = _normalize_game_id(row.get('game_id'))
        cached = cache_lookup.get(game_id) if game_id is not None else None
        if cached is None or pd.isna(pd.to_numeric(pd.Series([cached.get('closing_total')]), errors='coerce').iloc[0]):
            continue
        for field in LINE_FIELDS:
            if field in cached.index:
                out.at[idx, field] = cached.get(field)
        last_seen = _as_utc_timestamp(cached.get('line_last_seen_utc'))
        age_hours = np.nan
        if pd.notna(last_seen):
            age_hours = max(0.0, (timestamp - last_seen).total_seconds() / 3600.0)
            out.at[idx, 'line_last_seen_utc'] = last_seen.isoformat()
        out.at[idx, 'line_age_hours'] = age_hours
        out.at[idx, 'line_cache_status'] = 'CACHED'
        out.at[idx, 'line_is_cached'] = True
        restored += 1

    existing_first_seen: dict[int, Any] = {}
    if not cache.empty:
        existing_first_seen = {
            int(row['game_id']): row.get('line_first_seen_utc')
            for _, row in cache.iterrows()
        }

    live_rows: list[dict[str, Any]] = []
    for _, row in out.loc[fresh_mask].iterrows():
        game_id = _normalize_game_id(row.get('game_id'))
        if game_id is None:
            continue
        live_rows.append(_cache_row_from_live(row, timestamp, existing_first_seen.get(game_id)))

    if live_rows:
        live_cache = pd.DataFrame(live_rows, columns=CACHE_COLUMNS)
        if cache.empty:
            cache = live_cache
        else:
            live_ids = set(live_cache['game_id'].dropna().astype(int))
            cache = cache[~cache['game_id'].astype(int).isin(live_ids)].copy()
            cache = pd.concat([cache, live_cache], ignore_index=True)

    cache = _prune_cache(cache, timestamp)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(cache_path, index=False)

    stats = {
        'fresh_lines': int(fresh_mask.sum()),
        'restored_lines': int(restored),
        'usable_lines': int(out['closing_total'].notna().sum()),
        'cached_entries': int(len(cache)),
    }
    return out, stats


def cached_market_reason(row: pd.Series) -> str | None:
    if str(row.get('line_cache_status') or '').upper() != 'CACHED':
        return None
    age = pd.to_numeric(pd.Series([row.get('line_age_hours')]), errors='coerce').iloc[0]
    provider = str(row.get('line_provider') or 'previous market source').strip()
    if pd.notna(age):
        return (
            f'Last-known market total from {provider}, last confirmed {float(age):.1f} hours ago. '
            'The model may display an orientation, but a fresh market line is required before this game can qualify.'
        )
    return (
        f'Last-known market total from {provider}. The model may display an orientation, '
        'but a fresh market line is required before this game can qualify.'
    )
