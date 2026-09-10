from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .utils import ROOT

DEFAULT_SOURCES = (
    ROOT / 'outputs/weekly_board.csv',
    ROOT / 'outputs/prospective/2026/official_entries.csv',
)


def _utc(value: object | None = None) -> pd.Timestamp:
    if value is None or str(value).strip() == '':
        return pd.Timestamp.now(tz='UTC')
    ts = pd.to_datetime(value, utc=True, errors='coerce')
    if pd.isna(ts):
        raise ValueError(f'Could not parse UTC time: {value!r}')
    return ts


def candidate_kickoffs(
    now: object | None = None,
    window_minutes: int = 105,
    sources: tuple[Path, ...] = DEFAULT_SOURCES,
) -> pd.DataFrame:
    """Return locally known games that could enter the 90-minute close window.

    This deliberately uses only repository state. It is a cheap preflight gate
    so the high-frequency close-capture workflow does not spend CFBD/odds API
    requests when no tracked game is anywhere near kickoff. The authoritative
    capture routine still re-checks current CFBD kickoff times before writing an
    immutable benchmark.
    """
    now_ts = _utc(now)
    end = now_ts + pd.Timedelta(minutes=max(1, int(window_minutes)))
    parts: list[pd.DataFrame] = []

    for path in sources:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if frame.empty or 'start_date' not in frame.columns:
            continue
        keep = [c for c in ['game_id', 'start_date', 'away_team', 'home_team'] if c in frame.columns]
        if 'start_date' not in keep:
            continue
        part = frame[keep].copy()
        part['start_date'] = pd.to_datetime(part['start_date'], utc=True, errors='coerce')
        part = part[part['start_date'].notna()].copy()
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame(columns=['game_id', 'start_date', 'away_team', 'home_team'])

    candidates = pd.concat(parts, ignore_index=True)
    if 'game_id' in candidates.columns:
        candidates = candidates.sort_values('start_date').drop_duplicates('game_id', keep='last')
    candidates = candidates[
        candidates['start_date'].ge(now_ts)
        & candidates['start_date'].le(end)
    ].copy()
    return candidates.sort_values('start_date').reset_index(drop=True)


def should_capture(
    now: object | None = None,
    window_minutes: int = 105,
    sources: tuple[Path, ...] = DEFAULT_SOURCES,
) -> bool:
    return not candidate_kickoffs(now=now, window_minutes=window_minutes, sources=sources).empty


def main() -> None:
    parser = argparse.ArgumentParser(description='Zero-API preflight gate for near-kickoff market capture.')
    parser.add_argument('--window-minutes', type=int, default=105)
    parser.add_argument('--now-utc', default=os.getenv('CLOSE_CAPTURE_NOW_UTC', ''))
    args = parser.parse_args()

    now = args.now_utc or None
    candidates = candidate_kickoffs(now=now, window_minutes=args.window_minutes)
    decision = not candidates.empty
    output_path = os.getenv('GITHUB_OUTPUT', '').strip()
    if output_path:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"should_capture={'true' if decision else 'false'}\n")
            f.write(f"candidate_count={len(candidates)}\n")

    if decision:
        first = candidates.iloc[0]
        matchup = f"{first.get('away_team', '')} @ {first.get('home_team', '')}".strip(' @')
        print(
            f'Close-capture preflight: {len(candidates)} locally known game(s) within '
            f'{args.window_minutes} minutes; first={matchup or first.get("game_id", "unknown")} '
            f'at {first["start_date"]}. Authoritative API capture will run.'
        )
    else:
        print(
            f'Close-capture preflight: no locally known game is within {args.window_minutes} minutes. '
            'Skipping CFBD and odds requests for this scheduled tick.'
        )


if __name__ == '__main__':
    main()
