from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from ..utils import ROOT


DEFAULT_ENTRIES = ROOT / 'outputs/nfl/prospective/2026/official_entries.csv'
DEFAULT_BOARD = ROOT / 'outputs/nfl/live/weekly_board.csv'


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
    entries_path: Path = DEFAULT_ENTRIES,
    board_path: Path = DEFAULT_BOARD,
) -> pd.DataFrame:
    now_ts = _utc(now)
    end = now_ts + pd.Timedelta(minutes=max(1, int(window_minutes)))

    if not entries_path.exists() or entries_path.stat().st_size == 0:
        return pd.DataFrame(
            columns=['game_id', 'kickoff_utc', 'away_team', 'home_team']
        )
    try:
        entries = pd.read_csv(entries_path)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame(
            columns=['game_id', 'kickoff_utc', 'away_team', 'home_team']
        )
    if entries.empty or 'game_id' not in entries.columns:
        return pd.DataFrame(
            columns=['game_id', 'kickoff_utc', 'away_team', 'home_team']
        )

    keep = [
        c for c in [
            'game_id',
            'kickoff_utc',
            'away_team',
            'home_team',
        ]
        if c in entries.columns
    ]
    tracked = entries[keep].copy()
    tracked['_game_key'] = tracked['game_id'].astype(str)

    # Prefer the mutable live board's kickoff only for schedule updates. Entry
    # line/price/model/weather remain immutable in official_entries.csv.
    if board_path.exists() and board_path.stat().st_size > 0:
        try:
            board = pd.read_csv(board_path)
        except (
            OSError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            board = pd.DataFrame()
        if not board.empty and {
            'game_id',
            'kickoff_utc',
        }.issubset(board.columns):
            current = board[
                [
                    c for c in [
                        'game_id',
                        'kickoff_utc',
                        'away_team',
                        'home_team',
                    ]
                    if c in board.columns
                ]
            ].copy()
            current['_game_key'] = current['game_id'].astype(str)
            current = current.drop_duplicates(
                '_game_key',
                keep='first',
            )
            rename = {
                'kickoff_utc': '_current_kickoff_utc',
                'away_team': '_current_away_team',
                'home_team': '_current_home_team',
            }
            current = current.rename(columns=rename)
            tracked = tracked.merge(
                current.drop(columns=['game_id'], errors='ignore'),
                on='_game_key',
                how='left',
            )
            if '_current_kickoff_utc' in tracked.columns:
                tracked['kickoff_utc'] = tracked[
                    '_current_kickoff_utc'
                ].where(
                    tracked['_current_kickoff_utc'].notna(),
                    tracked.get('kickoff_utc'),
                )
            for side in ['away', 'home']:
                current_col = f'_current_{side}_team'
                base_col = f'{side}_team'
                if current_col in tracked.columns:
                    tracked[base_col] = tracked[current_col].where(
                        tracked[current_col].notna(),
                        tracked.get(base_col),
                    )

    tracked['kickoff_utc'] = pd.to_datetime(
        tracked.get('kickoff_utc'),
        utc=True,
        errors='coerce',
    )
    tracked = tracked[
        tracked['kickoff_utc'].notna()
        & tracked['kickoff_utc'].ge(now_ts)
        & tracked['kickoff_utc'].le(end)
    ].copy()
    drop_cols = [
        c for c in tracked.columns
        if c.startswith('_current_') or c == '_game_key'
    ]
    return (
        tracked.drop(columns=drop_cols, errors='ignore')
        .sort_values('kickoff_utc')
        .reset_index(drop=True)
    )


def should_capture(
    now: object | None = None,
    window_minutes: int = 105,
    entries_path: Path = DEFAULT_ENTRIES,
    board_path: Path = DEFAULT_BOARD,
) -> bool:
    return not candidate_kickoffs(
        now=now,
        window_minutes=window_minutes,
        entries_path=entries_path,
        board_path=board_path,
    ).empty


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Zero-API preflight for NFL near-kickoff CLV capture.'
    )
    parser.add_argument('--window-minutes', type=int, default=105)
    parser.add_argument(
        '--now-utc',
        default=os.getenv('NFL_CLOSE_CAPTURE_NOW_UTC', ''),
    )
    args = parser.parse_args()

    candidates = candidate_kickoffs(
        now=args.now_utc or None,
        window_minutes=args.window_minutes,
    )
    decision = not candidates.empty
    output_path = os.getenv('GITHUB_OUTPUT', '').strip()
    if output_path:
        with open(output_path, 'a', encoding='utf-8') as handle:
            handle.write(
                f'should_capture={"true" if decision else "false"}\n'
            )
            handle.write(f'candidate_count={len(candidates)}\n')

    if decision:
        first = candidates.iloc[0]
        matchup = (
            f'{first.get("away_team", "")} @ '
            f'{first.get("home_team", "")}'
        ).strip(' @')
        print(
            f'NFL close preflight: {len(candidates)} official entry game(s) '
            f'within {args.window_minutes} minutes; '
            f'first={matchup or first.get("game_id", "unknown")} '
            f'at {first["kickoff_utc"]}.'
        )
    else:
        print(
            'NFL close preflight: no official entry is near kickoff; '
            'skipping sportsbook API request.'
        )


if __name__ == '__main__':
    main()
