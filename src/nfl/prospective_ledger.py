from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import ROOT, load_yaml, read_df, write_df
from .live_week import (
    SCHEDULE_URL,
    _attach_market,
    _download_csv,
    _kickoff_utc,
    _odds_events,
)


PROTOCOL_PATH = ROOT / 'config/nfl_frozen_protocol.yml'
PROSPECTIVE_ROOT = ROOT / 'outputs/nfl/prospective/2026'
DECISION_DIR = PROSPECTIVE_ROOT / 'decision_snapshots'
MONITOR_DIR = PROSPECTIVE_ROOT / 'monitor_snapshots'
CLOSE_DIR = PROSPECTIVE_ROOT / 'close_captures'
LIVE_BOARD_PATH = ROOT / 'outputs/nfl/live/weekly_board.csv'
HASH_RE = re.compile(r'_([0-9a-f]{12})\.csv$')
CLOSE_WINDOW_MINUTES = 105


def load_protocol() -> dict[str, Any]:
    protocol = load_yaml('config/nfl_frozen_protocol.yml')
    if protocol.get('protocol', {}).get('id') != 'nfl_totals_paper_v1':
        raise RuntimeError('Unexpected NFL prospective protocol id.')
    if int(protocol.get('protocol', {}).get('version', 0)) != 1:
        raise RuntimeError('NFL prospective ledger requires protocol v1.')
    if protocol.get('protocol', {}).get('status') != (
        'frozen_for_prospective_paper_tracking'
    ):
        raise RuntimeError('NFL prospective protocol is not frozen.')
    if int(protocol.get('decision', {}).get('official_lead_hours', 0)) != 24:
        raise RuntimeError('NFL prospective ledger requires the frozen 24h horizon.')
    if str(protocol.get('decision', {}).get('side', '')).upper() != 'OVER':
        raise RuntimeError('NFL prospective ledger requires the frozen OVER side.')
    return protocol


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz='UTC')


def _iso_utc(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors='coerce')
    if pd.isna(ts):
        raise ValueError(f'Could not parse UTC timestamp: {value!r}')
    return ts.isoformat().replace('+00:00', 'Z')


def _run_metadata(now: pd.Timestamp, kind: str) -> dict[str, Any]:
    event_name = os.getenv(
        'NFL_PROSPECTIVE_EVENT_NAME',
        os.getenv('GITHUB_EVENT_NAME', ''),
    ).strip()
    event_schedule = os.getenv('NFL_PROSPECTIVE_EVENT_SCHEDULE', '').strip()
    run_attempt = int(
        os.getenv(
            'NFL_PROSPECTIVE_RUN_ATTEMPT',
            os.getenv('GITHUB_RUN_ATTEMPT', '1'),
        )
        or 1
    )
    official_eligible = event_name == 'schedule' and run_attempt == 1
    return {
        'record_kind': kind,
        'snapshot_timestamp_utc': _iso_utc(now),
        'protocol_id': 'nfl_totals_paper_v1',
        'protocol_version': 1,
        'protocol_sha256': protocol_sha256(),
        'github_event_name': event_name,
        'github_event_schedule': event_schedule,
        'github_run_id': os.getenv(
            'NFL_PROSPECTIVE_RUN_ID',
            os.getenv('GITHUB_RUN_ID', 'local'),
        ),
        'github_run_attempt': run_attempt,
        'source_git_sha': os.getenv(
            'NFL_PROSPECTIVE_SHA',
            os.getenv('GITHUB_SHA', 'local'),
        ),
        'official_eligible': bool(official_eligible),
    }


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator='\n').encode('utf-8')


def _immutable_write(
    frame: pd.DataFrame,
    directory: Path,
    prefix: str,
    now: pd.Timestamp,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = _csv_bytes(frame)
    digest = hashlib.sha256(payload).hexdigest()
    run_id = (
        str(frame['github_run_id'].iloc[0])
        if 'github_run_id' in frame.columns and len(frame)
        else os.getenv('GITHUB_RUN_ID', 'local')
    )
    attempt = (
        int(frame['github_run_attempt'].iloc[0])
        if 'github_run_attempt' in frame.columns and len(frame)
        else 1
    )
    stamp = pd.Timestamp(now).tz_convert('UTC').strftime('%Y%m%dT%H%M%SZ')
    safe_run = re.sub(r'[^A-Za-z0-9_.-]+', '-', run_id)
    path = (
        directory
        / f'{prefix}_{stamp}_run{safe_run}_a{attempt}_{digest[:12]}.csv'
    )
    if path.exists():
        raise RuntimeError(
            f'Refusing to overwrite immutable NFL prospective file: {path}'
        )
    with open(path, 'xb') as handle:
        handle.write(payload)
    return path


def _read_immutable(directory: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if not directory.exists():
        return pd.DataFrame()
    for path in sorted(directory.glob('*.csv')):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame['_immutable_file'] = str(
            path.relative_to(ROOT)
        ).replace('\\', '/')
        frame['_immutable_sha256'] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def verify_immutable_files() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kind, directory in [
        ('decision_snapshot', DECISION_DIR),
        ('monitor_snapshot', MONITOR_DIR),
        ('close_capture', CLOSE_DIR),
    ]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob('*.csv')):
            match = HASH_RE.search(path.name)
            if not match:
                raise RuntimeError(
                    'Immutable NFL prospective file is missing its '
                    f'content-hash suffix: {path}'
                )
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual[:12] != match.group(1):
                raise RuntimeError(
                    f'NFL prospective hash mismatch for {path.name}: '
                    f'filename={match.group(1)} content={actual[:12]}'
                )
            rows.append({
                'kind': kind,
                'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                'sha256': actual,
                'size_bytes': path.stat().st_size,
            })
    manifest = pd.DataFrame(rows)
    write_df(manifest, PROSPECTIVE_ROOT / 'manifest.csv')
    return manifest


def select_official_decisions(
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()

    work = snapshots.copy()
    work['official_eligible'] = work.get(
        'official_eligible',
        pd.Series(False, index=work.index),
    ).map(_bool)
    work['decision_due'] = work.get(
        'decision_due',
        pd.Series(False, index=work.index),
    ).map(_bool)
    work['github_run_attempt'] = pd.to_numeric(
        work.get('github_run_attempt'),
        errors='coerce',
    ).fillna(1).astype(int)
    work['season'] = pd.to_numeric(
        work.get('season'),
        errors='coerce',
    )
    work['snapshot_timestamp_utc'] = pd.to_datetime(
        work.get('snapshot_timestamp_utc'),
        utc=True,
        errors='coerce',
    )
    work['kickoff_utc'] = pd.to_datetime(
        work.get('kickoff_utc'),
        utc=True,
        errors='coerce',
    )
    work['decision_lead_hours'] = (
        work['kickoff_utc'] - work['snapshot_timestamp_utc']
    ).dt.total_seconds() / 3600.0

    valid = (
        work['official_eligible']
        & work['decision_due']
        & work['github_run_attempt'].eq(1)
        & work['season'].eq(2026)
        & work['snapshot_timestamp_utc'].notna()
        & work['kickoff_utc'].notna()
        & work['decision_state'].astype(str).eq('24H_DECISION_WINDOW')
        & work['record_kind'].astype(str).eq('decision_snapshot')
    )
    work = work[valid].copy()
    if work.empty:
        return pd.DataFrame()

    # The first scheduled snapshot that enters the frozen 24h decision window
    # is the operational decision. Later market/model movement may be studied
    # but cannot create, erase, or upgrade the official decision.
    work = work.sort_values(
        ['game_id', 'snapshot_timestamp_utc', 'github_run_id']
    )
    official = work.drop_duplicates('game_id', keep='first').copy()
    official['official_decision'] = True
    return official.sort_values(
        ['kickoff_utc', 'game_id']
    ).reset_index(drop=True)


def select_official_entries(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()

    work = decisions.copy()
    signal = work.get(
        'model_signal',
        pd.Series('', index=work.index),
    ).astype(str)
    market_total = pd.to_numeric(
        work.get('closing_total'),
        errors='coerce',
    )
    market_price = pd.to_numeric(
        work.get('consensus_over_price'),
        errors='coerce',
    )
    forecast_ok = work.get(
        'forecast_complete_24h',
        pd.Series(False, index=work.index),
    ).map(_bool)

    entries = work[
        signal.isin({'QUALIFIES', 'STRONG'})
        & market_total.notna()
        & market_price.notna()
        & forecast_ok
    ].copy()
    if entries.empty:
        return entries

    entries['official_entry'] = True
    entries['official_side'] = 'OVER'
    entries['entry_tier'] = entries['model_signal'].astype(str)
    entries['entry_total'] = pd.to_numeric(
        entries['closing_total'],
        errors='coerce',
    )
    entries['entry_over_price'] = pd.to_numeric(
        entries['consensus_over_price'],
        errors='coerce',
    )
    entries['entry_over_sportsbook'] = entries.get(
        'consensus_over_sportsbook',
        '',
    )
    entries['entry_best_over_price_same_line'] = pd.to_numeric(
        entries.get('best_over_price_same_line'),
        errors='coerce',
    )
    entries['entry_best_over_sportsbook_same_line'] = entries.get(
        'best_over_sportsbook_same_line',
        '',
    )
    return entries.reset_index(drop=True)


def select_benchmark_closes(
    close_captures: pd.DataFrame,
) -> pd.DataFrame:
    if close_captures.empty:
        return pd.DataFrame(columns=['game_id', 'benchmark_close_total'])

    work = close_captures.copy()
    work['snapshot_timestamp_utc'] = pd.to_datetime(
        work.get('snapshot_timestamp_utc'),
        utc=True,
        errors='coerce',
    )
    work['kickoff_utc'] = pd.to_datetime(
        work.get('kickoff_utc'),
        utc=True,
        errors='coerce',
    )
    work['benchmark_close_total'] = pd.to_numeric(
        work.get('benchmark_close_total'),
        errors='coerce',
    )
    work['github_run_attempt'] = pd.to_numeric(
        work.get('github_run_attempt'),
        errors='coerce',
    ).fillna(1).astype(int)
    work['close_capture_lead_minutes'] = (
        work['kickoff_utc'] - work['snapshot_timestamp_utc']
    ).dt.total_seconds() / 60.0

    valid = (
        work['benchmark_close_total'].notna()
        & work['snapshot_timestamp_utc'].notna()
        & work['kickoff_utc'].notna()
        & work['close_capture_lead_minutes'].between(
            0,
            CLOSE_WINDOW_MINUTES,
            inclusive='both',
        )
        & work['github_event_name'].astype(str).eq('schedule')
        & work['github_run_attempt'].eq(1)
        & work['record_kind'].astype(str).eq('close_capture')
    )
    work = work[valid].copy()
    if work.empty:
        return pd.DataFrame(columns=['game_id', 'benchmark_close_total'])

    # The latest valid pre-kickoff capture is the CLV benchmark.
    work = work.sort_values(['game_id', 'snapshot_timestamp_utc'])
    selected = work.groupby(
        'game_id',
        as_index=False,
        sort=False,
    ).tail(1).copy()

    rename = {
        'snapshot_timestamp_utc': 'benchmark_close_captured_at_utc',
        'benchmark_close_over_price': 'benchmark_close_over_price',
        'benchmark_close_over_sportsbook': 'benchmark_close_over_sportsbook',
        '_immutable_file': 'benchmark_close_immutable_file',
        '_immutable_sha256': 'benchmark_close_immutable_sha256',
    }
    selected = selected.rename(columns=rename)
    keep = [
        c for c in [
            'game_id',
            'benchmark_close_total',
            'benchmark_close_over_price',
            'benchmark_close_over_sportsbook',
            'benchmark_best_over_price_same_line',
            'benchmark_best_over_sportsbook_same_line',
            'benchmark_close_captured_at_utc',
            'close_capture_lead_minutes',
            'benchmark_close_immutable_file',
            'benchmark_close_immutable_sha256',
        ]
        if c in selected.columns
    ]
    return selected[keep].reset_index(drop=True)


def rebuild_derived() -> dict[str, pd.DataFrame]:
    verify_immutable_files()
    snapshots = _read_immutable(DECISION_DIR)
    closes = _read_immutable(CLOSE_DIR)

    decisions = select_official_decisions(snapshots)
    entries = select_official_entries(decisions)
    benchmarks = select_benchmark_closes(closes)

    write_df(
        decisions,
        PROSPECTIVE_ROOT / 'official_decisions.csv',
    )
    write_df(
        entries,
        PROSPECTIVE_ROOT / 'official_entries.csv',
    )
    write_df(
        benchmarks,
        PROSPECTIVE_ROOT / 'benchmark_closes.csv',
    )

    if entries.empty:
        with_clv = entries.copy()
    else:
        with_clv = entries.merge(
            benchmarks,
            on='game_id',
            how='left',
        )
        with_clv['clv_points'] = (
            pd.to_numeric(
                with_clv.get('benchmark_close_total'),
                errors='coerce',
            )
            - pd.to_numeric(
                with_clv.get('entry_total'),
                errors='coerce',
            )
        )
        with_clv['beat_close'] = np.where(
            with_clv['clv_points'].notna(),
            with_clv['clv_points'].gt(0),
            np.nan,
        )
    write_df(
        with_clv,
        PROSPECTIVE_ROOT / 'entries_with_clv.csv',
    )

    lines = [
        '# NFL 2026 Prospective Ledger',
        '',
        'Frozen protocol: nfl_totals_paper_v1',
        '',
        f'- Official decisions recorded: {len(decisions)}',
        f'- Official paper entries: {len(entries)}',
        f'- Near-kickoff benchmarks: {len(benchmarks)}',
        '',
        'Official decisions are selected from content-hashed immutable '
        'decision snapshots. The first eligible scheduled snapshot inside '
        'the frozen 24-hour window is permanent for that game.',
        '',
        'A later snapshot cannot retroactively create, erase, or upgrade '
        'an official entry. Near-kickoff captures are separate immutable '
        'records used only for CLV.',
    ]
    (PROSPECTIVE_ROOT / 'prospective_summary.md').parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (PROSPECTIVE_ROOT / 'prospective_summary.md').write_text(
        '\n'.join(lines),
        encoding='utf-8',
    )
    return {
        'decisions': decisions,
        'entries': entries,
        'benchmarks': benchmarks,
        'entries_with_clv': with_clv,
    }


def snapshot_board(now: pd.Timestamp | None = None) -> Path | None:
    load_protocol()
    now = now or _utc_now()
    metadata = _run_metadata(now, 'decision_snapshot')

    # Manual runs, PR runs, and reruns may refresh the live board, but they
    # cannot create prospective decisions or backfill missed official entries.
    if not metadata['official_eligible']:
        print(
            'NFL prospective snapshot skipped: only first-attempt scheduled '
            'live runs are official-eligible.'
        )
        rebuild_derived()
        return None

    if not LIVE_BOARD_PATH.exists():
        raise RuntimeError(
            'NFL live weekly_board.csv is missing; decision snapshot aborted.'
        )
    board = pd.read_csv(LIVE_BOARD_PATH)
    if board.empty:
        print('NFL live board is empty; no prospective snapshot written.')
        rebuild_derived()
        return None

    due = board[
        board.get(
            'decision_due',
            pd.Series(False, index=board.index),
        ).map(_bool)
        & board.get(
            'decision_state',
            pd.Series('', index=board.index),
        ).astype(str).eq('24H_DECISION_WINDOW')
    ].copy()
    due['season'] = pd.to_numeric(
        due.get('season'),
        errors='coerce',
    )
    due = due[due['season'].eq(2026)].copy()
    if due.empty:
        print('No 2026 game is in the frozen 24h decision window.')
        rebuild_derived()
        return None

    for key, value in reversed(list(metadata.items())):
        due.insert(0, key, value)

    path = _immutable_write(
        due,
        DECISION_DIR,
        'decision',
        now,
    )
    derived = rebuild_derived()
    print(
        'Wrote immutable NFL decision snapshot: '
        f'{path.relative_to(ROOT)}; '
        f'official decisions={len(derived["decisions"])} '
        f'entries={len(derived["entries"])}.'
    )
    return path


def monitor_board(now: pd.Timestamp | None = None) -> Path | None:
    load_protocol()
    now = now or _utc_now()
    metadata = _run_metadata(now, 'monitor_snapshot')

    if not metadata['official_eligible']:
        print(
            'NFL monitor snapshot skipped: only first-attempt scheduled '
            'live runs are accepted for post-freeze movement tracking.'
        )
        rebuild_derived()
        return None

    if not LIVE_BOARD_PATH.exists():
        print('NFL live board is missing; no monitor snapshot written.')
        rebuild_derived()
        return None

    board = pd.read_csv(LIVE_BOARD_PATH)
    if board.empty:
        print('NFL live board is empty; no monitor snapshot written.')
        rebuild_derived()
        return None

    derived = rebuild_derived()
    decisions = derived['decisions']
    if decisions.empty or 'game_id' not in decisions.columns:
        print('No official NFL decisions exist; no monitor snapshot needed.')
        return None

    official = decisions[
        [
            c for c in [
                'game_id',
                'snapshot_timestamp_utc',
                'model_signal',
                'closing_total',
                'pred_market_residual',
                'over_probability',
            ]
            if c in decisions.columns
        ]
    ].copy()
    official = official.rename(columns={
        'snapshot_timestamp_utc': 'official_decision_timestamp_utc',
        'model_signal': 'official_model_signal',
        'closing_total': 'official_total',
        'pred_market_residual': 'official_pred_market_residual',
        'over_probability': 'official_over_probability',
    })

    board['_game_key'] = board['game_id'].astype(str)
    official['_game_key'] = official['game_id'].astype(str)
    tracked = board.merge(
        official.drop(columns=['game_id'], errors='ignore'),
        on='_game_key',
        how='inner',
    )
    if tracked.empty:
        print('No current live-board games have an official decision.')
        return None

    tracked['kickoff_utc'] = pd.to_datetime(
        tracked.get('kickoff_utc'),
        utc=True,
        errors='coerce',
    )
    tracked['official_decision_timestamp_utc'] = pd.to_datetime(
        tracked.get('official_decision_timestamp_utc'),
        utc=True,
        errors='coerce',
    )
    tracked = tracked[
        tracked['kickoff_utc'].notna()
        & tracked['kickoff_utc'].gt(now)
        & tracked['official_decision_timestamp_utc'].notna()
        & tracked['official_decision_timestamp_utc'].lt(
            now - pd.Timedelta(minutes=30)
        )
    ].copy()
    if tracked.empty:
        print(
            'No post-freeze, pre-kickoff NFL games are eligible for '
            'movement monitoring.'
        )
        return None

    tracked = tracked.drop(columns=['_game_key'], errors='ignore')
    for key, value in reversed(list(metadata.items())):
        tracked.insert(0, key, value)

    path = _immutable_write(
        tracked,
        MONITOR_DIR,
        'monitor',
        now,
    )
    rebuild_derived()
    print(
        'Wrote immutable NFL post-freeze monitor snapshot: '
        f'{path.relative_to(ROOT)}; games={len(tracked)}.'
    )
    return path


def _current_tracked_games(
    entries: pd.DataFrame,
    now: pd.Timestamp,
) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()

    schedule = _download_csv(SCHEDULE_URL)
    schedule['season'] = pd.to_numeric(
        schedule['season'],
        errors='coerce',
    )
    schedule = schedule[
        schedule['season'].eq(2026)
        & schedule['game_type'].astype(str).str.upper().eq('REG')
    ].copy()
    schedule['kickoff_utc'] = [
        _kickoff_utc(day, time)
        for day, time in zip(schedule['gameday'], schedule['gametime'])
    ]

    tracked_ids = {
        str(v)
        for v in entries['game_id'].dropna().tolist()
    }
    schedule['_game_key'] = schedule['game_id'].astype(str)
    tracked = schedule[
        schedule['_game_key'].isin(tracked_ids)
    ].copy()
    if tracked.empty:
        return tracked

    end = now + pd.Timedelta(minutes=CLOSE_WINDOW_MINUTES)
    return tracked[
        tracked['kickoff_utc'].notna()
        & tracked['kickoff_utc'].ge(now)
        & tracked['kickoff_utc'].le(end)
    ].copy()


def capture_close(now: pd.Timestamp | None = None) -> Path | None:
    load_protocol()
    now = now or _utc_now()
    metadata = _run_metadata(now, 'close_capture')

    if not metadata['official_eligible']:
        print(
            'NFL close capture skipped: only first-attempt scheduled runs '
            'are accepted as CLV benchmarks.'
        )
        rebuild_derived()
        return None

    derived = rebuild_derived()
    entries = derived['entries']
    if entries.empty:
        print('No official NFL entries exist; no close capture needed.')
        return None

    imminent = _current_tracked_games(entries, now)
    if imminent.empty:
        print(
            f'No official NFL entry kicks off in the next '
            f'{CLOSE_WINDOW_MINUTES} minutes.'
        )
        return None

    events, quota = _odds_events()
    market = _attach_market(imminent, events, quota)
    market = market.merge(
        entries[
            [
                c for c in [
                    'game_id',
                    'entry_total',
                    'entry_over_price',
                    'entry_over_sportsbook',
                    'entry_tier',
                ]
                if c in entries.columns
            ]
        ],
        on='game_id',
        how='left',
    )
    market['benchmark_close_total'] = pd.to_numeric(
        market.get('closing_total'),
        errors='coerce',
    )
    market['benchmark_close_over_price'] = pd.to_numeric(
        market.get('consensus_over_price'),
        errors='coerce',
    )
    market['benchmark_close_over_sportsbook'] = market.get(
        'consensus_over_sportsbook',
        '',
    )
    market['benchmark_best_over_price_same_line'] = pd.to_numeric(
        market.get('best_over_price_same_line'),
        errors='coerce',
    )
    market['benchmark_best_over_sportsbook_same_line'] = market.get(
        'best_over_sportsbook_same_line',
        '',
    )
    market['capture_lead_minutes'] = (
        pd.to_datetime(
            market['kickoff_utc'],
            utc=True,
            errors='coerce',
        )
        - now
    ).dt.total_seconds() / 60.0

    for key, value in reversed(list(metadata.items())):
        market.insert(0, key, value)

    keep = [
        c for c in [
            *metadata.keys(),
            'season',
            'week',
            'game_id',
            'kickoff_utc',
            'gameday',
            'gametime',
            'away_team',
            'home_team',
            'entry_tier',
            'entry_total',
            'entry_over_price',
            'entry_over_sportsbook',
            'benchmark_close_total',
            'benchmark_close_over_price',
            'benchmark_close_over_sportsbook',
            'benchmark_best_over_price_same_line',
            'benchmark_best_over_sportsbook_same_line',
            'sportsbooks_at_selected_line',
            'sportsbooks_in_snapshot',
            'market_total_min',
            'market_total_max',
            'market_total_range',
            'odds_event_id',
            'odds_market_last_update',
            'capture_lead_minutes',
            'odds_request_remaining',
            'odds_request_used',
            'odds_request_last',
        ]
        if c in market.columns
    ]
    market = market[keep].copy()

    available = pd.to_numeric(
        market.get('benchmark_close_total'),
        errors='coerce',
    ).notna()
    if not available.any():
        raise RuntimeError(
            'Near-kickoff NFL capture matched tracked games but no '
            'benchmark total was available.'
        )

    path = _immutable_write(
        market,
        CLOSE_DIR,
        'close',
        now,
    )
    derived = rebuild_derived()
    print(
        'Wrote immutable NFL near-kickoff market capture: '
        f'{path.relative_to(ROOT)}; '
        f'benchmarks={len(derived["benchmarks"])}.'
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='NFL 2026 immutable prospective paper ledger.'
    )
    parser.add_argument(
        'command',
        choices=[
            'validate-protocol',
            'snapshot-board',
            'monitor-board',
            'capture-close',
            'rebuild',
            'verify',
        ],
    )
    args = parser.parse_args()

    if args.command == 'validate-protocol':
        protocol = load_protocol()
        print(
            'NFL prospective protocol validated: '
            f'{protocol["protocol"]["id"]} '
            f'v{protocol["protocol"]["version"]}.'
        )
    elif args.command == 'snapshot-board':
        snapshot_board()
    elif args.command == 'monitor-board':
        monitor_board()
    elif args.command == 'capture-close':
        capture_close()
    elif args.command == 'rebuild':
        result = rebuild_derived()
        print(
            'Rebuilt NFL prospective derived files: '
            f'{len(result["decisions"])} decisions, '
            f'{len(result["entries"])} entries, '
            f'{len(result["benchmarks"])} close benchmarks.'
        )
    else:
        manifest = verify_immutable_files()
        print(
            f'NFL prospective immutability verified for '
            f'{len(manifest)} file(s).'
        )


if __name__ == '__main__':
    main()
