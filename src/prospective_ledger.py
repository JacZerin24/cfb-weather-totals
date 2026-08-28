from __future__ import annotations

import argparse
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .build_dataset import pick_total
from .cfbd_client import CFBDClient
from .fcs_model import FCS_QUALIFY_EDGE, FCS_QUALIFY_TOTAL, division_track
from .odds_api_fallback import apply_fcs_odds_fallback
from .oddspapi_fallback import apply_fcs_oddspapi_fallback
from .predict_week import classify_row, line_market_context, normalize_games, normalize_lines
from .utils import ROOT, get_settings, load_yaml, read_df, write_df

PROTOCOL_PATH = ROOT / 'config/prospective_protocol_2026.yml'
PROSPECTIVE_ROOT = ROOT / 'outputs/prospective/2026'
SNAPSHOT_DIR = PROSPECTIVE_ROOT / 'snapshots'
CLOSE_DIR = PROSPECTIVE_ROOT / 'close_captures'
HASH_RE = re.compile(r'_([0-9a-f]{12})\.csv$')


def load_protocol() -> dict[str, Any]:
    protocol = load_yaml('config/prospective_protocol_2026.yml')
    if int(protocol.get('season', 0)) != 2026:
        raise RuntimeError('The prospective protocol must remain scoped to the 2026 season.')
    return protocol


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def validate_frozen_rules() -> None:
    protocol = load_protocol()
    general = protocol['rules']['general']
    fcs = protocol['rules']['fcs']

    edge = float(general['qualify_edge_points'])
    total = float(general['minimum_total'])
    fixture = pd.Series({
        'closing_total': total,
        'pred_market_residual': -edge,
        'nws_status': 'ok',
        'start_time_tbd': False,
    })
    if classify_row(fixture)[0] != 'QUALIFIES':
        raise RuntimeError('Frozen general threshold no longer qualifies at the protocol boundary.')
    weaker = fixture.copy()
    weaker['pred_market_residual'] = -(edge - 0.001)
    if classify_row(weaker)[0] == 'QUALIFIES':
        raise RuntimeError('Production general edge threshold is looser than the frozen protocol.')
    lower_total = fixture.copy()
    lower_total['closing_total'] = total - 0.001
    if classify_row(lower_total)[0] == 'QUALIFIES':
        raise RuntimeError('Production general total threshold is looser than the frozen protocol.')

    checks = [
        ('FCS qualify edge', float(fcs['qualify_edge_points']), float(FCS_QUALIFY_EDGE)),
        ('FCS minimum total', float(fcs['minimum_total']), float(FCS_QUALIFY_TOTAL)),
    ]
    drift = [f'{name}: protocol={expected:g}, code={actual:g}' for name, expected, actual in checks if expected != actual]
    if drift:
        raise RuntimeError('Frozen 2026 prospective protocol no longer matches production FCS rules: ' + '; '.join(drift))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz='UTC')


def _iso_utc(value: pd.Timestamp | datetime) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts.isoformat().replace('+00:00', 'Z')


def _run_metadata(now: pd.Timestamp, kind: str) -> dict[str, Any]:
    protocol = load_protocol()
    event_name = os.getenv('PROSPECTIVE_EVENT_NAME', os.getenv('GITHUB_EVENT_NAME', '')).strip()
    event_schedule = os.getenv('PROSPECTIVE_EVENT_SCHEDULE', '').strip()
    official_policy = protocol['official_entry_policy']
    eligible = (
        event_name == str(official_policy['eligible_event_name'])
        and event_schedule in {str(v) for v in official_policy.get('eligible_crons', [])}
    )
    return {
        'record_kind': kind,
        'snapshot_timestamp_utc': _iso_utc(now),
        'protocol_version': str(protocol['protocol_version']),
        'protocol_sha256': protocol_sha256(),
        'github_event_name': event_name,
        'github_event_schedule': event_schedule,
        'github_run_id': os.getenv('PROSPECTIVE_RUN_ID', os.getenv('GITHUB_RUN_ID', 'local')),
        'github_run_attempt': int(os.getenv('PROSPECTIVE_RUN_ATTEMPT', os.getenv('GITHUB_RUN_ATTEMPT', '1')) or 1),
        'source_git_sha': os.getenv('PROSPECTIVE_SHA', os.getenv('GITHUB_SHA', 'local')),
        'official_eligible': bool(eligible),
    }


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator='\n').encode('utf-8')


def _immutable_write(frame: pd.DataFrame, directory: Path, prefix: str, now: pd.Timestamp) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = _csv_bytes(frame)
    digest = hashlib.sha256(payload).hexdigest()
    run_id = str(frame['github_run_id'].iloc[0]) if 'github_run_id' in frame.columns and len(frame) else os.getenv('GITHUB_RUN_ID', 'local')
    attempt = int(frame['github_run_attempt'].iloc[0]) if 'github_run_attempt' in frame.columns and len(frame) else 1
    stamp = pd.Timestamp(now).tz_convert('UTC').strftime('%Y%m%dT%H%M%SZ')
    safe_run = re.sub(r'[^A-Za-z0-9_.-]+', '-', run_id)
    path = directory / f'{prefix}_{stamp}_run{safe_run}_a{attempt}_{digest[:12]}.csv'
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite immutable prospective file: {path}')
    with open(path, 'xb') as f:
        f.write(payload)
    return path


def verify_immutable_files() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kind, directory in [('board_snapshot', SNAPSHOT_DIR), ('close_capture', CLOSE_DIR)]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob('*.csv')):
            match = HASH_RE.search(path.name)
            if not match:
                raise RuntimeError(f'Immutable prospective file is missing its content hash suffix: {path}')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            expected_prefix = match.group(1)
            if actual[:12] != expected_prefix:
                raise RuntimeError(
                    f'Prospective immutability check failed for {path.name}: '
                    f'filename hash={expected_prefix}, content hash={actual[:12]}'
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


def snapshot_board(now: pd.Timestamp | None = None) -> Path:
    validate_frozen_rules()
    now = now or _utc_now()
    board = read_df('outputs/weekly_board.csv').copy()
    if board.empty:
        raise RuntimeError('weekly_board.csv is empty; prospective snapshot was not created.')

    metadata = _run_metadata(now, 'board_snapshot')
    for key, value in reversed(list(metadata.items())):
        board.insert(0, key, value)

    protocol = load_protocol()
    board.insert(len(metadata), 'official_minimum_lead_minutes', int(protocol['official_entry_policy']['minimum_lead_minutes']))
    board.insert(len(metadata) + 1, 'frozen_general_edge', float(protocol['rules']['general']['qualify_edge_points']))
    board.insert(len(metadata) + 2, 'frozen_general_total_min', float(protocol['rules']['general']['minimum_total']))
    board.insert(len(metadata) + 3, 'frozen_fcs_edge', float(protocol['rules']['fcs']['qualify_edge_points']))
    board.insert(len(metadata) + 4, 'frozen_fcs_total_min', float(protocol['rules']['fcs']['minimum_total']))

    path = _immutable_write(board, SNAPSHOT_DIR, 'board', now)
    verify_immutable_files()
    print(f'Wrote immutable prospective board snapshot: {path.relative_to(ROOT)}')
    print(
        f"Snapshot official-eligible={metadata['official_eligible']} "
        f"event={metadata['github_event_name'] or 'unknown'} "
        f"schedule={metadata['github_event_schedule'] or 'none'}"
    )
    return path


def _merge_current_lines(
    board: pd.DataFrame,
    client: CFBDClient,
    season: int,
    season_type: str,
) -> pd.DataFrame:
    settings = get_settings()
    preferred = settings['cfbd'].get('preferred_line_providers', [])
    line_parts: list[pd.DataFrame] = []
    weeks = sorted({int(v) for v in board.get('week', pd.Series(dtype='float64')).dropna().tolist()})
    for week in weeks:
        records = client.get('/lines', {'year': season, 'week': week, 'seasonType': season_type})
        part = normalize_lines(records)
        if not part.empty:
            line_parts.append(part)
    lines = pd.concat(line_parts, ignore_index=True) if line_parts else pd.DataFrame()

    selected = pick_total(lines, preferred)
    out = board.merge(selected, on='game_id', how='left')
    context = line_market_context(lines)
    if not context.empty:
        out = out.merge(context, on='game_id', how='left')
        out['selected_vs_market_median'] = out['closing_total'] - out['line_total_median']
    else:
        for col in ['line_provider_count', 'line_total_min', 'line_total_max', 'line_total_median', 'line_total_range', 'selected_vs_market_median']:
            out[col] = np.nan

    out['line_source'] = np.where(out['closing_total'].notna(), 'CFBD', '')
    out, oddspapi_stats = apply_fcs_oddspapi_fallback(out, preferred)
    out, odds_api_stats = apply_fcs_odds_fallback(out, preferred)
    print(
        f"Near-kickoff line capture: CFBD + FCS fallbacks; "
        f"OddsPapi status={oddspapi_stats.get('oddspapi_status', 'unknown')}, "
        f"secondary Odds API status={odds_api_stats.get('odds_api_status', 'unknown')}."
    )
    return out


def capture_close(now: pd.Timestamp | None = None) -> Path | None:
    validate_frozen_rules()
    protocol = load_protocol()
    settings = get_settings()
    season = int(protocol['season'])
    season_type = settings['data'].get('season_type', 'regular')
    window_minutes = int(protocol['closing_benchmark_policy']['capture_window_minutes'])
    now = now or _utc_now()

    client = CFBDClient()
    records = client.get('/games', {'year': season, 'seasonType': season_type})
    games = normalize_games(records, season)
    if games.empty:
        print('No games returned; no close capture written.')
        return None

    end = now + pd.Timedelta(minutes=window_minutes)
    imminent = games[
        games['start_date'].notna()
        & games['start_date'].ge(now)
        & games['start_date'].le(end)
    ].copy()
    if imminent.empty:
        print(f'No games kick off in the next {window_minutes} minutes; no close capture written.')
        return None

    imminent['division_track'] = division_track(imminent)
    market = _merge_current_lines(imminent, client, season, season_type)
    market['benchmark_close_total'] = pd.to_numeric(market.get('line_total_median'), errors='coerce')
    market['benchmark_close_total'] = market['benchmark_close_total'].fillna(pd.to_numeric(market.get('closing_total'), errors='coerce'))
    market['capture_lead_minutes'] = (
        pd.to_datetime(market['start_date'], utc=True, errors='coerce') - now
    ).dt.total_seconds() / 60.0

    metadata = _run_metadata(now, 'close_capture')
    for key, value in reversed(list(metadata.items())):
        market.insert(0, key, value)

    keep = [c for c in [
        *metadata.keys(),
        'season', 'week', 'game_id', 'start_date', 'away_team', 'home_team', 'division_track',
        'closing_total', 'benchmark_close_total', 'line_provider', 'line_source', 'line_provider_count',
        'line_total_min', 'line_total_max', 'line_total_median', 'line_total_range',
        'selected_vs_market_median', 'odds_match_confidence', 'capture_lead_minutes',
        'home_classification', 'away_classification',
    ] if c in market.columns]
    market = market[keep].copy()

    path = _immutable_write(market, CLOSE_DIR, 'close', now)
    verify_immutable_files()
    available = int(pd.to_numeric(market['benchmark_close_total'], errors='coerce').notna().sum())
    print(
        f'Wrote immutable near-kickoff market capture for {len(market)} game(s): '
        f'{available} with a benchmark total.'
    )
    return path


def _read_immutable(directory: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if not directory.exists():
        return pd.DataFrame()
    for path in sorted(directory.glob('*.csv')):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame['_immutable_file'] = str(path.relative_to(ROOT)).replace('\\', '/')
        frame['_immutable_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def select_official_entries(snapshots: pd.DataFrame, protocol: dict[str, Any] | None = None) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    protocol = protocol or load_protocol()
    minimum_lead = float(protocol['official_entry_policy']['minimum_lead_minutes'])

    work = snapshots.copy()
    if 'official_eligible' not in work.columns:
        work['official_eligible'] = False
    work['official_eligible'] = work['official_eligible'].map(_bool)
    work['snapshot_timestamp_utc'] = pd.to_datetime(work['snapshot_timestamp_utc'], utc=True, errors='coerce')
    work['start_date'] = pd.to_datetime(work['start_date'], utc=True, errors='coerce')
    work['github_run_attempt'] = pd.to_numeric(work.get('github_run_attempt'), errors='coerce').fillna(1).astype(int)
    work['entry_lead_minutes'] = (work['start_date'] - work['snapshot_timestamp_utc']).dt.total_seconds() / 60.0

    eligible = work[
        work['official_eligible']
        & work['snapshot_timestamp_utc'].notna()
        & work['start_date'].notna()
        & work['entry_lead_minutes'].ge(minimum_lead)
    ].copy()
    if eligible.empty:
        return pd.DataFrame()

    eligible = eligible.sort_values(['github_run_id', 'game_id', 'github_run_attempt', 'snapshot_timestamp_utc'])
    eligible = eligible.drop_duplicates(['github_run_id', 'game_id'], keep='first')
    eligible = eligible.sort_values(['game_id', 'snapshot_timestamp_utc'])
    official = eligible.groupby('game_id', as_index=False, sort=False).tail(1).copy()
    official['official_entry'] = True
    return official.sort_values(['start_date', 'game_id']).reset_index(drop=True)


def select_benchmark_closes(close_captures: pd.DataFrame) -> pd.DataFrame:
    if close_captures.empty:
        return pd.DataFrame(columns=['game_id', 'benchmark_close_total'])
    work = close_captures.copy()
    work['snapshot_timestamp_utc'] = pd.to_datetime(work['snapshot_timestamp_utc'], utc=True, errors='coerce')
    work['start_date'] = pd.to_datetime(work['start_date'], utc=True, errors='coerce')
    work['benchmark_close_total'] = pd.to_numeric(work.get('benchmark_close_total'), errors='coerce')
    work = work[
        work['benchmark_close_total'].notna()
        & work['snapshot_timestamp_utc'].notna()
        & work['start_date'].notna()
        & work['snapshot_timestamp_utc'].lt(work['start_date'])
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=['game_id', 'benchmark_close_total'])
    work['close_capture_lead_minutes'] = (work['start_date'] - work['snapshot_timestamp_utc']).dt.total_seconds() / 60.0
    work = work.sort_values(['game_id', 'snapshot_timestamp_utc'])
    selected = work.groupby('game_id', as_index=False, sort=False).tail(1).copy()
    rename = {
        'snapshot_timestamp_utc': 'benchmark_close_captured_at_utc',
        'line_provider': 'benchmark_close_provider',
        'line_source': 'benchmark_close_source',
        'line_provider_count': 'benchmark_close_provider_count',
        '_immutable_file': 'benchmark_close_immutable_file',
        '_immutable_sha256': 'benchmark_close_immutable_sha256',
    }
    selected = selected.rename(columns=rename)
    keep = [c for c in [
        'game_id', 'benchmark_close_total', 'benchmark_close_captured_at_utc',
        'close_capture_lead_minutes', 'benchmark_close_provider', 'benchmark_close_source',
        'benchmark_close_provider_count', 'benchmark_close_immutable_file', 'benchmark_close_immutable_sha256',
    ] if c in selected.columns]
    return selected[keep]


def _final_results(client: CFBDClient | None = None) -> pd.DataFrame:
    protocol = load_protocol()
    settings = get_settings()
    client = client or CFBDClient()
    try:
        records = client.get('/games', {
            'year': int(protocol['season']),
            'seasonType': settings['data'].get('season_type', 'regular'),
        })
    except Exception as exc:
        print(f'Final-result refresh unavailable: {type(exc).__name__}: {exc}')
        return pd.DataFrame(columns=['game_id', 'final_home_points', 'final_away_points', 'actual_total_points'])

    games = pd.json_normalize(records)
    if games.empty:
        return pd.DataFrame(columns=['game_id', 'final_home_points', 'final_away_points', 'actual_total_points'])
    games = games.rename(columns={
        'id': 'game_id',
        'homePoints': 'final_home_points',
        'awayPoints': 'final_away_points',
    })
    for col in ['final_home_points', 'final_away_points']:
        games[col] = pd.to_numeric(games.get(col), errors='coerce')
    games['actual_total_points'] = games['final_home_points'] + games['final_away_points']
    keep = [c for c in ['game_id', 'final_home_points', 'final_away_points', 'actual_total_points'] if c in games.columns]
    return games[keep].drop_duplicates('game_id')


def _settle(actual: Any, total: Any, side: str) -> str:
    actual_num = pd.to_numeric(pd.Series([actual]), errors='coerce').iloc[0]
    total_num = pd.to_numeric(pd.Series([total]), errors='coerce').iloc[0]
    if pd.isna(actual_num) or pd.isna(total_num):
        return ''
    if actual_num == total_num:
        return 'push'
    if str(side).lower() == 'under':
        return 'win' if actual_num < total_num else 'loss'
    if str(side).lower() == 'over':
        return 'win' if actual_num > total_num else 'loss'
    return ''


def _paper_units(result: str, protocol: dict[str, Any]) -> float | None:
    grading = protocol['paper_grading']
    if result == 'win':
        return float(grading['win_profit_units'])
    if result == 'loss':
        return float(grading['loss_profit_units'])
    if result == 'push':
        return float(grading['push_profit_units'])
    return None


def grade_official_entries(
    official: pd.DataFrame,
    closes: pd.DataFrame,
    finals: pd.DataFrame,
    protocol: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if official.empty:
        return pd.DataFrame()
    protocol = protocol or load_protocol()
    out = official.copy()
    if not closes.empty:
        out = out.merge(closes, on='game_id', how='left')
    if not finals.empty:
        out = out.merge(finals, on='game_id', how='left')

    out['entry_total'] = pd.to_numeric(out.get('closing_total'), errors='coerce')
    out['actual_total_points'] = pd.to_numeric(out.get('actual_total_points'), errors='coerce')
    out['entry_market_residual_actual'] = out['actual_total_points'] - out['entry_total']
    qualifying_status = str(protocol['paper_grading']['qualifying_status'])
    status = out['status'].astype(str) if 'status' in out.columns else pd.Series('', index=out.index)
    out['paper_qualifier'] = status.eq(qualifying_status)

    model_side = out['model_side'] if 'model_side' in out.columns else pd.Series('', index=out.index)
    all_results = [
        _settle(actual, total, side)
        for actual, total, side in zip(out['actual_total_points'], out['entry_total'], model_side)
    ]
    out['result_vs_entry_line'] = all_results
    out['paper_result'] = np.where(out['paper_qualifier'], out['result_vs_entry_line'], '')
    out['paper_units_1u'] = [
        _paper_units(result, protocol) if qualifier else np.nan
        for result, qualifier in zip(out['result_vs_entry_line'], out['paper_qualifier'])
    ]

    if 'benchmark_close_total' not in out.columns:
        out['benchmark_close_total'] = np.nan
    out['benchmark_close_total'] = pd.to_numeric(out['benchmark_close_total'], errors='coerce')
    side = model_side.astype(str).str.lower()
    under_clv = out['entry_total'] - out['benchmark_close_total']
    over_clv = out['benchmark_close_total'] - out['entry_total']
    out['clv_points'] = np.where(side.eq('under'), under_clv, np.where(side.eq('over'), over_clv, np.nan))
    out['positive_clv'] = np.where(out['clv_points'].notna(), out['clv_points'] > 0, np.nan)
    return out


def summarize_graded(graded: pd.DataFrame) -> pd.DataFrame:
    if graded.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [('ALL', graded)]
    if 'model_track' in graded.columns:
        groups.extend((str(name), group) for name, group in graded.groupby('model_track', dropna=False))

    for label, frame in groups:
        qualifiers = frame[frame['paper_qualifier'].map(_bool)].copy()
        settled = qualifiers[qualifiers['paper_result'].isin(['win', 'loss', 'push'])].copy()
        wins = int(settled['paper_result'].eq('win').sum())
        losses = int(settled['paper_result'].eq('loss').sum())
        pushes = int(settled['paper_result'].eq('push').sum())
        n = wins + losses
        net = float(pd.to_numeric(settled.get('paper_units_1u'), errors='coerce').sum()) if len(settled) else 0.0
        clv = pd.to_numeric(qualifiers.get('clv_points'), errors='coerce')
        rows.append({
            'scope': label,
            'official_games': int(len(frame)),
            'qualifying_entries': int(len(qualifiers)),
            'settled_qualifying_entries': int(len(settled)),
            'graded_ex_pushes': int(n),
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'hit_rate_ex_pushes': wins / n if n else np.nan,
            'net_units_1u_at_-110': net,
            'roi_per_graded_entry': net / n if n else np.nan,
            'qualifiers_with_clv': int(clv.notna().sum()),
            'average_clv_points': float(clv.mean()) if clv.notna().any() else np.nan,
            'median_clv_points': float(clv.median()) if clv.notna().any() else np.nan,
            'positive_clv_rate': float((clv.dropna() > 0).mean()) if clv.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def _write_summary_markdown(summary: pd.DataFrame, graded: pd.DataFrame, manifest: pd.DataFrame) -> None:
    protocol = load_protocol()
    out = PROSPECTIVE_ROOT / 'prospective_summary.md'
    board_count = int(manifest['kind'].eq('board_snapshot').sum()) if not manifest.empty and 'kind' in manifest.columns else 0
    close_count = int(manifest['kind'].eq('close_capture').sum()) if not manifest.empty and 'kind' in manifest.columns else 0
    lines = [
        '# 2026 Prospective Validation Ledger',
        '',
        f"Protocol version: **{protocol['protocol_version']}**",
        '',
        f"Protocol SHA-256: `{protocol_sha256()}`",
        '',
        'The immutable snapshot files are the source of truth. Derived ledgers and summaries can be rebuilt from them at any time.',
        '',
        '## Frozen rules',
        '',
        f"- General: HGB UNDER edge >= {float(protocol['rules']['general']['qualify_edge_points']):g}, total >= {float(protocol['rules']['general']['minimum_total']):g}.",
        f"- FCS: FCS-only HGB UNDER edge >= {float(protocol['rules']['fcs']['qualify_edge_points']):g}, total >= {float(protocol['rules']['fcs']['minimum_total']):g}.",
        f"- Official entry: latest eligible scheduled snapshot at least {int(protocol['official_entry_policy']['minimum_lead_minutes'])} minutes before kickoff.",
        f"- CLV benchmark: latest immutable pre-kickoff market capture within the {int(protocol['closing_benchmark_policy']['capture_window_minutes'])}-minute capture window; median across books when available.",
        '',
        '## Prospective results',
        '',
        summary.to_markdown(index=False) if not summary.empty else '_No official qualifying entries have settled yet._',
        '',
        '## Data integrity',
        '',
        f'- Immutable board snapshots: {board_count}',
        f'- Immutable close captures: {close_count}',
        f'- Official game entries selected: {len(graded)}',
        '',
        'Every immutable CSV filename contains the first 12 characters of its SHA-256 content hash. The workflow verifies those hashes before rebuilding derived results.',
        '',
        '## Interpretation',
        '',
        'These are prospective paper results, not a retrospective re-optimization. Protocol changes require a new version and apply prospectively only. Missing close captures remain missing rather than being backfilled from an unverified postgame line.',
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')


def build_ledger(client: CFBDClient | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_frozen_rules()
    manifest = verify_immutable_files()
    snapshots = _read_immutable(SNAPSHOT_DIR)
    close_captures = _read_immutable(CLOSE_DIR)
    protocol = load_protocol()

    official = select_official_entries(snapshots, protocol)
    closes = select_benchmark_closes(close_captures)
    finals = _final_results(client)
    graded = grade_official_entries(official, closes, finals, protocol)
    summary = summarize_graded(graded)

    write_df(official, PROSPECTIVE_ROOT / 'official_entries.csv')
    write_df(graded, PROSPECTIVE_ROOT / 'graded_entries.csv')
    write_df(summary, PROSPECTIVE_ROOT / 'summary.csv')
    _write_summary_markdown(summary, graded, manifest)

    qualifier_count = int(graded['paper_qualifier'].map(_bool).sum()) if not graded.empty and 'paper_qualifier' in graded.columns else 0
    print(
        f'Prospective ledger rebuilt: {len(official)} official game entries, '
        f'{qualifier_count} qualifying entries.'
    )
    return graded, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Immutable 2026 prospective validation and CLV ledger.')
    parser.add_argument('command', choices=['snapshot-board', 'capture-close', 'build', 'verify', 'validate-protocol'])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == 'snapshot-board':
        snapshot_board()
    elif args.command == 'capture-close':
        capture_close()
    elif args.command == 'build':
        build_ledger()
    elif args.command == 'verify':
        manifest = verify_immutable_files()
        print(f'Immutable prospective files verified: {len(manifest)}')
    elif args.command == 'validate-protocol':
        validate_frozen_rules()
        print('Frozen 2026 protocol matches production thresholds.')


if __name__ == '__main__':
    main()
