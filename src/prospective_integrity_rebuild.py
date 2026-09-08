from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cfbd_client import CFBDClient
from .prospective_ledger import (
    CLOSE_DIR,
    PROSPECTIVE_ROOT,
    SNAPSHOT_DIR,
    _read_immutable,
    grade_official_entries,
    load_protocol,
    protocol_sha256,
    select_benchmark_closes,
    select_official_entries,
    summarize_graded,
    validate_frozen_rules,
    verify_immutable_files,
)
from .utils import get_settings, write_df


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def _integrity_config(protocol: dict[str, Any]) -> dict[str, Any]:
    return protocol.get('data_integrity_corrections', {}) or {}


def apply_official_eligibility_corrections(
    snapshots: pd.DataFrame,
    protocol: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Repair explicitly documented metadata mistakes without editing immutable files."""
    if snapshots.empty:
        return snapshots.copy()
    protocol = protocol or load_protocol()
    work = snapshots.copy()
    work['eligibility_integrity_override'] = False
    work['eligibility_integrity_reason'] = ''

    run_id = work.get('github_run_id', pd.Series('', index=work.index)).astype(str)
    event_name = work.get('github_event_name', pd.Series('', index=work.index)).astype(str)
    event_schedule = work.get('github_event_schedule', pd.Series('', index=work.index)).astype(str)

    for correction in _integrity_config(protocol).get('official_entry_eligibility_overrides', []):
        mask = run_id.eq(str(correction.get('github_run_id', '')))
        expected_event = str(correction.get('github_event_name', '')).strip()
        expected_schedule = str(correction.get('github_event_schedule', '')).strip()
        if expected_event:
            mask &= event_name.eq(expected_event)
        if expected_schedule:
            mask &= event_schedule.eq(expected_schedule)
        if not mask.any():
            continue
        work.loc[mask, 'official_eligible'] = True
        work.loc[mask, 'eligibility_integrity_override'] = True
        work.loc[mask, 'eligibility_integrity_reason'] = str(correction.get('reason', 'Documented eligibility metadata correction.'))

    return work


def final_results_completed(client: CFBDClient | None = None) -> pd.DataFrame:
    """Return only CFBD games explicitly marked completed.

    This prevents postponed/suspended 0-0 placeholders from being settled as finals.
    """
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
    games['game_id'] = pd.to_numeric(games.get('game_id'), errors='coerce')
    for col in ['final_home_points', 'final_away_points']:
        games[col] = pd.to_numeric(games.get(col), errors='coerce')
    completed = games.get('completed', pd.Series(False, index=games.index)).map(_bool)
    games['actual_total_points'] = games['final_home_points'] + games['final_away_points']
    games = games[completed & games['actual_total_points'].notna()].copy()
    keep = ['game_id', 'final_home_points', 'final_away_points', 'actual_total_points']
    return games[keep].drop_duplicates('game_id')


def apply_grading_exclusions(
    graded: pd.DataFrame,
    protocol: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if graded.empty:
        return graded.copy()
    protocol = protocol or load_protocol()
    out = graded.copy()
    paper_qualifier = out.get('paper_qualifier', pd.Series(False, index=out.index)).map(_bool)
    paper_result = out.get('paper_result', pd.Series('', index=out.index)).astype(str)
    out['grading_disposition'] = np.where(
        paper_result.isin(['win', 'loss', 'push']),
        'SETTLED',
        np.where(paper_qualifier, 'PENDING', 'NOT QUALIFIER'),
    )
    out['grading_exclusion_reason'] = ''

    game_ids = pd.to_numeric(out.get('game_id'), errors='coerce')
    for exclusion in _integrity_config(protocol).get('grading_exclusions', []):
        target = pd.to_numeric(pd.Series([exclusion.get('game_id')]), errors='coerce').iloc[0]
        if pd.isna(target):
            continue
        mask = game_ids.eq(target)
        if not mask.any():
            continue
        disposition = str(exclusion.get('disposition', 'EXCLUDED'))
        reason = str(exclusion.get('reason', 'Documented prospective grading exclusion.'))
        out.loc[mask, 'grading_disposition'] = disposition
        out.loc[mask, 'grading_exclusion_reason'] = reason
        out.loc[mask, 'result_vs_entry_line'] = 'excluded'
        out.loc[mask, 'paper_result'] = 'excluded'
        out.loc[mask, 'paper_units_1u'] = np.nan
        out.loc[mask, 'clv_points'] = np.nan
        out.loc[mask, 'positive_clv'] = np.nan

    return out


def summarize_with_exclusions(graded: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_graded(graded)
    if summary.empty:
        return summary

    for column in ['excluded_qualifying_entries', 'pending_qualifying_entries']:
        summary[column] = 0

    groups: list[tuple[str, pd.DataFrame]] = [('ALL', graded)]
    if 'model_track' in graded.columns:
        groups.extend((str(name), group) for name, group in graded.groupby('model_track', dropna=False))

    for label, frame in groups:
        qualifiers = frame[frame.get('paper_qualifier', pd.Series(False, index=frame.index)).map(_bool)].copy()
        results = qualifiers.get('paper_result', pd.Series('', index=qualifiers.index)).astype(str)
        summary.loc[summary['scope'].eq(label), 'excluded_qualifying_entries'] = int(results.eq('excluded').sum())
        summary.loc[summary['scope'].eq(label), 'pending_qualifying_entries'] = int(results.eq('').sum())

    ordered = list(summary.columns)
    if 'settled_qualifying_entries' in ordered:
        idx = ordered.index('settled_qualifying_entries') + 1
        for col in ['excluded_qualifying_entries', 'pending_qualifying_entries']:
            ordered.remove(col)
            ordered.insert(idx, col)
            idx += 1
        summary = summary[ordered]
    return summary


def _write_summary_markdown(
    summary: pd.DataFrame,
    graded: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    protocol = load_protocol()
    out = PROSPECTIVE_ROOT / 'prospective_summary.md'
    board_count = int(manifest['kind'].eq('board_snapshot').sum()) if not manifest.empty else 0
    close_count = int(manifest['kind'].eq('close_capture').sum()) if not manifest.empty else 0
    override_count = int(graded.get('eligibility_integrity_override', pd.Series(False, index=graded.index)).map(_bool).sum()) if not graded.empty else 0
    excluded = graded[graded.get('paper_result', pd.Series('', index=graded.index)).astype(str).eq('excluded')].copy() if not graded.empty else pd.DataFrame()

    lines = [
        '# 2026 Prospective Validation Ledger',
        '',
        f"Protocol version: **{protocol['protocol_version']}**",
        '',
        f"Protocol SHA-256: `{protocol_sha256()}`",
        '',
        'Immutable board snapshots and close captures remain the source of truth. Derived ledgers may apply only the explicitly documented integrity corrections in the active protocol.',
        '',
        '## Frozen rules',
        '',
        f"- General: HGB UNDER edge >= {float(protocol['rules']['general']['qualify_edge_points']):g}, total >= {float(protocol['rules']['general']['minimum_total']):g}.",
        f"- FCS: FCS-only HGB UNDER edge >= {float(protocol['rules']['fcs']['qualify_edge_points']):g}, total >= {float(protocol['rules']['fcs']['minimum_total']):g}.",
        f"- Official entry: latest eligible scheduled snapshot at least {int(protocol['official_entry_policy']['minimum_lead_minutes'])} minutes before kickoff.",
        f"- CLV benchmark: latest immutable pre-kickoff market capture within the {int(protocol['closing_benchmark_policy']['capture_window_minutes'])}-minute capture window.",
        '',
        '## Prospective results',
        '',
        summary.to_markdown(index=False) if not summary.empty else '_No official entries are available yet._',
        '',
        '## Data-integrity corrections',
        '',
        f'- Official-entry rows selected through a documented eligibility metadata override: {override_count}',
        f'- Qualifying entries explicitly excluded from performance grading: {len(excluded)}',
    ]
    if not excluded.empty:
        for _, row in excluded.iterrows():
            lines.append(
                f"  - {row.get('away_team', '')} at {row.get('home_team', '')} (game {row.get('game_id', '')}): "
                f"{row.get('grading_disposition', 'EXCLUDED')} - {row.get('grading_exclusion_reason', '')}"
            )
    lines += [
        '',
        '## Data integrity',
        '',
        f'- Immutable board snapshots: {board_count}',
        f'- Immutable close captures: {close_count}',
        f'- Official game entries selected: {len(graded)}',
        '',
        'Every immutable CSV filename contains the first 12 characters of its SHA-256 content hash. The rebuild verifies those hashes before selecting entries.',
        '',
        '## Interpretation',
        '',
        'These are prospective paper results, not a retrospective re-optimization. Eligibility corrections are limited to documented automation metadata errors, and postponed/stale entries remain visible but do not count as wins, losses, units, ROI, or CLV.',
    ]
    out.write_text('\n'.join(lines), encoding='utf-8')


def build_integrity_ledger(client: CFBDClient | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_frozen_rules()
    manifest = verify_immutable_files()
    protocol = load_protocol()
    snapshots = apply_official_eligibility_corrections(_read_immutable(SNAPSHOT_DIR), protocol)
    close_captures = _read_immutable(CLOSE_DIR)

    official = select_official_entries(snapshots, protocol)
    closes = select_benchmark_closes(close_captures)
    finals = final_results_completed(client)
    graded = grade_official_entries(official, closes, finals, protocol)
    graded = apply_grading_exclusions(graded, protocol)
    summary = summarize_with_exclusions(graded)

    write_df(official, PROSPECTIVE_ROOT / 'official_entries.csv')
    write_df(graded, PROSPECTIVE_ROOT / 'graded_entries.csv')
    write_df(summary, PROSPECTIVE_ROOT / 'summary.csv')
    _write_summary_markdown(summary, graded, manifest)

    qualifiers = int(graded.get('paper_qualifier', pd.Series(False, index=graded.index)).map(_bool).sum()) if not graded.empty else 0
    excluded = int(graded.get('paper_result', pd.Series('', index=graded.index)).astype(str).eq('excluded').sum()) if not graded.empty else 0
    print(
        f'Prospective integrity ledger rebuilt: {len(official)} official game entries, '
        f'{qualifiers} qualifying entries, {excluded} excluded from performance grading.'
    )
    return graded, summary


def main() -> None:
    build_integrity_ledger()


if __name__ == '__main__':
    main()
