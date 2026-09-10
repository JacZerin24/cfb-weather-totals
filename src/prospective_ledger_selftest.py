from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .close_capture_gate import should_capture
from .prospective_integrity_rebuild import (
    apply_grading_exclusions,
    apply_official_eligibility_corrections,
    final_results_completed,
    summarize_with_exclusions,
)
from .prospective_ledger import (
    _csv_bytes,
    _immutable_write,
    grade_official_entries,
    load_protocol,
    protocol_sha256,
    select_benchmark_closes,
    select_official_entries,
    validate_frozen_rules,
)
from .shadow_integrity_rebuild import _apply_shadow_eligibility_corrections
from .utils import ROOT


class FakeClient:
    def __init__(self, records: list[dict]):
        self.records = records

    def get(self, path: str, params: dict) -> list[dict]:
        assert path == '/games'
        return self.records


def main() -> None:
    validate_frozen_rules()
    protocol = load_protocol()
    assert protocol['protocol_version'] == '2026.5'
    assert protocol.get('supersedes') == '2026.4'
    assert int(protocol['closing_benchmark_policy']['capture_window_minutes']) == 90
    assert int(protocol['closing_benchmark_policy']['preflight_window_minutes']) == 105
    assert len(protocol_sha256()) == 64

    expected_official_crons = {
        '17 10 * * 4,5',
        '17 11 * * 4,5',
        '17 13 * * 4,5',
        '17 14 * * 4,5',
        '17 6 * * 6',
        '17 7 * * 6',
        '17 11 * 11 2,3',
        '17 14 * 11 2,3',
    }
    assert set(map(str, protocol['official_entry_policy']['eligible_crons'])) == expected_official_crons
    assert '17 8 * * 6' not in expected_official_crons
    assert '17 11 * * 6' not in expected_official_crons
    assert '17 12 * * 6' not in expected_official_crons
    # Bowl/CFP-only operational crons must not accidentally become official
    # regular-season prospective entries.
    assert '17 11 * 12,1 2,3' not in expected_official_crons
    assert '17 14 * 12,1 2,3' not in expected_official_crons

    kickoff = pd.Timestamp('2026-09-06T00:00:00Z')
    snapshots = pd.DataFrame([
        {
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-04T16:56:00Z',
            'official_eligible': True,
            'github_event_name': 'schedule',
            'github_event_schedule': '17 13 * * 4,5',
            'github_run_id': 'friday',
            'github_run_attempt': 1,
            'closing_total': 58.5,
            'status': 'NO PLAY',
            'model_side': 'over',
            'model_track': 'FCS-only HGB',
        },
        # Regression fixture for the actual Sep. 5 schedule-metadata bug. The
        # immutable run was a real scheduled production build but was stamped
        # false before the protocol cron list caught up with the workflow.
        {
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-05T10:38:59Z',
            'official_eligible': False,
            'github_event_name': 'schedule',
            'github_event_schedule': '17 6 * * 6',
            'github_run_id': '33961120017',
            'github_run_attempt': 1,
            'closing_total': 59.5,
            'status': 'QUALIFIES',
            'model_side': 'under',
            'model_track': 'FCS-only HGB',
        },
        # A different false/manual snapshot must remain ineligible.
        {
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-05T11:00:00Z',
            'official_eligible': False,
            'github_event_name': 'workflow_dispatch',
            'github_event_schedule': '',
            'github_run_id': 'manual',
            'github_run_attempt': 1,
            'closing_total': 61.0,
            'status': 'QUALIFIES',
            'model_side': 'under',
            'model_track': 'FCS-only HGB',
        },
    ])

    corrected = apply_official_eligibility_corrections(snapshots, protocol)
    corrected_row = corrected[corrected['github_run_id'].astype(str).eq('33961120017')].iloc[0]
    manual_row = corrected[corrected['github_run_id'].astype(str).eq('manual')].iloc[0]
    assert bool(corrected_row['official_eligible'])
    assert bool(corrected_row['eligibility_integrity_override'])
    assert not bool(manual_row['official_eligible'])

    official = select_official_entries(corrected, protocol)
    assert len(official) == 1
    assert str(official.iloc[0]['github_run_id']) == '33961120017'
    assert float(official.iloc[0]['closing_total']) == 59.5
    assert str(official.iloc[0]['status']) == 'QUALIFIES'

    # The research shadows use the same tightly scoped run-level correction.
    shadow_fixture = pd.DataFrame([{
        'github_run_id': '33961120017',
        'github_event_name': 'schedule',
        'github_event_schedule': '17 6 * * 6',
        'official_evaluation_eligible': False,
    }])
    orientation_protocol = __import__(
        'src.orientation_shadow_grade', fromlist=['load_protocol']
    ).load_protocol()
    shadow_fixed = _apply_shadow_eligibility_corrections(shadow_fixture, orientation_protocol)
    assert bool(shadow_fixed.iloc[0]['official_evaluation_eligible'])
    assert bool(shadow_fixed.iloc[0]['shadow_eligibility_integrity_override'])
    assert set(map(str, orientation_protocol['official_entry_policy']['eligible_crons'])) == expected_official_crons

    joint_protocol = __import__(
        'src.joint_core_shadow_grade', fromlist=['load_protocol']
    ).load_protocol()
    assert set(map(str, joint_protocol['official_entry_policy']['eligible_crons'])) == expected_official_crons

    closes = pd.DataFrame([
        {
            'record_kind': 'close_capture',
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-05T22:30:00Z',
            'benchmark_close_total': 58.0,
            'github_event_name': 'schedule',
            'github_run_attempt': 1,
            'line_provider': 'Book A',
            'line_source': 'fixture',
            'line_provider_count': 2,
        },
        {
            'record_kind': 'close_capture',
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-05T23:30:00Z',
            'benchmark_close_total': 57.5,
            'github_event_name': 'schedule',
            'github_run_attempt': 1,
            'line_provider': 'Book B',
            'line_source': 'fixture',
            'line_provider_count': 3,
        },
        # Manual/rerun captures can never replace a valid scheduled capture.
        {
            'record_kind': 'close_capture',
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-05T23:40:00Z',
            'benchmark_close_total': 56.5,
            'github_event_name': 'workflow_dispatch',
            'github_run_attempt': 1,
        },
        {
            'record_kind': 'close_capture',
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-05T23:45:00Z',
            'benchmark_close_total': 56.0,
            'github_event_name': 'schedule',
            'github_run_attempt': 2,
        },
    ])
    selected_close = select_benchmark_closes(closes)
    assert len(selected_close) == 1
    assert float(selected_close.iloc[0]['benchmark_close_total']) == 57.5
    assert abs(float(selected_close.iloc[0]['close_capture_lead_minutes']) - 30.0) < 0.01

    # Generic final-score protection: incomplete 0-0 placeholders are absent.
    finals = final_results_completed(FakeClient([
        {'id': 10, 'homePoints': 0, 'awayPoints': 0, 'completed': False},
        {'id': 11, 'homePoints': 28, 'awayPoints': 21, 'completed': True},
    ]))
    assert list(finals['game_id'].astype(int)) == [11]
    assert float(finals.iloc[0]['actual_total_points']) == 49.0

    # Explicit stale/postponed exclusions preserve the prospective qualifier
    # while removing it from wins/losses, units, ROI, and CLV.
    grading_official = pd.DataFrame([
        {
            'game_id': 401866625,
            'away_team': 'Western Carolina',
            'home_team': 'Campbell',
            'closing_total': 67.5,
            'status': 'QUALIFIES',
            'model_side': 'under',
            'model_track': 'FCS-only HGB',
        },
        {
            'game_id': 1,
            'away_team': 'Illinois State',
            'home_team': 'Western Illinois',
            'closing_total': 59.5,
            'status': 'QUALIFIES',
            'model_side': 'under',
            'model_track': 'FCS-only HGB',
        },
    ])
    grading_finals = pd.DataFrame([
        {'game_id': 401866625, 'final_home_points': 0, 'final_away_points': 0, 'actual_total_points': 0},
        {'game_id': 1, 'final_home_points': 41, 'final_away_points': 10, 'actual_total_points': 51},
    ])
    graded = grade_official_entries(grading_official, pd.DataFrame(), grading_finals, protocol)
    graded = apply_grading_exclusions(graded, protocol)
    campbell = graded[graded['game_id'].eq(401866625)].iloc[0]
    illinois = graded[graded['game_id'].eq(1)].iloc[0]
    assert bool(campbell['paper_qualifier'])
    assert campbell['paper_result'] == 'excluded'
    assert pd.isna(campbell['paper_units_1u'])
    assert pd.isna(campbell['clv_points'])
    assert illinois['paper_result'] == 'win'

    summary = summarize_with_exclusions(graded)
    overall = summary[summary['scope'].eq('ALL')].iloc[0]
    assert int(overall['qualifying_entries']) == 2
    assert int(overall['settled_qualifying_entries']) == 1
    assert int(overall['excluded_qualifying_entries']) == 1
    assert int(overall['wins']) == 1
    assert int(overall['losses']) == 0

    fixture = pd.DataFrame([{
        'github_run_id': 'fixture',
        'github_run_attempt': 1,
        'value': 42,
    }])
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        now = pd.Timestamp('2026-09-05T13:00:00Z')
        path = _immutable_write(fixture, directory, 'board', now)
        digest = hashlib.sha256(_csv_bytes(fixture)).hexdigest()
        assert digest[:12] in path.name
        try:
            _immutable_write(fixture, directory, 'board', now)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Immutable writer allowed an overwrite.')

    # Zero-API close-capture gate: no nearby kickoff means no expensive API
    # call; a tracked game within the guard window opens the authoritative path.
    with TemporaryDirectory() as tmp:
        gate_path = Path(tmp) / 'weekly_board.csv'
        pd.DataFrame([{
            'game_id': 99,
            'start_date': '2026-11-03T23:00:00Z',
            'away_team': 'Away',
            'home_team': 'Home',
        }]).to_csv(gate_path, index=False)
        sources = (gate_path,)
        assert not should_capture('2026-11-03T20:00:00Z', 105, sources)
        assert should_capture('2026-11-03T21:30:00Z', 105, sources)

    weekly_workflow = (ROOT / '.github/workflows/weekly-cfb-weather.yml').read_text(encoding='utf-8')
    assert 'central-time-gate' in weekly_workflow
    assert 'python -m src.prospective_integrity_rebuild' in weekly_workflow
    assert 'python -m src.site_smoke_selftest' in weekly_workflow
    assert 'deploy-site:' in weekly_workflow
    assert "ref: ${{ needs.weekly-paper-run.outputs.site_sha }}" in weekly_workflow
    for cron in expected_official_crons:
        assert cron in weekly_workflow, f'Official protocol cron is not scheduled: {cron}'

    watchdog_workflow = (ROOT / '.github/workflows/weekly-safety-watchdog.yml').read_text(encoding='utf-8')
    assert 'actions: write' in watchdog_workflow
    assert '/dispatches' in watchdog_workflow
    assert 'weekly-paper-run' in watchdog_workflow
    assert 'SAFETY_HOUR=1' in watchdog_workflow
    assert '47 11 * 11,12,1 2,3' in watchdog_workflow

    close_workflow = (ROOT / '.github/workflows/prospective-close-capture.yml').read_text(encoding='utf-8')
    assert 'workflow_dispatch' not in close_workflow, 'Close benchmark workflow must not allow manual captures.'
    assert 'github.run_attempt == 1' in close_workflow
    assert 'python -m src.close_capture_gate --window-minutes 105' in close_workflow
    assert "steps.preflight.outputs.should_capture == 'true'" in close_workflow
    for cron in protocol['closing_benchmark_policy']['capture_crons']:
        assert str(cron) in close_workflow

    grade_workflow = (ROOT / '.github/workflows/prospective-grade.yml').read_text(encoding='utf-8')
    assert 'python -m src.prospective_integrity_rebuild' in grade_workflow
    assert 'python -m src.shadow_integrity_rebuild orientation' in grade_workflow
    assert 'python -m src.shadow_integrity_rebuild joint-core' in grade_workflow
    for cron in protocol['paper_grading']['postgame_grade_crons']:
        assert str(cron) in grade_workflow

    print('Prospective protocol 2026.5, schedule alignment, API preflight, immutable selection, completed-game grading, exclusions, CLV, and shadow-integrity checks passed.')


if __name__ == '__main__':
    main()
