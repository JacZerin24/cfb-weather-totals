from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .prospective_ledger import (
    _csv_bytes,
    _immutable_write,
    grade_official_entries,
    load_protocol,
    protocol_sha256,
    select_benchmark_closes,
    select_official_entries,
    summarize_graded,
    validate_frozen_rules,
)
from .utils import ROOT


def main() -> None:
    validate_frozen_rules()
    protocol = load_protocol()
    assert protocol['protocol_version'] == '2026.2'
    assert protocol.get('supersedes') == '2026.1'
    assert int(protocol['closing_benchmark_policy']['capture_window_minutes']) == 90
    assert len(protocol_sha256()) == 64

    kickoff = pd.Timestamp('2026-09-05T16:00:00Z')
    snapshots = pd.DataFrame([
        {
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-04T14:00:00Z',
            'official_eligible': True,
            'github_run_id': 'friday',
            'github_run_attempt': 1,
            'closing_total': 58.5,
            'status': 'QUALIFIES',
            'model_side': 'under',
            'model_track': 'GENERAL HGB',
        },
        {
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-05T13:00:00Z',
            'official_eligible': True,
            'github_run_id': 'saturday',
            'github_run_attempt': 1,
            'closing_total': 59.0,
            'status': 'QUALIFIES',
            'model_side': 'under',
            'model_track': 'GENERAL HGB',
        },
        # A rerun of the same scheduled model run cannot opportunistically
        # replace the first successful attempt.
        {
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-05T13:20:00Z',
            'official_eligible': True,
            'github_run_id': 'saturday',
            'github_run_attempt': 2,
            'closing_total': 60.0,
            'status': 'QUALIFIES',
            'model_side': 'under',
            'model_track': 'GENERAL HGB',
        },
        # Manual refreshes are archived but never eligible for official entry.
        {
            'game_id': 1,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-05T13:30:00Z',
            'official_eligible': False,
            'github_run_id': 'manual',
            'github_run_attempt': 1,
            'closing_total': 61.0,
            'status': 'QUALIFIES',
            'model_side': 'under',
            'model_track': 'GENERAL HGB',
        },
        # This game has no snapshot at least two hours before kickoff and must
        # therefore remain outside the official prospective sample.
        {
            'game_id': 2,
            'start_date': kickoff,
            'snapshot_timestamp_utc': '2026-09-05T15:00:00Z',
            'official_eligible': True,
            'github_run_id': 'late',
            'github_run_attempt': 1,
            'closing_total': 52.0,
            'status': 'NO PLAY',
            'model_side': 'under',
            'model_track': 'GENERAL HGB',
        },
    ])

    official = select_official_entries(snapshots, protocol)
    assert len(official) == 1
    assert int(official.iloc[0]['game_id']) == 1
    assert float(official.iloc[0]['closing_total']) == 59.0
    assert str(official.iloc[0]['github_run_id']) == 'saturday'
    assert int(official.iloc[0]['github_run_attempt']) == 1
    assert abs(float(official.iloc[0]['entry_lead_minutes']) - 180.0) < 0.01

    closes = pd.DataFrame([
    {
        'record_kind': 'close_capture',
        'game_id': 1,
        'start_date': kickoff,
        'snapshot_timestamp_utc': '2026-09-05T14:45:00Z',
        'benchmark_close_total': 58.0,
        'github_event_name': 'schedule',
        'github_run_attempt': 1,
        'line_provider': 'Book A',
        'line_source': 'fixture',
        'line_provider_count': 2,
        '_immutable_file': 'old.csv',
        '_immutable_sha256': 'a' * 64,
    },
    {
        'record_kind': 'close_capture',
        'game_id': 1,
        'start_date': kickoff,
        'snapshot_timestamp_utc': '2026-09-05T15:30:00Z',
        'benchmark_close_total': 57.5,
        'github_event_name': 'schedule',
        'github_run_attempt': 1,
        'line_provider': 'Book B',
        'line_source': 'fixture',
        'line_provider_count': 3,
        '_immutable_file': 'latest.csv',
        '_immutable_sha256': 'b' * 64,
    },
    # Scheduled but outside the frozen 90-minute close window.
    {
        'record_kind': 'close_capture',
        'game_id': 1,
        'start_date': kickoff,
        'snapshot_timestamp_utc': '2026-09-05T14:00:00Z',
        'benchmark_close_total': 60.0,
        'github_event_name': 'schedule',
        'github_run_attempt': 1,
    },
    # A manual capture cannot replace the scheduled benchmark.
    {
        'record_kind': 'close_capture',
        'game_id': 1,
        'start_date': kickoff,
        'snapshot_timestamp_utc': '2026-09-05T15:40:00Z',
        'benchmark_close_total': 56.5,
        'github_event_name': 'workflow_dispatch',
        'github_run_attempt': 1,
    },
    # A rerun cannot opportunistically backfill a closer number.
    {
        'record_kind': 'close_capture',
        'game_id': 1,
        'start_date': kickoff,
        'snapshot_timestamp_utc': '2026-09-05T15:45:00Z',
        'benchmark_close_total': 56.0,
        'github_event_name': 'schedule',
        'github_run_attempt': 2,
    },
])
    selected_close = select_benchmark_closes(closes)
    assert len(selected_close) == 1
    assert float(selected_close.iloc[0]['benchmark_close_total']) == 57.5
    assert abs(float(selected_close.iloc[0]['close_capture_lead_minutes']) - 30.0) < 0.01

    finals = pd.DataFrame([{
        'game_id': 1,
        'final_home_points': 31,
        'final_away_points': 23,
        'actual_total_points': 54,
    }])
    graded = grade_official_entries(official, selected_close, finals, protocol)
    assert graded.iloc[0]['paper_result'] == 'win'
    assert abs(float(graded.iloc[0]['paper_units_1u']) - (100 / 110)) < 1e-8
    assert abs(float(graded.iloc[0]['clv_points']) - 1.5) < 1e-8
    assert bool(graded.iloc[0]['positive_clv'])

    summary = summarize_graded(graded)
    overall = summary[summary['scope'].eq('ALL')].iloc[0]
    assert int(overall['wins']) == 1
    assert int(overall['losses']) == 0
    assert float(overall['hit_rate_ex_pushes']) == 1.0
    assert abs(float(overall['average_clv_points']) - 1.5) < 1e-8

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

    weekly_workflow = (ROOT / '.github/workflows/weekly-cfb-weather.yml').read_text(encoding='utf-8')
    for cron in protocol['official_entry_policy']['eligible_crons']:
        assert str(cron) in weekly_workflow, f'Official protocol cron is not scheduled: {cron}'

    close_workflow = (ROOT / '.github/workflows/prospective-close-capture.yml').read_text(encoding='utf-8')
    assert 'workflow_dispatch' not in close_workflow, 'Close benchmark workflow must not allow manual captures.'
    assert 'github.run_attempt == 1' in close_workflow, 'Close benchmark workflow must exclude rerun backfill.'
    assert 'twice per hour' in close_workflow, 'Protocol 2026.2 close-capture reliability note is missing.'
    for cron in protocol['closing_benchmark_policy']['capture_crons']:
        assert str(cron) in close_workflow, f'Frozen close-capture cron is not scheduled: {cron}'

    grade_workflow = (ROOT / '.github/workflows/prospective-grade.yml').read_text(encoding='utf-8')
    assert 'workflow_dispatch' not in grade_workflow, 'Prospective grading workflow should remain schedule-only.'
    for cron in protocol['paper_grading']['postgame_grade_crons']:
        assert str(cron) in grade_workflow, f'Frozen postgame grading cron is not scheduled: {cron}'

    print('Prospective ledger protocol, entry selection, CLV, grading cadence, and immutability checks passed.')


if __name__ == '__main__':
    main()
