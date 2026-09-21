from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ..utils import ROOT
from .close_capture_gate import candidate_kickoffs, should_capture
from .prospective_ledger import (
    _csv_bytes,
    _immutable_write,
    select_benchmark_closes,
    select_official_decisions,
    select_official_entries,
)


def test_official_decision_is_first_snapshot() -> None:
    snapshots = pd.DataFrame([
        {
            'record_kind': 'decision_snapshot',
            'official_eligible': True,
            'github_run_attempt': 1,
            'github_run_id': '100',
            'season': 2026,
            'game_id': 'g1',
            'snapshot_timestamp_utc': '2026-09-20T16:00:00Z',
            'kickoff_utc': '2026-09-21T16:45:00Z',
            'decision_due': True,
            'decision_state': '24H_DECISION_WINDOW',
            'model_signal': 'QUALIFIES',
            'closing_total': 45.5,
            'consensus_over_price': -110,
            'consensus_over_sportsbook': 'Book A',
            'forecast_complete_24h': True,
            'pred_market_residual': 3.2,
            'over_probability': 0.61,
        },
        {
            'record_kind': 'decision_snapshot',
            'official_eligible': True,
            'github_run_attempt': 1,
            'github_run_id': '101',
            'season': 2026,
            'game_id': 'g1',
            'snapshot_timestamp_utc': '2026-09-20T17:30:00Z',
            'kickoff_utc': '2026-09-21T16:45:00Z',
            'decision_due': True,
            'decision_state': '24H_DECISION_WINDOW',
            'model_signal': 'STRONG',
            'closing_total': 44.5,
            'consensus_over_price': -108,
            'consensus_over_sportsbook': 'Book B',
            'forecast_complete_24h': True,
            'pred_market_residual': 4.5,
            'over_probability': 0.66,
        },
        {
            # g2 was NO PLAY at the official decision. A later qualifying
            # refresh must not manufacture an entry.
            'record_kind': 'decision_snapshot',
            'official_eligible': True,
            'github_run_attempt': 1,
            'github_run_id': '200',
            'season': 2026,
            'game_id': 'g2',
            'snapshot_timestamp_utc': '2026-09-20T18:00:00Z',
            'kickoff_utc': '2026-09-21T18:30:00Z',
            'decision_due': True,
            'decision_state': '24H_DECISION_WINDOW',
            'model_signal': 'NO PLAY',
            'closing_total': 47.0,
            'consensus_over_price': -110,
            'forecast_complete_24h': True,
            'pred_market_residual': 2.8,
            'over_probability': 0.59,
        },
        {
            'record_kind': 'decision_snapshot',
            'official_eligible': True,
            'github_run_attempt': 1,
            'github_run_id': '201',
            'season': 2026,
            'game_id': 'g2',
            'snapshot_timestamp_utc': '2026-09-20T19:00:00Z',
            'kickoff_utc': '2026-09-21T18:30:00Z',
            'decision_due': True,
            'decision_state': '24H_DECISION_WINDOW',
            'model_signal': 'QUALIFIES',
            'closing_total': 46.0,
            'consensus_over_price': -105,
            'forecast_complete_24h': True,
            'pred_market_residual': 3.4,
            'over_probability': 0.63,
        },
        {
            # Manual/non-scheduled snapshots are never official.
            'record_kind': 'decision_snapshot',
            'official_eligible': False,
            'github_run_attempt': 1,
            'github_run_id': 'manual',
            'season': 2026,
            'game_id': 'g3',
            'snapshot_timestamp_utc': '2026-09-20T20:00:00Z',
            'kickoff_utc': '2026-09-21T20:00:00Z',
            'decision_due': True,
            'decision_state': '24H_DECISION_WINDOW',
            'model_signal': 'STRONG',
            'closing_total': 48.0,
            'consensus_over_price': -110,
            'forecast_complete_24h': True,
        },
        {
            # Reruns cannot backfill.
            'record_kind': 'decision_snapshot',
            'official_eligible': True,
            'github_run_attempt': 2,
            'github_run_id': '300',
            'season': 2026,
            'game_id': 'g4',
            'snapshot_timestamp_utc': '2026-09-20T21:00:00Z',
            'kickoff_utc': '2026-09-21T21:00:00Z',
            'decision_due': True,
            'decision_state': '24H_DECISION_WINDOW',
            'model_signal': 'STRONG',
            'closing_total': 49.0,
            'consensus_over_price': -110,
            'forecast_complete_24h': True,
        },
    ])

    decisions = select_official_decisions(snapshots)
    assert list(decisions['game_id']) == ['g1', 'g2']

    g1 = decisions[decisions['game_id'].eq('g1')].iloc[0]
    assert g1['model_signal'] == 'QUALIFIES'
    assert float(g1['closing_total']) == 45.5

    g2 = decisions[decisions['game_id'].eq('g2')].iloc[0]
    assert g2['model_signal'] == 'NO PLAY'

    entries = select_official_entries(decisions)
    assert list(entries['game_id']) == ['g1']
    assert entries.iloc[0]['official_side'] == 'OVER'
    assert float(entries.iloc[0]['entry_total']) == 45.5
    assert float(entries.iloc[0]['entry_over_price']) == -110


def test_latest_valid_close_and_clv_direction() -> None:
    closes = pd.DataFrame([
        {
            'record_kind': 'close_capture',
            'github_event_name': 'schedule',
            'github_run_attempt': 1,
            'game_id': 'g1',
            'kickoff_utc': '2026-09-21T16:45:00Z',
            'snapshot_timestamp_utc': '2026-09-21T15:15:00Z',
            'benchmark_close_total': 46.0,
            'benchmark_close_over_price': -110,
            'benchmark_close_over_sportsbook': 'Book A',
        },
        {
            'record_kind': 'close_capture',
            'github_event_name': 'schedule',
            'github_run_attempt': 1,
            'game_id': 'g1',
            'kickoff_utc': '2026-09-21T16:45:00Z',
            'snapshot_timestamp_utc': '2026-09-21T16:25:00Z',
            'benchmark_close_total': 47.0,
            'benchmark_close_over_price': -108,
            'benchmark_close_over_sportsbook': 'Book B',
        },
        {
            # A rerun closer to kickoff must not replace attempt 1.
            'record_kind': 'close_capture',
            'github_event_name': 'schedule',
            'github_run_attempt': 2,
            'game_id': 'g1',
            'kickoff_utc': '2026-09-21T16:45:00Z',
            'snapshot_timestamp_utc': '2026-09-21T16:40:00Z',
            'benchmark_close_total': 48.0,
            'benchmark_close_over_price': -105,
        },
        {
            # Manual capture is never a benchmark.
            'record_kind': 'close_capture',
            'github_event_name': 'workflow_dispatch',
            'github_run_attempt': 1,
            'game_id': 'g1',
            'kickoff_utc': '2026-09-21T16:45:00Z',
            'snapshot_timestamp_utc': '2026-09-21T16:42:00Z',
            'benchmark_close_total': 48.5,
            'benchmark_close_over_price': -105,
        },
    ])
    selected = select_benchmark_closes(closes)
    assert len(selected) == 1
    assert float(selected.iloc[0]['benchmark_close_total']) == 47.0
    assert abs(float(selected.iloc[0]['close_capture_lead_minutes']) - 20.0) < 0.01

    # OVER 45.5 closing 47.0 is +1.5 points of CLV.
    clv = float(selected.iloc[0]['benchmark_close_total']) - 45.5
    assert clv == 1.5


def test_immutable_writer_rejects_overwrite() -> None:
    fixture = pd.DataFrame([{
        'github_run_id': 'fixture',
        'github_run_attempt': 1,
        'value': 42,
    }])
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        now = pd.Timestamp('2026-09-20T16:00:00Z')
        path = _immutable_write(fixture, directory, 'decision', now)
        digest = hashlib.sha256(_csv_bytes(fixture)).hexdigest()
        assert digest[:12] in path.name

        try:
            _immutable_write(fixture, directory, 'decision', now)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Immutable NFL writer allowed overwrite.')


def test_zero_api_close_gate_uses_only_entries() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        entries = root / 'entries.csv'
        board = root / 'board.csv'

        pd.DataFrame([{
            'game_id': 'g1',
            'kickoff_utc': '2026-09-21T16:45:00Z',
            'away_team': 'Away',
            'home_team': 'Home',
        }]).to_csv(entries, index=False)

        # Current board updates the kickoff to 17:00Z.
        pd.DataFrame([{
            'game_id': 'g1',
            'kickoff_utc': '2026-09-21T17:00:00Z',
            'away_team': 'Away',
            'home_team': 'Home',
        }, {
            # Non-entry game must not open the gate.
            'game_id': 'g2',
            'kickoff_utc': '2026-09-21T16:00:00Z',
            'away_team': 'Other',
            'home_team': 'Other Home',
        }]).to_csv(board, index=False)

        assert not should_capture(
            '2026-09-21T14:00:00Z',
            105,
            entries,
            board,
        )
        candidates = candidate_kickoffs(
            '2026-09-21T15:30:00Z',
            105,
            entries,
            board,
        )
        assert len(candidates) == 1
        assert str(candidates.iloc[0]['game_id']) == 'g1'
        assert should_capture(
            '2026-09-21T15:30:00Z',
            105,
            entries,
            board,
        )


def test_workflow_guards() -> None:
    weekly = (
        ROOT / '.github/workflows/weekly-nfl-weather.yml'
    ).read_text(encoding='utf-8')
    assert 'python -m src.nfl.prospective_ledger snapshot-board' in weekly
    assert 'NFL_PROSPECTIVE_RUN_ATTEMPT' in weekly
    assert 'outputs/nfl/prospective/2026/' in weekly

    close = (
        ROOT / '.github/workflows/nfl-prospective-close-capture.yml'
    ).read_text(encoding='utf-8')
    assert 'workflow_dispatch' not in close
    assert 'github.run_attempt == 1' in close
    assert (
        'python -m src.nfl.close_capture_gate --window-minutes 105'
        in close
    )
    assert 'python -m src.nfl.prospective_ledger capture-close' in close
    assert '15,45 12-23 * 9,10,11,12,1 *' in close


def main() -> None:
    test_official_decision_is_first_snapshot()
    test_latest_valid_close_and_clv_direction()
    test_immutable_writer_rejects_overwrite()
    test_zero_api_close_gate_uses_only_entries()
    test_workflow_guards()
    print(
        'NFL prospective ledger self-test passed: first-decision freeze, '
        'no-play preservation, immutable hashes, zero-API close gate, '
        'first-attempt-only benchmarks, and OVER CLV direction.'
    )


if __name__ == '__main__':
    main()
