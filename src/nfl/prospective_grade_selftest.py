from __future__ import annotations

import math

import pandas as pd

from ..utils import load_yaml
from .prospective_grade import (
    _grade_decisions,
    _grade_entries,
    _signal_stability,
    _status,
    _summary,
)


def _fixtures():
    decisions = pd.DataFrame([
        {
            'game_id': 'g1',
            'week': 3,
            'away_team': 'A',
            'home_team': 'B',
            'snapshot_timestamp_utc': '2026-09-20T16:00:00Z',
            'kickoff_utc': '2026-09-21T16:00:00Z',
            'closing_total': 45.5,
            'over_probability': 0.64,
            'model_signal': 'QUALIFIES',
        },
        {
            'game_id': 'g2',
            'week': 3,
            'away_team': 'C',
            'home_team': 'D',
            'snapshot_timestamp_utc': '2026-09-20T17:00:00Z',
            'kickoff_utc': '2026-09-21T17:00:00Z',
            'closing_total': 44.0,
            'over_probability': 0.62,
            'model_signal': 'QUALIFIES',
        },
        {
            'game_id': 'g3',
            'week': 3,
            'away_team': 'E',
            'home_team': 'F',
            'snapshot_timestamp_utc': '2026-09-20T18:00:00Z',
            'kickoff_utc': '2026-09-21T18:00:00Z',
            'closing_total': 43.0,
            'over_probability': 0.61,
            'model_signal': 'STRONG',
        },
        {
            'game_id': 'g4',
            'week': 3,
            'away_team': 'G',
            'home_team': 'H',
            'snapshot_timestamp_utc': '2026-09-20T19:00:00Z',
            'kickoff_utc': '2026-09-21T19:00:00Z',
            'closing_total': 46.5,
            'over_probability': 0.56,
            'model_signal': 'NO PLAY',
        },
    ])

    entries = decisions.iloc[:3].copy()
    entries['entry_total'] = [45.5, 44.0, 43.0]
    entries['entry_over_price'] = [-110, 120, -105]
    entries['entry_tier'] = ['QUALIFIES', 'QUALIFIES', 'STRONG']

    schedule = pd.DataFrame([
        {
            'game_id': 'g1',
            'week': 3,
            'actual_total_points': 48.0,
            'gradeable_final': True,
        },
        {
            'game_id': 'g2',
            'week': 3,
            'actual_total_points': 41.0,
            'gradeable_final': True,
        },
        {
            'game_id': 'g3',
            'week': 3,
            'actual_total_points': 43.0,
            'gradeable_final': True,
        },
        {
            'game_id': 'g4',
            'week': 3,
            'actual_total_points': 50.0,
            'gradeable_final': True,
        },
    ])

    monitors = pd.DataFrame([
        {
            'record_kind': 'monitor_snapshot',
            'official_eligible': True,
            'github_run_attempt': 1,
            'game_id': 'g1',
            'snapshot_timestamp_utc': '2026-09-21T10:00:00Z',
            'model_signal': 'NO PLAY',
        },
        {
            'record_kind': 'monitor_snapshot',
            'official_eligible': True,
            'github_run_attempt': 1,
            'game_id': 'g2',
            'snapshot_timestamp_utc': '2026-09-21T11:00:00Z',
            'model_signal': 'STRONG',
        },
        {
            'record_kind': 'monitor_snapshot',
            'official_eligible': True,
            'github_run_attempt': 1,
            'game_id': 'g3',
            'snapshot_timestamp_utc': '2026-09-21T12:00:00Z',
            'model_signal': 'QUALIFIES',
        },
        {
            'record_kind': 'monitor_snapshot',
            'official_eligible': True,
            'github_run_attempt': 1,
            'game_id': 'g4',
            'snapshot_timestamp_utc': '2026-09-21T13:00:00Z',
            'model_signal': 'QUALIFIES',
        },
    ])
    return decisions, entries, schedule, monitors


def test_grading_and_units() -> None:
    decisions, entries, schedule, _ = _fixtures()
    graded_decisions = _grade_decisions(decisions, schedule)
    assert list(graded_decisions['decision_market_result']) == [
        'OVER', 'UNDER', 'PUSH', 'OVER'
    ]

    graded_entries = _grade_entries(entries, schedule)
    assert list(graded_entries['entry_result']) == [
        'WIN', 'LOSS', 'PUSH'
    ]
    assert abs(float(graded_entries.iloc[0]['paper_units']) - (100 / 110)) < 1e-9
    assert float(graded_entries.iloc[1]['paper_units']) == -1.0
    assert float(graded_entries.iloc[2]['paper_units']) == 0.0


def test_signal_change_labels() -> None:
    decisions, _, _, monitors = _fixtures()
    stability = _signal_stability(decisions, monitors)
    by_game = stability.set_index('game_id')

    assert bool(by_game.loc['g1', 'official_entry_faded'])
    assert bool(by_game.loc['g2', 'tier_upgraded_later'])
    assert bool(by_game.loc['g3', 'tier_downgraded_later'])
    assert bool(by_game.loc['g4', 'late_qualifier_after_freeze'])


def test_small_sample_never_promotes() -> None:
    decisions, entries, schedule, monitors = _fixtures()
    graded_decisions = _grade_decisions(decisions, schedule)
    graded_entries = _grade_entries(entries, schedule)
    stability = _signal_stability(decisions, monitors)

    with_clv = entries.copy()
    with_clv['clv_points'] = [1.0, 0.5, 1.5]
    metrics = _summary(
        graded_decisions,
        graded_entries,
        with_clv,
        stability,
    )
    config = load_yaml('config/nfl_prospective_evaluation.yml')
    status, checks = _status(metrics, config)
    assert status == 'INSUFFICIENT_SAMPLE'
    assert not checks['sample_graded_entries']
    assert metrics['graded_entries'] == 3
    assert metrics['graded_official_decisions'] == 4
    assert math.isfinite(float(metrics['brier_score']))


def test_review_status_requires_all_preregistered_checks() -> None:
    config = load_yaml('config/nfl_prospective_evaluation.yml')
    metrics = {
        'graded_entries': 30,
        'graded_official_decisions': 150,
        'entries_with_clv': 25,
        'distinct_entry_weeks': 10,
        'monitor_coverage_fraction': 0.95,
        'captured_price_roi': 0.08,
        'mean_clv_points': 0.7,
        'median_clv_points': 0.5,
        'positive_clv_fraction': 0.60,
        'brier_score': 0.235,
        'empirical_base_rate_brier': 0.245,
    }
    status, _ = _status(metrics, config)
    assert status == 'REVIEW_ELIGIBLE'

    metrics['mean_clv_points'] = -0.1
    status, checks = _status(metrics, config)
    assert status == 'DO_NOT_PROMOTE'
    assert not checks['performance_positive_mean_clv']


def main() -> None:
    test_grading_and_units()
    test_signal_change_labels()
    test_small_sample_never_promotes()
    test_review_status_requires_all_preregistered_checks()
    print(
        'NFL prospective evaluation self-test passed: settlement, pushes, '
        'calibration, signal drift, sample gate, and review gate.'
    )


if __name__ == '__main__':
    main()
