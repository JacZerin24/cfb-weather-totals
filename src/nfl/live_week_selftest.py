from __future__ import annotations

import pandas as pd

from .live_week import (
    _classify,
    _decision_roof,
    _select_total_market,
)


def _protocol() -> dict:
    return {
        'qualifier': {
            'minimum_regression_edge_points': 3.0,
            'minimum_over_probability': 0.60,
        },
        'strong': {
            'minimum_regression_edge_points': 4.0,
            'minimum_over_probability': 0.60,
        },
    }


def test_roof_policy() -> None:
    retractable = pd.Series({
        'stadium_id': 'ATL97',
        'roof': 'outdoors',
        'roof_type': 'Outdoors',
    })
    assert _decision_roof(retractable) == 'retractable'

    outdoor = pd.Series({
        'stadium_id': 'BAL00',
        'roof': 'outdoors',
        'roof_type': 'Outdoors',
    })
    assert _decision_roof(outdoor) == 'outdoors'

    dome = pd.Series({
        'stadium_id': 'MIN00',
        'roof': 'dome',
        'roof_type': 'Dome',
    })
    assert _decision_roof(dome) == 'dome'


def test_consensus_market() -> None:
    event = {
        'bookmakers': [
            {
                'title': 'Book A',
                'markets': [{
                    'key': 'totals',
                    'last_update': '2026-09-19T12:00:00Z',
                    'outcomes': [
                        {'name': 'Over', 'point': 45.5, 'price': -110},
                        {'name': 'Under', 'point': 45.5, 'price': -110},
                    ],
                }],
            },
            {
                'title': 'Book B',
                'markets': [{
                    'key': 'totals',
                    'last_update': '2026-09-19T12:01:00Z',
                    'outcomes': [
                        {'name': 'Over', 'point': 45.5, 'price': -105},
                        {'name': 'Under', 'point': 45.5, 'price': -115},
                    ],
                }],
            },
            {
                'title': 'Book C',
                'markets': [{
                    'key': 'totals',
                    'last_update': '2026-09-19T12:02:00Z',
                    'outcomes': [
                        {'name': 'Over', 'point': 46.0, 'price': -108},
                        {'name': 'Under', 'point': 46.0, 'price': -112},
                    ],
                }],
            },
        ],
    }
    selected = _select_total_market(event)
    assert selected is not None
    assert selected['closing_total'] == 45.5
    assert selected['sportsbooks_at_selected_line'] == 2
    assert selected['sportsbooks_in_snapshot'] == 3
    assert selected['best_over_price_same_line'] == -105.0


def test_signal_policy() -> None:
    now = pd.Timestamp('2026-09-19T18:00:00Z')
    kickoff = now + pd.Timedelta(hours=24)
    board = pd.DataFrame([
        {
            'game_id': 'q',
            'kickoff_utc': kickoff,
            'closing_total': 45.5,
            'forecast_complete_24h': True,
            'pred_market_residual': 3.2,
            'over_probability': 0.61,
        },
        {
            'game_id': 's',
            'kickoff_utc': kickoff,
            'closing_total': 47.0,
            'forecast_complete_24h': True,
            'pred_market_residual': 4.4,
            'over_probability': 0.64,
        },
        {
            'game_id': 'n',
            'kickoff_utc': kickoff,
            'closing_total': 44.0,
            'forecast_complete_24h': True,
            'pred_market_residual': 2.9,
            'over_probability': 0.70,
        },
    ])
    out = _classify(board, _protocol(), now)
    status = dict(zip(out['game_id'], out['status']))
    assert status['q'] == 'QUALIFIES'
    assert status['s'] == 'STRONG'
    assert status['n'] == 'NO PLAY'
    assert out['decision_due'].all()


def main() -> None:
    test_roof_policy()
    test_consensus_market()
    test_signal_policy()
    print('NFL live-week self-test passed.')


if __name__ == '__main__':
    main()
