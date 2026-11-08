"""Offline tests for the historical odds acquisition budget planner."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from research.phase2.historical_odds_request_planner import floor_utc, make_plan, utc, count_by_season


class SnapshotPlannerTests(unittest.TestCase):
    def games(self):
        return [
            dict(game_id='1', season=2024, kickoff=utc('2024-09-21T19:30:00Z'), home_team='A', away_team='B'),
            dict(game_id='2', season=2024, kickoff=utc('2024-09-21T19:30:00Z'), home_team='C', away_team='D')
        ]

    def test_same_time_batches_games(self):
        assignments, requests = make_plan(self.games(), [24, 6, 2], 5)
        self.assertEqual(len(assignments), 6)
        self.assertEqual(len(requests), 3)
        self.assertEqual(sum(row['credit_budget_upper_bound'] for row in requests), 30)

    def test_stale_snapshot_never_after_cutoff(self):
        games = [dict(**{**self.games()[0], 'kickoff': utc('2024-09-21T19:47:00Z')})]
        assignments, requests = make_plan(games, [24, 6, 2], 60)
        self.assertTrue(all(utc(row['query_snapshot_utc']) <= utc(row['decision_time_utc']) for row in assignments))
        self.assertTrue(all(row['additional_snapshot_age_minutes'] == 47 for row in assignments))

    def test_reject_naive_time(self):
        with self.assertRaises(ValueError):
            utc('2024-09-21T19:30:00')

    def test_48h_odds_request_is_supported_even_when_hrrr_is_not(self):
        assignments, requests = make_plan(self.games(), [48], 5)
        self.assertEqual(len(requests), 1)
        self.assertEqual(assignments[0]['horizon_hours'], 48)

    def test_invalid_bucket(self):
        with self.assertRaises(ValueError):
            floor_utc(datetime.now(timezone.utc), 45)

    def test_credit_estimate_without_provider_access(self):
        self.assertEqual(count_by_season(self.games(), [24, 6, 2], 5)['credits_upper_bound'], 30)


if __name__ == '__main__':
    unittest.main()
