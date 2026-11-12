from __future__ import annotations

import unittest

import pandas as pd

from research.phase1.decision_time_audit import audit, summarize


class DecisionTimeAuditTests(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame([dict(game_id=100, season=2024,
            start_date='2024-10-06T00:00:00Z', actual_total_points=43)])
        self.odds = pd.DataFrame([dict(game_id=100, quote_id='odds-v1',
            captured_at_utc='2024-10-04T22:00:00Z',
            quoted_kickoff_utc='2024-10-06T00:00:00Z', bookmaker='book-A',
            total=60, under_odds_american=-110)])
        self.weather = pd.DataFrame([dict(game_id=100, weather_type='forecast',
            issued_at_utc='2024-10-04T19:00:00Z',
            available_at_utc='2024-10-04T20:00:00Z',
            valid_at_utc='2024-10-06T00:00:00Z', vintage_id='forecast-v1',
            wind_mph=17, temperature_f=56)])
        self.predictions = pd.DataFrame([dict(game_id=100,
            generated_at_utc='2024-10-04T23:00:00Z',
            training_cutoff_utc='2024-09-01T00:00:00Z',
            quote_id='odds-v1',weather_vintage_id='forecast-v1',
            pred_market_residual=-5)])

    def single(self, odds=None, weather=None, predictions=None):
        return audit(self.games,
            self.odds if odds is None else odds,
            self.weather if weather is None else weather,
            self.predictions if predictions is None else predictions,
            lead_hours=(24,)).iloc[0]

    def test_complete_replay_graded_at_real_price(self):
        row = self.single()
        self.assertTrue(row['eligible'])
        self.assertEqual(row['result'], 'win')
        self.assertAlmostEqual(row['profit_units'], 100/110)

    def test_future_quote_rejected(self):
        o = self.odds.copy()
        o.loc[0, 'captured_at_utc'] = '2024-10-05T01:00:00Z'
        row = self.single(odds=o)
        self.assertFalse(row['eligible'])
        self.assertIn('NO_PREGAME_ODDS_SNAPSHOT', row['exclusion_reason'])

    def test_observation_not_forecast(self):
        w = self.weather.copy()
        w.loc[0, 'weather_type'] = 'observation'
        row = self.single(weather=w)
        self.assertFalse(row['forecast_ok'])
        self.assertIn('NO_PREGAME_FORECAST_VINTAGE', row['exclusion_reason'])

    def test_mismatched_prediction_provenance(self):
        p = self.predictions.copy()
        p.loc[0, 'weather_vintage_id'] = 'some-other-run'
        row = self.single(predictions=p)
        self.assertFalse(row['prediction_ok'])
        self.assertIn('PREDICTION_PROVENANCE_INVALID', row['exclusion_reason'])

    def test_no_prediction_does_not_publish_zero_roi(self):
        empty = self.predictions.iloc[:0].copy()
        result = audit(self.games, self.odds, self.weather, empty, lead_hours=(24,))
        _, report = summarize(result)
        self.assertIsNone(report['roi_per_stake'])
        self.assertEqual(report['status'], 'BLOCKED_NO_VERIFIABLE_BETS')


if __name__ == '__main__':
    unittest.main()
