import unittest
from datetime import datetime,timezone,timedelta
from research.phase2.hrrr_archive_pilot import choose_run,parse_idx,fields_from_idx
from research.phase2.odds_snapshot_import import normalize_historical_snapshot,eligible_quotes

class ArchivePilotTests(unittest.TestCase):
    def test_48h_hrrr_is_unavailable_given_two_hour_lag(self):
        self.assertIsNone(choose_run(datetime(2024,9,21,19,30,tzinfo=timezone.utc),48))

    def test_24h_run_really_available_and_valid_near_kickoff(self):
        kick=datetime(2024,9,21,19,30,tzinfo=timezone.utc)
        r=choose_run(kick,24)
        self.assertIsNotNone(r)
        self.assertLessEqual(r['assumed_available_at_utc'],r['decision_time_utc'])
        self.assertLessEqual(abs(r['valid_offset_minutes']),30)
        self.assertIn(datetime.fromisoformat(r['run_time_utc']).hour,[0,6,12,18])

    def test_6h_and_2h_validity_and_availability(self):
        kick=datetime(2023,9,23,23,tzinfo=timezone.utc)
        for horizon in [6,2]:
            r=choose_run(kick,horizon)
            self.assertIsNotNone(r)
            self.assertLessEqual(r['assumed_available_at_utc'],r['decision_time_utc'])
            self.assertNotEqual(r['forecast_hour'],0)

    def test_hrrr_wgrib2_index_is_parsed_and_ambiguous_rejected(self):
        fields=['TMP:2 m above ground','DPT:2 m above ground',
                'UGRD:10 m above ground','VGRD:10 m above ground','GUST:surface']
        inp='\n'.join(f'{i+1}:{i*200}:d=2024092112:{name}:12 hour fcst:'
                      for i,name in enumerate(fields))+'\n6:1000:d=2024092112:MSLET:mean sea level:12 hour fcst:'
        rows=parse_idx(inp)
        self.assertEqual(rows[0]['end'],199)
        self.assertEqual(rows[4]['end'],999)
        self.assertEqual(len(fields_from_idx(rows)),5)
        with self.assertRaises(ValueError):
            fields_from_idx(rows+[{**rows[0],'start':1200}])

    def make_odds(self, *, year=2024):
        timestamp=f'{year}-09-21T18:05:00Z'
        kickoff=f'{year}-09-21T19:30:00Z'
        market={'key':'totals','outcomes':[{'name':'Under','price':-112,'point':56.5},
                                          {'name':'Over','price':-108,'point':56.5}]}
        book={'key':'draftkings','last_update':f'{year}-09-21T18:04:00Z','markets':[market]}
        event={'id':'event1','sport_key':'americanfootball_ncaaf','home_team':'LSU',
               'away_team':'UCLA','commence_time':kickoff,'bookmakers':[book]}
        return normalize_historical_snapshot({'timestamp':timestamp,'data':[event]})

    def test_odds_snapshot_is_not_avail_before_capture(self):
        odds=self.make_odds()
        self.assertEqual(len(odds),1)
        self.assertEqual(odds[0]['under_american_price'],-112)
        mapping=[{'event_id':'event1','game_id':'401628365','home_team':'LSU',
                  'away_team':'UCLA','kickoff_utc':'2024-09-21T19:30:00Z'}]
        early=[{'game_id':'401628365','decision_time_utc':'2024-09-21T18:00:00Z','horizon_hours':'fake'}]
        later=[{'game_id':'401628365','decision_time_utc':'2024-09-21T18:10:00Z','horizon_hours':'fake'}]
        self.assertEqual(len(eligible_quotes(odds,mapping,early)),0)
        self.assertEqual(len(eligible_quotes(odds,mapping,later)),1)

    def test_bad_teams_or_rescheduled_kickoff_block_match(self):
        odds=self.make_odds()
        gate=[{'game_id':'401628365','decision_time_utc':'2024-09-21T18:10:00Z','horizon_hours':'fake'}]
        mapping=[{'event_id':'event1','game_id':'401628365','home_team':'LSU',
                 'away_team':'Other','kickoff_utc':'2024-09-21T19:30:00Z'}]
        self.assertEqual(len(eligible_quotes(odds,mapping,gate)),0)
        mapping[0]['away_team']='UCLA'
        mapping[0]['kickoff_utc']='2024-09-22T01:30:00Z'
        self.assertEqual(len(eligible_quotes(odds,mapping,gate)),0)

if __name__=='__main__': unittest.main()
