"""Research-only historical NCAAF totals snapshot request planner.

No API calls, credentials, charges, or writes outside --out. Estimates ten
The Odds API credits per US totals historical snapshot (November 2026 terms).
A query can cover multiple upcoming games; counts are upper bounds, not
evidence that a sportsbook offered a fresh quote for each game.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

CREDIT_COST = 10
HORIZONS = (48, 24, 6, 2)
BUCKET_MINUTES = (5, 15, 30, 60)


def utc(x: str) -> datetime:
    ts = datetime.fromisoformat(str(x).strip().replace('Z', '+00:00'))
    if ts.tzinfo is None:
        raise ValueError('kickoff must include timezone')
    return ts.astimezone(timezone.utc)


def floor_utc(ts: datetime, minutes: int) -> datetime:
    if minutes not in BUCKET_MINUTES:
        raise ValueError('Unsupported bucket')
    if ts.tzinfo is None:
        raise ValueError('UTC instant required')
    v = ts.astimezone(timezone.utc)
    return v.replace(minute=(v.minute // minutes)*minutes, second=0, microsecond=0)


def read_games(path: Path, *, fbs_only: bool = False) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('No games provided')
    required = {'game_id', 'season', 'start_date', 'home_team', 'away_team'}
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if fbs_only and not {'home_classification','away_classification'} <= rows[0].keys():
        raise ValueError('No classification columns for FBS filter')
    games = []
    seen = set()
    for r in rows:
        if int(r['season']) not in (2023, 2024, 2025):
            continue
        if fbs_only and not (r['home_classification'].casefold()=='fbs' and r['away_classification'].casefold()=='fbs'):
            continue
        game_id = r['game_id'].strip()
        if not game_id or game_id in seen:
            raise ValueError('Missing or duplicate game ID')
        seen.add(game_id)
        kickoff = utc(r['start_date'])
        games.append({'game_id':game_id, 'season':int(r['season']), 'kickoff':kickoff,
                      'home_team':r['home_team'], 'away_team':r['away_team']})
    return sorted(games, key=lambda r:(r['kickoff'],r['game_id']))


def make_plan(games: Iterable[dict], horizons: Iterable[int], bucket_minutes: int) -> tuple[list[dict],list[dict]]:
    groups = defaultdict(list)
    assignments = []
    for game in games:
        for h in horizons:
            decision = game['kickoff']-timedelta(hours=h)
            snapshot = floor_utc(decision, bucket_minutes)
            age_min = (decision-snapshot).total_seconds()/60
            if not (0 <= age_min < bucket_minutes):
                raise AssertionError('snapshot must be earlier than cutoff')
            record = {
                'game_id': game['game_id'], 'season': game['season'],
                'away_team':game['away_team'], 'home_team':game['home_team'],
                'kickoff_utc': game['kickoff'].isoformat(),
                'horizon_hours':h, 'decision_time_utc':decision.isoformat(),
                'query_snapshot_utc':snapshot.isoformat(),
                'additional_snapshot_age_minutes':int(age_min),
                'market':'totals', 'region':'us',
                'executable_quote_verified': 'false',
            }
            assignments.append(record)
            groups[snapshot].append(record)
    requests=[]
    for snapshot,rows in sorted(groups.items()):
        requests.append({
            'query_snapshot_utc':snapshot.isoformat(),
            'sport_key':'americanfootball_ncaaf',
            'region':'us', 'market':'totals',
            'credit_budget_upper_bound': CREDIT_COST,
            'game_horizon_joins':len(rows),
            'distinct_games':len(set(r['game_id'] for r in rows)),
            'horizons':','.join(str(x) for x in sorted({r['horizon_hours'] for r in rows})),
            'max_additional_age_minutes':max(r['additional_snapshot_age_minutes'] for r in rows),
        })
    return assignments,requests


def count_by_season(games: list[dict], horizons: list[int],bucket: int) -> dict:
    assignments,requests=make_plan(games,horizons,bucket)
    credits = len(requests)*CREDIT_COST
    return {'games':len(games),'horizons':horizons,'bucket_minutes':bucket,
            'game_horizon_joins':len(assignments),'unique_api_snapshot_requests':len(requests),
            'credits_upper_bound':credits,
            'minimum_published_monthly_plan_usd':(30 if credits <=20000 else 59 if credits <=100000 else 119),
            'note':'Pricing assumes totals/us only; bookstore coverage not confirmed; no retries.'}


def write_csv(path:Path, rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        raise ValueError('No rows to export')
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--fbs-only',action='store_true')
    p.add_argument('--horizons',default='24,6,2')
    p.add_argument('--bucket',type=int,default=5,choices=BUCKET_MINUTES)
    args=p.parse_args()
    hs=[int(x) for x in args.horizons.split(',')]
    if not hs or len(set(hs))!=len(hs) or any(h not in HORIZONS for h in hs):
        raise ValueError('Use unique horizons from 48,24,6,2')
    games=read_games(args.input,fbs_only=args.fbs_only)
    assignments,requests=make_plan(games,hs,args.bucket)
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/'snapshot_request_plan.csv',requests)
    write_csv(args.out/'game_horizon_request_map.csv',assignments)
    summary=count_by_season(games,hs,args.bucket)
    summary.update({'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    'source_input':str(args.input),
                    'assumptions':['No retrospective sportsbook quote access',
                                   'Snapshot returned at/before requested query time',
                                   'Each region=us, markets=totals query costs at most ten credits',
                                   'Do not infer quote age from API snapshot time',
                                   'FBS classification filter is for request budgeting only',
                                   'One archived snapshot can cover many games; response coverage untested']})
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
