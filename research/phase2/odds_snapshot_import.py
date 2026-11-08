"""Parse supplied The Odds API historical NCAAF snapshot JSON, without API calls.

Strictly offline. Never infers UNDER price, market quote time, match ID or
kickoff. A quoted bet is eligible only when its *snapshot* was visible at the
specified decision time and event mapping is explicit and kickoff-compatible.
Do not mistake this parser for access to paid historical data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

UTC = timezone.utc

def parse_time(x: str) -> datetime:
    ts=datetime.fromisoformat(str(x).replace('Z','+00:00'))
    if ts.tzinfo is None:
        raise ValueError('Missing timezone on odds timestamp')
    return ts.astimezone(UTC)


def normalize_historical_snapshot(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all bookmaker UNDER quotes with conservative available-at timestamp."""
    snapshot=parse_time(data['timestamp'])
    events=data['data']
    if not isinstance(events,list):
        raise ValueError('Expected a historical snapshot data list')
    rows=[]
    for event in events:
        if not str(event.get('sport_key','')).startswith('americanfootball_ncaaf'):
            continue
        event_id=event.get('id')
        kickoff=parse_time(event['commence_time'])
        for book in event.get('bookmakers',[]):
            book_key=book.get('key')
            if not event_id or not book_key:
                continue
            for market in book.get('markets',[]):
                if market.get('key')!='totals':
                    continue
                market_last_update=market.get('last_update') or book.get('last_update')
                if not market_last_update:
                    continue
                quoted=parse_time(market_last_update)
                # A book update timestamp AFTER the returned historical snapshot
                # indicates inconsistent source provenance; do not interpolate.
                if quoted>snapshot:
                    continue
                outcomes=[x for x in market.get('outcomes',[]) if x.get('name')=='Under']
                if len(outcomes)!=1:
                    continue
                item=outcomes[0]
                try:
                    point=float(item['point']);price=float(item['price'])
                except (TypeError,ValueError,KeyError):
                    continue
                if not 15<=point<=110 or price==0 or not (-10000<=price<=10000):
                    continue
                rows.append({
                    'event_id':str(event_id),'sport_key':event['sport_key'],
                    'home_team':event['home_team'],'away_team':event['away_team'],
                    'kickoff_utc':kickoff.isoformat(),'bookmaker':str(book_key),
                    'market':'totals','selection':'Under','total':point,
                    'under_american_price':price,'quoted_utc':quoted.isoformat(),
                    'snapshot_utc':snapshot.isoformat(),
                    'available_at_utc':snapshot.isoformat(),
                    'quote_id':f"oddsapi:{event_id}:{book_key}:{quoted.isoformat()}:{point}:{price}",
                })
    return rows


def eligible_quotes(odds_rows: list[dict[str, Any]], mapping_rows: list[dict[str, Any]],
                    decision_times: list[dict[str, Any]], *, max_quote_age_hours: float=12,
                    kickoff_tolerance_minutes: float=45) -> list[dict[str, Any]]:
    """Match by explicit event-to-CFBD mapping, not fuzzy guesses."""
    mapping={x['event_id']:x for x in mapping_rows}
    results=[]
    for row in odds_rows:
        m=mapping.get(row['event_id'])
        if m is None:
            continue
        # Explicit mapping still requires exact team identity from prepared source
        # (or a separately human-reviewed alias audit) and kickoff tolerance.
        if (str(m['home_team']).casefold()!=str(row['home_team']).casefold() or
            str(m['away_team']).casefold()!=str(row['away_team']).casefold()):
            continue
        if abs((parse_time(m['kickoff_utc'])-parse_time(row['kickoff_utc'])).total_seconds())>kickoff_tolerance_minutes*60:
            continue
        for gate in decision_times:
            if str(gate['game_id'])!=str(m['game_id']):
                continue
            asof=parse_time(gate['decision_time_utc'])
            available=parse_time(row['available_at_utc'])
            if not (available <= asof < parse_time(row['kickoff_utc'])):
                continue
            age=(asof-available).total_seconds()/3600
            if age>max_quote_age_hours:
                continue
            results.append({**row,'game_id':m['game_id'],
                            'decision_time_utc':asof.isoformat(),
                            'horizon_hours':gate['horizon_hours'],'age_hours':age})
    return results


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--snapshots',type=Path,required=True,help='Exported paid historical snapshot JSON (file or directory)')
    p.add_argument('--event-map',type=Path,required=True,help='Explicitly verified mapping CSV: event_id,game_id,home_team,away_team,kickoff_utc')
    p.add_argument('--schedule',type=Path,required=True,help='Pilot schedule CSV, including decision_time_utc and horizon_hours')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    files=sorted(a.snapshots.glob('*.json')) if a.snapshots.is_dir() else [a.snapshots]
    rows=[]
    for f in files: rows+=normalize_historical_snapshot(json.loads(f.read_text()))
    with a.event_map.open() as f: maps=list(csv.DictReader(f))
    with a.schedule.open() as f: schedule=list(csv.DictReader(f))
    matched=eligible_quotes(rows,maps,schedule)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    cols=['event_id','game_id','sport_key','home_team','away_team','kickoff_utc','bookmaker','market',
          'selection','total','under_american_price','quoted_utc','snapshot_utc','available_at_utc',
          'quote_id','decision_time_utc','horizon_hours','age_hours']
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(matched)
    print(json.dumps({'input_quotes':len(rows),'eligible_quote_horizons':len(matched),'output':str(a.out)}))

if __name__=='__main__':main()
