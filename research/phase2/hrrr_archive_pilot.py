"""Isolated NOAA HRRR forecast-vintage reconstruction for 2023--25 CFB games.

Neither imports nor writes any production module/file. No sportsbook data is inferred.
Data is a *retrospective reconstruction* from run vintages, not proof of actual
publication time at the sportsbook decision. Conservative publication-lag proxy
is exposed in every row. The weather is extracted at a stadium, not observed.

Dependencies for archive mode: requests, eccodes (with bundled ecCodes library).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

UTC = timezone.utc
BASE = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com'
MPH_PER_MPS = 2.2369362921
FIELD_LEVELS = {
    'temperature_k': ('TMP', '2 m above ground'),
    'dewpoint_k': ('DPT', '2 m above ground'),
    'u10_mps': ('UGRD', '10 m above ground'),
    'v10_mps': ('VGRD', '10 m above ground'),
    'gust_mps': ('GUST', 'surface'),
}
PILOT_HORIZONS = (48, 24, 6, 2)


def dt(value: Any) -> datetime:
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamp must include UTC offset')
    return result.astimezone(UTC)


def choose_run(kickoff: datetime, horizon_hours: int, *, publication_lag_hours: int = 2,
               max_valid_offset_minutes: int = 30) -> dict[str, Any] | None:
    """Latest nominally available HRRR run, not future forecast/analysis.

    48h mostly fails correctly: run needs a publication lag AND HRRR max f48.
    The selected valid time cannot differ from kickoff by >30 minutes.
    """
    if horizon_hours < 0 or publication_lag_hours < 0:
        raise ValueError('Nonnegative horizon and lag required')
    kickoff = dt(kickoff.isoformat())
    asof = kickoff - timedelta(hours=horizon_hours)
    cutoff = asof - timedelta(hours=publication_lag_hours)
    cursor = cutoff.replace(minute=0, second=0, microsecond=0)
    for _ in range(56):
        lead = (kickoff - cursor).total_seconds() / 3600
        fxx = round(lead)
        max_lead = 48 if cursor.hour in (0, 6, 12, 18) else 18
        valid = cursor + timedelta(hours=fxx)
        if 1 <= fxx <= max_lead and abs((valid - kickoff).total_seconds()) <= max_valid_offset_minutes * 60:
            # A 48-h candidate with 2-h lag is necessarily >48 forecast hours.
            if cursor + timedelta(hours=publication_lag_hours) <= asof:
                return {
                    'decision_time_utc': asof.isoformat(),
                    'run_time_utc': cursor.isoformat(),
                    'assumed_available_at_utc': (cursor + timedelta(hours=publication_lag_hours)).isoformat(),
                    'forecast_hour': fxx,
                    'valid_time_utc': valid.isoformat(),
                    'valid_offset_minutes': (valid - kickoff).total_seconds() / 60,
                    'publication_lag_assumption_hours': publication_lag_hours,
                }
        cursor -= timedelta(hours=1)
    return None


def parse_idx(txt: str) -> list[dict[str, Any]]:
    """Parse wgrib2 inventory, explicitly compute byte end from next record."""
    rows: list[dict[str, Any]] = []
    for line in txt.splitlines():
        fields = line.strip().split(':')
        if len(fields) < 6:
            continue
        try:
            num = int(fields[0]); start = int(fields[1])
        except ValueError:
            continue
        rows.append({'number':num, 'start':start, 'short':fields[3], 'level':fields[4],
                     'step':fields[5], 'raw_line':line})
    if not rows or rows[0]['start'] != 0:
        raise ValueError('Index missing first GRIB message at byte 0')
    for i, row in enumerate(rows):
        if i > 0 and row['start'] <= rows[i-1]['start']:
            raise ValueError('Index byte offsets are nonmonotonic')
        row['end'] = rows[i+1]['start'] - 1 if i+1 < len(rows) else None
    return rows


def fields_from_idx(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chosen = {}
    for name, (short, level) in FIELD_LEVELS.items():
        found = [r for r in rows if r['short'] == short and r['level'] == level]
        # Ambiguous GRIB records cannot be chosen silently.
        if len(found) != 1:
            raise ValueError(f'Expected exactly one {short}:{level} record; found {len(found)}')
        chosen[name] = found[0]
    return chosen


def grib_url(run: datetime, forecast_hour: int) -> str:
    return (f'{BASE}/hrrr.{run:%Y%m%d}/conus/'
            f'hrrr.t{run:%H}z.wrfsfcf{forecast_hour:02d}.grib2')


def _fetch(session, url: str, *, range_header: str | None = None, timeout: int = 65) -> bytes:
    headers = {'Range': f'bytes={range_header}'} if range_header else {}
    r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    if range_header:
        if r.status_code != 206 or not r.headers.get('Content-Range', '').startswith('bytes '):
            raise ValueError(f'Byte range not honored ({r.status_code}) at {url}')
        a, b = map(int, range_header.split('-'))
        if len(r.content) != b - a + 1:
            raise ValueError('Unexpected partial GRIB message byte length')
    return r.content


def point_from_message(data: bytes, *, lat: float, lon: float,
                       expected_cycle: datetime, expected_valid: datetime) -> tuple[float, float]:
    """ecCodes decodes one GRIB2 message into a spatial nearest-gridpoint value."""
    from eccodes import codes_new_from_message, codes_get, codes_grib_find_nearest, codes_release
    h = codes_new_from_message(data)
    try:
        cycle = datetime.strptime(f"{int(codes_get(h,'dataDate')):08d}{int(codes_get(h,'dataTime')):04d}",
                                  '%Y%m%d%H%M').replace(tzinfo=UTC)
        valid = datetime.strptime(f"{int(codes_get(h,'validityDate')):08d}{int(codes_get(h,'validityTime')):04d}",
                                  '%Y%m%d%H%M').replace(tzinfo=UTC)
        if cycle != expected_cycle or valid != expected_valid:
            raise ValueError(f'GRIB metadata mismatch cycle={cycle} valid={valid}')
        nearest = codes_grib_find_nearest(h, lat, lon % 360)[0]
        return float(nearest['value']), float(nearest['distance'])
    finally:
        codes_release(h)


def read_archive_forecast(session, case: dict, chosen: dict) -> dict:
    run = dt(chosen['run_time_utc'])
    valid = dt(chosen['valid_time_utc'])
    url = grib_url(run, chosen['forecast_hour'])
    idx = _fetch(session, url + '.idx')
    rows = parse_idx(idx.decode('utf-8'))
    records = fields_from_idx(rows)
    lat,lon = float(case['venue_latitude']),float(case['venue_longitude'])
    values = {}; distances = {}; checksums={}
    for field, rec in records.items():
        # If last in file, safe refusal rather than fetching an unbounded tail.
        if rec['end'] is None:
            raise ValueError(f'Cannot determine final byte boundary for {field}')
        data = _fetch(session, url, range_header=f"{rec['start']}-{rec['end']}")
        val, distance = point_from_message(data, lat=lat, lon=lon,
                                           expected_cycle=run, expected_valid=valid)
        values[field]=val; distances[field]=distance
        checksums[field]=hashlib.sha256(data).hexdigest()
    t = values['temperature_k']-273.15
    td = values['dewpoint_k']-273.15
    rh = 100*math.exp((17.625*td)/(243.04+td)-(17.625*t)/(243.04+t))
    return {
        **values,
        'temperature_f':t*9/5+32, 'dewpoint_f':td*9/5+32,
        'relative_humidity_pct':max(0,min(100,rh)),
        'wind_mph':math.hypot(values['u10_mps'],values['v10_mps'])*MPH_PER_MPS,
        'wind_direction_degrees':(math.degrees(math.atan2(-values['u10_mps'],-values['v10_mps'])) + 360) % 360,
        'gust_mph':values['gust_mps']*MPH_PER_MPS,
        'gridpoint_max_distance_km':max(distances.values()),
        'grib_url':url,'index_sha256':hashlib.sha256(idx).hexdigest(),
        'selected_messages_sha256_json':json.dumps(checksums,sort_keys=True),
        'archive_file_source':'NOAA AWS public HRRR wrfsfc',
        'archive_retrieval_utc':datetime.now(UTC).isoformat(),
        'forecast_availability_is_assumed_not_timestamp_verified':True,
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--games',default='research/phase2/sample_games_2023_2025.csv')
    parser.add_argument('--out',default='research/phase2/pilot_results')
    parser.add_argument('--per-season',type=int,default=1,help='number of predetermined games per season in supplied sample (1, 2, or 4)')
    parser.add_argument('--dry-run',action='store_true',help='Only create deterministic as-of schedule and availability flags')
    args=parser.parse_args()
    if args.per_season not in [1,2,4]:
        parser.error('Pilot sample must be 1, 2, or 4 games per season')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with open(args.games,newline='') as f: cases=list(csv.DictReader(f))
    cases=[g for i,year in enumerate(('2023','2024','2025'))
           for g in [x for x in cases if x['season']==year][:args.per_season]]
    if len(cases) != 3*args.per_season:
        raise ValueError('Pilot requires equally sized predetermined cases in 2023,2024,2025')
    import requests
    session=requests.Session()
    outputs=[]
    for case in cases:
        kickoff=dt(case['start_date'])
        for h in PILOT_HORIZONS:
            row={**case,'horizon_hours':h,'kickoff_utc':kickoff.isoformat(),
                 'status':'missing_archive_run_or_forecast_range','market_quote_verified':False,
                 'retrospective_weather_reconstruction':True}
            selected=choose_run(kickoff,h)
            if selected is not None:
                row.update(selected)
                row['status']='schedule_only' if args.dry_run else 'archive_fetch_pending'
                if not args.dry_run:
                    try:
                        row.update(read_archive_forecast(session,case,selected))
                        row['status']='forecast_reconstructed'
                    except Exception as e:
                        row['status']='forecast_fetch_failed'
                        row['reason']=f'{type(e).__name__}: {str(e)[:350]}'
            outputs.append(row)
            print(f"{case['season']} {case['away_team']} @ {case['home_team']}, {h}h: {row['status']}",flush=True)
    cols=list(dict.fromkeys(key for row in outputs for key in row))
    with (out/'forecast_vintage_pilot.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(outputs)
    counts={s:sum(r['status']==s for r in outputs) for s in sorted({r['status'] for r in outputs})}
    summary={'pilot_years':[2023,2024,2025],'games':len(cases),'horizons':list(PILOT_HORIZONS),
             'forecast_rows':len(outputs),'by_status':counts,
             'archive_weather_successes':counts.get('forecast_reconstructed',0),
             'genuine_timestamped_bookmaker_quotes':0,
             'execution_verified_plays':0,'execution_verified_roi':None,
             'note':'No sportsbook data or archived model predictions supplied. Forecast publication lags are assumptions, not measured issue availability. No betting ROI can be computed.'}
    (out/'pilot_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
