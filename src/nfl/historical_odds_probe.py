from __future__ import annotations

import json
import os
from pathlib import Path

import requests

URL = 'https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds'
OUT = Path('outputs/nfl/historical_odds_access_probe.json')


def main() -> None:
    key = os.getenv('ODDS_API_KEY', '').strip()
    summary = {
        'historical_endpoint': URL,
        'test_snapshot': '2025-09-07T16:00:00Z',
        'market': 'totals',
        'region': 'us',
        'key_present': bool(key),
    }

    if not key:
        summary.update({
            'status': 'missing_key',
            'historical_access': False,
        })
    else:
        response = requests.get(
            URL,
            params={
                'apiKey': key,
                'regions': 'us',
                'markets': 'totals',
                'oddsFormat': 'american',
                'dateFormat': 'iso',
                'date': summary['test_snapshot'],
            },
            timeout=60,
        )
        summary['http_status'] = response.status_code
        summary['requests_remaining'] = response.headers.get('x-requests-remaining')
        summary['requests_used'] = response.headers.get('x-requests-used')
        summary['requests_last'] = response.headers.get('x-requests-last')

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.status_code == 200 and isinstance(payload, dict):
            events = payload.get('data') or []
            summary.update({
                'status': 'ok',
                'historical_access': True,
                'event_count': len(events),
                'snapshot_timestamp': payload.get('timestamp'),
                'previous_timestamp': payload.get('previous_timestamp'),
                'next_timestamp': payload.get('next_timestamp'),
            })
            if events:
                event = events[0]
                summary['sample_event'] = {
                    'home_team': event.get('home_team'),
                    'away_team': event.get('away_team'),
                    'commence_time': event.get('commence_time'),
                    'bookmaker_count': len(event.get('bookmakers') or []),
                }
        else:
            summary.update({
                'status': 'unavailable',
                'historical_access': False,
                'error': payload if payload is not None else response.text[:500],
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
