from __future__ import annotations

import argparse
from pathlib import Path

from ..utils import ROOT
from .inject_sport_selector import START, strip_selector


NFL_INDEX = ROOT / 'docs/nfl/index.html'
CFB_INDEX = ROOT / 'docs/index.html'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfb-before', default='')
    args = parser.parse_args()

    if not NFL_INDEX.exists():
        raise AssertionError('docs/nfl/index.html is missing.')
    if not CFB_INDEX.exists():
        raise AssertionError('docs/index.html is missing.')

    nfl = NFL_INDEX.read_text(encoding='utf-8')
    cfb = CFB_INDEX.read_text(encoding='utf-8')

    required_nfl = {
        'sport switch': 'aria-label="Sport mode"',
        'paper mode': 'Paper mode · frozen protocol',
        'live cards': 'id="gameCards"',
        'status filter': 'id="statusFilter"',
        'signal filter': 'id="signalFilter"',
        'prospective ledger': 'Frozen 24-hour decisions',
        'prospective performance': 'Prospective performance',
        'evaluation status': 'Prospective evidence status',
        'captured ROI': 'Captured-price ROI',
        'qualifies rule': 'OVER edge ≥ +3.0',
        'strong rule': 'OVER edge ≥ +4.0',
        'back to NCAA': 'href="../"',
    }
    missing_nfl = [name for name, token in required_nfl.items() if token not in nfl]
    if missing_nfl:
        raise AssertionError(
            'NFL site missing required UI: ' + ', '.join(missing_nfl)
        )

    required_cfb = {
        'selector marker': START,
        'NFL link': 'href="nfl/"',
        'weather UI marker': 'SITE_WEATHER_TOOLS',
        'radar marker': 'SITE_RADAR_SYNC',
        'radar toggle': 'id="radarToggle"',
        'weather filter': 'id="weatherFilter"',
        'status filter': 'id="statusFilter"',
        'division filter': 'id="divisionFilter"',
    }
    missing_cfb = [name for name, token in required_cfb.items() if token not in cfb]
    if missing_cfb:
        raise AssertionError(
            'NCAA site missing required existing/shared UI: '
            + ', '.join(missing_cfb)
        )

    if cfb.count(START) != 1:
        raise AssertionError(
            f'NCAA selector marker count is {cfb.count(START)}, expected 1.'
        )

    if args.cfb_before:
        before_path = Path(args.cfb_before)
        before = before_path.read_text(encoding='utf-8')
        if strip_selector(cfb) != strip_selector(before):
            raise AssertionError(
                'NCAA page changed outside the isolated sport-selector block.'
            )

    print(
        'NFL site self-test passed: cards/filters/protocol present, '
        'NCAA production UI preserved, selector isolated.'
    )


if __name__ == '__main__':
    main()
