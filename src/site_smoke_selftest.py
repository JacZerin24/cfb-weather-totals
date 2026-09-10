from __future__ import annotations

from html import escape
import re

import pandas as pd

from .utils import ROOT

INDEX = ROOT / 'docs/index.html'
BOARD = ROOT / 'outputs/weekly_board.csv'


def main() -> None:
    if not INDEX.exists():
        raise AssertionError('docs/index.html is missing.')
    if not BOARD.exists():
        raise AssertionError('outputs/weekly_board.csv is missing.')

    html = INDEX.read_text(encoding='utf-8')
    board = pd.read_csv(BOARD)

    required = {
        'weather augmentation marker': '/* SITE_WEATHER_TOOLS */',
        'radar synchronization marker': '/* SITE_RADAR_SYNC */',
        'radar toggle': 'id="radarToggle"',
        'radar status': 'id="radarStatus"',
        'weather filter': 'id="weatherFilter"',
        'status filter': 'id="statusFilter"',
        'division filter': 'id="divisionFilter"',
        'kickoff sort': 'data-weather-sort="2"',
        'temperature sort': 'data-weather-sort="4"',
        'wind sort': 'data-weather-sort="5"',
        'gust sort': 'data-weather-sort="6"',
        'humidity sort': 'data-weather-sort="7"',
        'precipitation sort': 'data-weather-sort="8"',
        'weather type metadata': 'data-weather-types=',
        'weather sort javascript': '// SITE_WEATHER_TOOLS',
        'radar javascript': '// SITE_RADAR_SYNC',
        'radar excluded from status filters': "querySelectorAll('.map-filter[data-map-status]')",
    }
    missing = [name for name, needle in required.items() if needle not in html]
    if missing:
        raise AssertionError('Final website is missing required production UI: ' + ', '.join(missing))

    for option in ['RAIN', 'SNOW', 'WIND', 'HOT', 'COLD', 'INDOOR']:
        if f'value="{option}"' not in html:
            raise AssertionError(f'Weather filter option is missing: {option}')

    # Only count actual board rows. Site-only diagnostic detail rows use a
    # different <tr> signature and intentionally do not count here.
    rendered_rows = len(re.findall(r'<tr data-status="[^"]+"[^>]*data-division="[^"]+"', html))
    if rendered_rows != len(board):
        raise AssertionError(
            f'Final website row count drifted from weekly_board.csv: html={rendered_rows}, board={len(board)}'
        )

    if not board.empty:
        sample = board.iloc[0]
        matchup = f"{sample.get('away_team', '')} @ {sample.get('home_team', '')}"
        # The dashboard correctly HTML-escapes team names (for example,
        # "East Texas A&M" renders as "East Texas A&amp;M"). Compare against
        # the escaped representation so the smoke test validates content
        # without falsely rejecting safe HTML encoding.
        rendered_matchup = escape(matchup)
        if rendered_matchup not in html:
            raise AssertionError(f'Final website does not contain first board matchup: {matchup}')

    print(
        f'Final website smoke test passed: {len(board)} board rows, radar present, '
        'kickoff/weather sorting present, and weather-type filtering present.'
    )


if __name__ == '__main__':
    main()
