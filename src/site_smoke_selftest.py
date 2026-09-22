from __future__ import annotations

from html import escape
import re

import pandas as pd

from .predict_week import GENERAL_QUALIFY_EDGE, GENERAL_QUALIFY_TOTAL
from .utils import ROOT

INDEX = ROOT / 'docs/index.html'
BOARD = ROOT / 'outputs/weekly_board.csv'
RESEARCH = ROOT / 'docs/research.html'


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

    current_screen = (
        f'FBS/general screen: HGB under edge ≥{GENERAL_QUALIFY_EDGE:.1f} '
        f'+ total ≥{GENERAL_QUALIFY_TOTAL:.0f}.'
    )
    if current_screen not in html:
        raise AssertionError(f'Live website is missing the current production screen copy: {current_screen}')
    stale_screen = 'FBS/general screen: HGB under edge ≥3.5 + total ≥56.'
    if GENERAL_QUALIFY_EDGE != 3.5 and stale_screen in html:
        raise AssertionError('Live website still presents the legacy 3.5/56 screen as current.')

    if RESEARCH.exists():
        research_html = RESEARCH.read_text(encoding='utf-8')
        if 'Current production protocol 2026.6' not in research_html:
            raise AssertionError('Research page is missing the current Protocol 2026.6 production banner.')
        if 'legacy research' not in research_html.lower():
            raise AssertionError('Research page does not distinguish legacy research from current production.')
        stale_research_phrases = [
            'Current conclusion:</strong> the best historical candidate is HGB-driven unders with a 3.5+ point edge',
            'HGB under, model edge ≥ 3.5 points, total bin 56+.',
            'The current historical strategy is selective unders only, especially high-total games with at least a 3.5-point model edge.',
        ]
        stale = [phrase for phrase in stale_research_phrases if phrase in research_html]
        if stale:
            raise AssertionError('Research page still presents legacy 3.5 research as current production.')

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
        'kickoff/weather sorting present, weather-type filtering present, and current protocol copy aligned.'
    )


if __name__ == '__main__':
    main()
