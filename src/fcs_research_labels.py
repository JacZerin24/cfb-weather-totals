from __future__ import annotations

import re
from html import escape

import pandas as pd

from .utils import ROOT


FCS_LABEL = 'FCS RESEARCH QUALIFIES'


def _fcs_qualifier_matchups(board: pd.DataFrame) -> set[tuple[str, str]]:
    if board.empty or 'status' not in board.columns:
        return set()
    division = board.get('division_track', pd.Series('', index=board.index)).astype(str).str.upper()
    mask = board['status'].astype(str).eq('QUALIFIES') & division.eq('FCS')
    return {
        (str(row.get('away_team') or '').strip(), str(row.get('home_team') or '').strip())
        for _, row in board[mask].iterrows()
    }


def _label_target_cards(html: str, matchups: set[tuple[str, str]]) -> str:
    if not matchups:
        return html

    def repl(match: re.Match[str]) -> str:
        article = match.group(0)
        for away, home in matchups:
            if escape(away) in article and escape(home) in article:
                return article.replace(
                    '<span class="badge qualifies">QUALIFIES</span>',
                    f'<span class="badge fcs-research-qualifies">{FCS_LABEL}</span>',
                    1,
                )
        return article

    return re.sub(r'<article class="target-card">.*?</article>', repl, html, flags=re.S)


def _label_fcs_table_rows(html: str) -> str:
    def repl(match: re.Match[str]) -> str:
        row = match.group(0)
        return row.replace(
            '<span class="badge qualifies">QUALIFIES</span>',
            f'<span class="badge fcs-research-qualifies">{FCS_LABEL}</span>',
            1,
        )

    return re.sub(
        r'<tr data-status="QUALIFIES" data-division="FCS">.*?</tr>',
        repl,
        html,
        flags=re.S,
    )


def _label_map_popup(html: str) -> str:
    old = "escapeHtml(game.status + ' · ' + (game.division || 'OTHER'))"
    new = "escapeHtml((game.status === 'QUALIFIES' && game.division === 'FCS' ? 'FCS RESEARCH QUALIFIES' : game.status) + ' · ' + (game.division || 'OTHER'))"
    return html.replace(old, new)


def label_html(html: str, board: pd.DataFrame) -> str:
    matchups = _fcs_qualifier_matchups(board)
    html = _label_target_cards(html, matchups)
    html = _label_fcs_table_rows(html)
    html = _label_map_popup(html)

    if '.fcs-research-qualifies {' not in html:
        css = '''
    .fcs-research-qualifies { color:#ddd6fe; background:rgba(139,92,246,.16); border:1px solid rgba(139,92,246,.48); }
'''
        html = html.replace('    .parlay { padding:20px; }', css + '    .parlay { padding:20px; }', 1)
    return html


def main() -> None:
    board_path = ROOT / 'outputs/weekly_board.csv'
    if not board_path.exists():
        raise RuntimeError('weekly_board.csv is required before applying FCS research labels.')
    board = pd.read_csv(board_path)

    changed = 0
    for relative in ['docs/index.html', 'outputs/live_dashboard.html']:
        path = ROOT / relative
        if not path.exists():
            continue
        html = path.read_text(encoding='utf-8')
        updated = label_html(html, board)
        path.write_text(updated, encoding='utf-8')
        changed += 1

    if not changed:
        raise RuntimeError('No live dashboard HTML files were available for FCS research labeling.')
    print(f'Applied distinct FCS research qualifier labels to {len(_fcs_qualifier_matchups(board))} game(s).')


if __name__ == '__main__':
    main()
