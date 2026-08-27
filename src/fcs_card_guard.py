from __future__ import annotations

import re
from html import escape

import pandas as pd

from .utils import ROOT, write_df


def fmt_total(value) -> str:
    try:
        return f'{float(value):.1f}'
    except Exception:
        return '—'


def eligible_card_targets(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return board
    mask = board['status'].eq('QUALIFIES')
    if 'division_track' in board.columns:
        mask &= ~board['division_track'].astype(str).str.upper().eq('FCS')
    out = board[mask].copy()
    if 'abs_pred_edge' in out.columns:
        out = out.sort_values('abs_pred_edge', ascending=False)
    return out


def card_html(eligible: pd.DataFrame, fcs_qualifiers: int) -> str:
    if len(eligible) >= 2:
        two = eligible.head(2)
        legs = ' + '.join(
            f"UNDER {fmt_total(row.get('closing_total'))} ({escape(str(row.get('away_team', '')))} @ {escape(str(row.get('home_team', '')))})"
            for _, row in two.iterrows()
        )
        note = 'Built only from the two highest-edge qualifying targets on the original validated weekly-card track.'
        if fcs_qualifiers:
            note += f' {fcs_qualifiers} FCS qualifier(s) are shown as straight research targets but excluded from this legacy card.'
        return f'''<section id="card"><div class="section-head"><div><div class="eyebrow">Historical weekly-card method</div><h2>Top two-leg card</h2></div></div><div class="parlay active-card"><span class="eyebrow">Top weekly 2-leg research card</span><h3>{legs}</h3><p>{escape(note)}</p></div></section>'''

    note = 'Fewer than two non-FCS qualifying targets are available, so the original weekly-card method does not force a parlay.'
    if fcs_qualifiers:
        note += f' {fcs_qualifiers} FCS qualifier(s) are intentionally excluded because the FCS two-leg-card strategy has not been separately validated.'
    return f'''<section id="card"><div class="section-head"><div><div class="eyebrow">Historical weekly-card method</div><h2>Top two-leg card</h2></div></div><div class="parlay"><span class="eyebrow">Top weekly 2-leg research card</span><h3>No card</h3><p>{escape(note)}</p></div></section>'''


def main() -> None:
    board_path = ROOT / 'outputs/weekly_board.csv'
    if not board_path.exists():
        raise RuntimeError('weekly_board.csv is required before applying the FCS card guard.')
    board = pd.read_csv(board_path)
    eligible = eligible_card_targets(board)
    fcs_qualifiers = int(
        (board['status'].eq('QUALIFIES') & board.get('division_track', pd.Series('', index=board.index)).astype(str).str.upper().eq('FCS')).sum()
    )

    if len(eligible) >= 2:
        card = eligible.head(2).copy()
        card['card_leg'] = [1, 2]
        card['card_status'] = 'TOP_2_LEG_RESEARCH_CARD'
    else:
        card = pd.DataFrame([{
            'card_status': 'NO_CARD',
            'note': 'Fewer than two qualifying targets from the original non-FCS weekly-card track are available; FCS qualifiers are excluded from this card.',
        }])
    write_df(card, 'outputs/weekly_card.csv')

    replacement = card_html(eligible, fcs_qualifiers)
    for relative in ['docs/index.html', 'outputs/live_dashboard.html']:
        path = ROOT / relative
        if not path.exists():
            continue
        html = path.read_text(encoding='utf-8')
        updated, count = re.subn(r'<section id="card">.*?</section>', replacement, html, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError(f'Could not locate the weekly card section in {relative}.')
        path.write_text(updated, encoding='utf-8')

    print(f'Applied FCS card guard: {fcs_qualifiers} FCS qualifier(s) excluded from legacy two-leg card.')


if __name__ == '__main__':
    main()
