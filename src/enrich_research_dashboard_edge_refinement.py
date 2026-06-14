from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from .utils import ROOT

TAB_INSERT_AFTER = '<button class="tab" data-target="weekly">Weekly Card</button>'
SECTION_INSERT_BEFORE = '    <section class="section" id="method">'
TAB_HTML = '<button class="tab" data-target="refinement">Edge Refinement</button>'


def read_output(name: str) -> pd.DataFrame:
    path = ROOT / 'outputs' / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value, col: str = '') -> str:
    if pd.isna(value):
        return ''
    if isinstance(value, (int, float)):
        if 'rate' in col or 'roi' in col or 'gap' in col:
            return f'{value * 100:.1f}%'
        if 'units' in col or 'drawdown' in col:
            return f'{value:+.1f}'
        if abs(value) >= 100:
            return f'{value:.0f}'
        return f'{value:.2f}'
    return escape(str(value))


def table_html(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return '<p>No rows generated yet. Run the manual workflow again.</p>'
    view = df[[c for c in cols if c in df.columns]].head(max_rows).copy()
    if view.empty:
        return '<p>No display columns found.</p>'
    head = ''.join(f'<th>{escape(c)}</th>' for c in view.columns)
    rows = []
    for _, row in view.iterrows():
        rows.append('<tr>' + ''.join(f'<td>{fmt(row[c], c)}</td>' for c in view.columns) + '</tr>')
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def build_section() -> str:
    summary = read_output('edge_refinement_summary.csv')
    shortlist = read_output('edge_refinement_shortlist.csv')
    by_season = read_output('edge_refinement_by_season.csv')

    if not summary.empty:
        top_score = summary.sort_values(['score', 'graded'], ascending=[False, False]).head(20)
        top_roi = summary.sort_values(['roi_per_1u', 'graded'], ascending=[False, False]).head(20)
        overfit_watch = summary[summary['overfit_flags'].astype(str) != 'none'].sort_values(['roi_per_1u', 'graded'], ascending=[False, False]).head(15)
    else:
        top_score = top_roi = overfit_watch = pd.DataFrame()

    shortlist_cols = ['filter_name', 'category', 'graded', 'hit_rate', 'roi_per_1u', 'recent_2022_plus_roi', 'positive_season_rate', 'max_drawdown_units', 'overfit_flags', 'score']
    top_cols = ['filter_name', 'category', 'description', 'graded', 'hit_rate', 'roi_per_1u', 'target_roi_10pct_gap', 'recent_2022_plus_roi', 'positive_season_rate', 'max_drawdown_units', 'overfit_flags']
    season_cols = ['filter_name', 'season', 'graded', 'wins', 'losses', 'hit_rate', 'roi_per_1u', 'net_units_1u_each']

    return f'''
    <section class="section" id="refinement">
      <div class="panel">
        <h2>Edge refinement research</h2>
        <p>This section starts with the HGB under signal and tests whether extra filters can improve the profile toward a modest 10%+ historical ROI without relying on obvious overfit. It tests high-total buckets, weather interactions, provider/line-market proxies, team-style proxies, model consensus, and no-play filters.</p>
        <div class="grid3">
          <div class="rule"><strong>Goal</strong><span>Find tighter under candidates that can improve ROI while keeping enough sample size.</span></div>
          <div class="rule"><strong>Guardrail</strong><span>ROI alone is not enough. Rows are ranked by ROI, recent ROI, season stability, sample size, and drawdown.</span></div>
          <div class="rule"><strong>CLV limitation</strong><span>True current-line-to-closing-line value is not available historically yet. Provider spread is only a proxy until live tracking begins.</span></div>
        </div>
      </div>

      <div class="panel">
        <h2>Shortlist candidates</h2>
        <p>These are rows that passed the basic guardrails: enough graded plays, positive recent ROI, reasonable season stability, and positive overall ROI. They are still research candidates, not automatically live-ready.</p>
        {table_html(shortlist, shortlist_cols, 25)}
      </div>

      <div class="grid">
        <div class="panel">
          <h2>Top rows by stability score</h2>
          <p>This ranking balances ROI with sample size, recent performance, season stability, and drawdown.</p>
          {table_html(top_score, top_cols, 20)}
        </div>
        <div class="panel">
          <h2>Top rows by raw ROI</h2>
          <p>Useful for finding research leads, but high ROI with small samples should be treated carefully.</p>
          {table_html(top_roi, top_cols, 20)}
        </div>
      </div>

      <div class="panel">
        <h2>High-ROI rows with overfit flags</h2>
        <p>These are not necessarily bad, but the flags show why they should be treated as research leads instead of production rules.</p>
        {table_html(overfit_watch, top_cols, 15)}
      </div>

      <div class="panel">
        <h2>Season-by-season check</h2>
        <p>Use this to see whether a candidate worked across several years or was carried by one or two seasons.</p>
        {table_html(by_season, season_cols, 80)}
      </div>
    </section>
'''


def enrich_html(html: str) -> str:
    if 'data-target="refinement"' not in html:
        html = html.replace(TAB_INSERT_AFTER, TAB_INSERT_AFTER + '\n      ' + TAB_HTML)
    if 'id="refinement"' not in html:
        html = html.replace(SECTION_INSERT_BEFORE, build_section() + '\n' + SECTION_INSERT_BEFORE)
    return html


def enrich_file(path: Path) -> None:
    if not path.exists():
        return
    html = path.read_text(encoding='utf-8')
    path.write_text(enrich_html(html), encoding='utf-8')


def main() -> None:
    enrich_file(ROOT / 'docs' / 'index.html')
    enrich_file(ROOT / 'outputs' / 'research_dashboard.html')
    print('Added edge refinement section to research dashboard HTML')


if __name__ == '__main__':
    main()
