from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ROOT, ensure_dir


CSV_SOURCES = {
    'model_bakeoff': 'outputs/model_bakeoff_summary.csv',
    'edge_recent': 'outputs/model_edge_by_recent_period.csv',
    'edge_side': 'outputs/model_edge_by_side.csv',
    'edge_total': 'outputs/model_edge_by_total_bin.csv',
    'edge_provider': 'outputs/model_edge_by_provider.csv',
    'combo_summary': 'outputs/parlay_backtest_summary.csv',
    'combo_season': 'outputs/parlay_by_season.csv',
    'weekly_cards': 'outputs/weekly_card_backtest_summary.csv',
    'weekly_cards_by_season': 'outputs/weekly_card_backtest_by_season.csv',
    'straight_equivalent': 'outputs/weekly_card_straight_equivalent_summary.csv',
}


def read_csv(path: str) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    cleaned = df.copy()
    for col in cleaned.columns:
        if pd.api.types.is_float_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].round(6)
    cleaned = cleaned.replace({pd.NA: None, float('inf'): None, float('-inf'): None})
    cleaned = cleaned.where(pd.notnull(cleaned), None)
    return cleaned.to_dict(orient='records')


def first_row(df: pd.DataFrame, **filters: Any) -> dict[str, Any]:
    if df.empty:
        return {}
    out = df.copy()
    for col, val in filters.items():
        if col not in out.columns:
            return {}
        out = out[out[col] == val]
    if out.empty:
        return {}
    return out.iloc[0].to_dict()


def fmt_pct(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return 'n/a'
        return f'{float(value) * 100:.1f}%'
    except Exception:
        return 'n/a'


def fmt_units(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return 'n/a'
        return f'{float(value):+.1f}u'
    except Exception:
        return 'n/a'


def build_metrics(data: dict[str, pd.DataFrame]) -> dict[str, str]:
    side = data['edge_side']
    weekly = data['weekly_cards']
    straight = data['straight_equivalent']
    recent = data['edge_recent']

    hgb_under_35 = first_row(side, model='hist_gradient_boosting', threshold=3.5, side='under')
    hgb_under_50 = first_row(side, model='hist_gradient_boosting', threshold=5.0, side='under')
    weekly_top = first_row(weekly, card_strategy='hgb_under_3p5_high_total_2leg_top1_nonoverlap')
    straight_top = first_row(straight, card_strategy='hgb_under_3p5_high_total_2leg_top1_nonoverlap')
    recent_35 = first_row(recent, model='hist_gradient_boosting', threshold=3.5, recent_period='2022-2025')

    return {
        'straight_hit': fmt_pct(hgb_under_35.get('hit_rate')),
        'straight_roi': fmt_pct(hgb_under_35.get('roi_per_1u')),
        'strict_hit': fmt_pct(hgb_under_50.get('hit_rate')),
        'strict_roi': fmt_pct(hgb_under_50.get('roi_per_1u')),
        'weekly_cards': str(int(weekly_top.get('cards', 0))) if weekly_top else 'n/a',
        'weekly_roi': fmt_pct(weekly_top.get('roi_per_card')),
        'weekly_dd': fmt_units(weekly_top.get('max_drawdown_units')),
        'leg_hit': fmt_pct(straight_top.get('hit_rate')),
        'leg_roi': fmt_pct(straight_top.get('roi_per_leg')),
        'recent_roi': fmt_pct(recent_35.get('roi_per_1u')),
        'recent_hit': fmt_pct(recent_35.get('hit_rate')),
    }


def dashboard_html(payload: dict[str, Any], metrics: dict[str, str]) -> str:
    data_json = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CFB Weather Totals Research Dashboard</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: rgba(18, 27, 51, 0.88);
      --panel2: rgba(23, 35, 66, 0.92);
      --text: #eef4ff;
      --muted: #aab8d7;
      --line: rgba(255,255,255,0.12);
      --accent: #7dd3fc;
      --accent2: #a7f3d0;
      --warn: #fbbf24;
      --bad: #fb7185;
      --good: #34d399;
      --shadow: 0 18px 45px rgba(0,0,0,.32);
      --radius: 22px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 15%, rgba(125,211,252,.20), transparent 30%),
        radial-gradient(circle at 85% 5%, rgba(167,243,208,.14), transparent 28%),
        linear-gradient(135deg, #080b14 0%, #0b1020 42%, #111827 100%);
      font-family: var(--sans);
      line-height: 1.5;
    }}
    a {{ color: var(--accent); }}
    .shell {{ width: min(1320px, calc(100vw - 32px)); margin: 0 auto; }}
    header {{ padding: 44px 0 22px; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: 22px;
      align-items: stretch;
    }}
    .hero-card, .panel {{
      background: linear-gradient(180deg, rgba(18,27,51,.94), rgba(14,23,42,.88));
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero-card {{ padding: 32px; position: relative; overflow: hidden; }}
    .hero-card:after {{
      content: '';
      position: absolute;
      inset: auto -80px -120px auto;
      width: 280px; height: 280px;
      background: radial-gradient(circle, rgba(125,211,252,.24), transparent 68%);
      pointer-events: none;
    }}
    .eyebrow {{ color: var(--accent2); font-weight: 800; letter-spacing: .14em; text-transform: uppercase; font-size: 12px; }}
    h1 {{ font-size: clamp(34px, 5vw, 62px); line-height: .98; margin: 12px 0 18px; }}
    h2 {{ font-size: 26px; margin: 0 0 14px; }}
    h3 {{ font-size: 17px; margin: 0 0 8px; }}
    p {{ color: var(--muted); margin: 0 0 14px; }}
    .pills {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }}
    .pill {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,.06);
      color: var(--text);
      padding: 9px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
    }}
    .summary-card {{ padding: 24px; display: grid; gap: 14px; }}
    .verdict {{
      padding: 18px;
      border-radius: 18px;
      background: rgba(52,211,153,.10);
      border: 1px solid rgba(52,211,153,.28);
    }}
    .verdict strong {{ color: var(--accent2); }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 24px 0; }}
    .metric {{
      padding: 18px;
      border-radius: 18px;
      background: rgba(255,255,255,.06);
      border: 1px solid var(--line);
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 800; }}
    .metric .value {{ font-size: 28px; font-weight: 900; margin-top: 5px; }}
    .metric .sub {{ color: var(--muted); font-size: 13px; }}
    nav.sticky {{
      position: sticky; top: 0; z-index: 50;
      backdrop-filter: blur(16px);
      background: rgba(8,11,20,.78);
      border-bottom: 1px solid var(--line);
    }}
    .tabs {{ display: flex; gap: 8px; overflow-x: auto; padding: 12px 0; }}
    .tab {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,.06);
      color: var(--muted);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      white-space: nowrap;
      font-weight: 800;
    }}
    .tab.active {{ color: #04111f; background: linear-gradient(135deg, var(--accent), var(--accent2)); border-color: transparent; }}
    main {{ padding: 24px 0 60px; }}
    .section {{ display: none; }}
    .section.active {{ display: block; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }}
    .panel {{ padding: 22px; margin-bottom: 18px; }}
    .callout {{
      background: rgba(251,191,36,.10);
      border: 1px solid rgba(251,191,36,.32);
      border-radius: 18px;
      padding: 16px;
      color: #fde68a;
    }}
    .good {{ color: var(--good); }} .bad {{ color: var(--bad); }} .warn {{ color: var(--warn); }}
    .rule-list {{ display: grid; gap: 10px; }}
    .rule {{
      padding: 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.045);
    }}
    .rule strong {{ display: block; margin-bottom: 4px; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin: 12px 0 14px; }}
    input, select {{
      background: rgba(255,255,255,.08);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      outline: none;
    }}
    input::placeholder {{ color: rgba(238,244,255,.50); }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 16px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 820px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,.08); font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #121b33; color: #dbeafe; z-index: 1; cursor: pointer; }}
    td {{ color: #d9e5fb; }}
    tr:hover td {{ background: rgba(125,211,252,.06); }}
    .chart {{ display: grid; gap: 10px; margin-top: 12px; }}
    .bar-row {{ display: grid; grid-template-columns: 180px 1fr 88px; gap: 10px; align-items: center; }}
    .bar-label {{ color: var(--muted); font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ height: 12px; background: rgba(255,255,255,.08); border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent2)); width: 0%; }}
    .bar-value {{ font-family: var(--mono); color: var(--text); font-size: 12px; text-align: right; }}
    .footer {{ color: var(--muted); padding: 28px 0 50px; font-size: 13px; }}
    @media (max-width: 920px) {{
      .hero, .grid, .grid3, .metrics {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 42px; }}
      .bar-row {{ grid-template-columns: 130px 1fr 70px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="shell hero">
      <section class="hero-card">
        <div class="eyebrow">CFB Weather Totals</div>
        <h1>Research Dashboard</h1>
        <p>An interactive, readable view of the historical modeling work: straight model edges, weather/total filters, combo research, realistic weekly-card testing, and the disciplined strategy moving forward.</p>
        <div class="pills">
          <span class="pill">Market residual target</span>
          <span class="pill">HGB under focus</span>
          <span class="pill">High-total screen</span>
          <span class="pill">Weekly-card method</span>
        </div>
      </section>
      <aside class="hero-card summary-card">
        <div class="verdict"><strong>Current conclusion:</strong> the best historical candidate is HGB-driven unders with a 3.5+ point edge, especially in high-total games. Weekly cards should remain selective and variable by slate.</div>
        <p><strong>Not a lock machine.</strong> The dashboard is designed to show where the historical signal is strongest, where it failed, and when a week should have few or no targets.</p>
      </aside>
    </div>
    <div class="shell metrics">
      <div class="metric"><div class="label">HGB 3.5+ under</div><div class="value">{escape(metrics['straight_hit'])}</div><div class="sub">straight hit rate · {escape(metrics['straight_roi'])} ROI</div></div>
      <div class="metric"><div class="label">Top weekly 2-leg</div><div class="value">{escape(metrics['weekly_roi'])}</div><div class="sub">ROI · {escape(metrics['weekly_dd'])} max drawdown</div></div>
      <div class="metric"><div class="label">Same legs straight</div><div class="value">{escape(metrics['leg_hit'])}</div><div class="sub">hit rate · {escape(metrics['leg_roi'])} ROI</div></div>
      <div class="metric"><div class="label">Recent HGB 3.5+</div><div class="value">{escape(metrics['recent_hit'])}</div><div class="sub">2022–2025 · {escape(metrics['recent_roi'])} ROI</div></div>
    </div>
  </header>

  <nav class="sticky">
    <div class="shell tabs" id="tabs">
      <button class="tab active" data-target="overview">Overview</button>
      <button class="tab" data-target="straight">Straight Model Edge</button>
      <button class="tab" data-target="combos">Combo Research</button>
      <button class="tab" data-target="weekly">Weekly Card</button>
      <button class="tab" data-target="method">Methodology & Strategy</button>
      <button class="tab" data-target="tables">Explore Tables</button>
    </div>
  </nav>

  <main class="shell">
    <section class="section active" id="overview">
      <div class="grid">
        <div class="panel">
          <h2>What the model is trying to beat</h2>
          <p>The target is market residual: actual total points minus the closing total. Positive residuals mean the game went over the market total. Negative residuals mean the game went under.</p>
          <div class="rule-list">
            <div class="rule"><strong>Core idea</strong><span>Control for the total first, then look for repeatable weather/context/model signals that explain residual movement.</span></div>
            <div class="rule"><strong>Best signal so far</strong><span>HistGradientBoosting unders at 3.5+ model edge, with high-total games as the cleanest weekly-card filter.</span></div>
            <div class="rule"><strong>What changed after combo testing</strong><span>The broad all-combo test looked flashy, but the weekly-card test is the more realistic method because it limits selections by week.</span></div>
          </div>
        </div>
        <div class="panel">
          <h2>Current decision thresholds</h2>
          <div class="rule-list">
            <div class="rule"><strong class="good">Target candidate</strong><span>HGB under, model edge ≥ 3.5 points.</span></div>
            <div class="rule"><strong class="good">Strongest screen</strong><span>HGB under, model edge ≥ 3.5 points, total bin 56+.</span></div>
            <div class="rule"><strong class="warn">Strict but thinner</strong><span>HGB under, model edge ≥ 5.0 points. Useful, but some weekly-card versions were less stable.</span></div>
            <div class="rule"><strong class="bad">No-play / paper only</strong><span>Overs, edge below 3.5, missing line/weather confidence, or extra combo legs beyond the top weekly 2-leg card.</span></div>
          </div>
        </div>
      </div>
      <div class="panel">
        <h2>Best weekly-card results</h2>
        <div id="weeklyOverviewChart" class="chart"></div>
      </div>
    </section>

    <section class="section" id="straight">
      <div class="panel">
        <h2>Straight model edge</h2>
        <p>Use this section to compare edge thresholds, side splits, recent periods, total bins, and providers. The key test is whether the model remains positive after filters and across time.</p>
        <div class="grid">
          <div><h3>Side split ROI</h3><div id="sideChart" class="chart"></div></div>
          <div><h3>Recent-period ROI</h3><div id="recentChart" class="chart"></div></div>
        </div>
      </div>
      <div class="panel"><h2>Model edge by side</h2><div data-table="edge_side"></div></div>
      <div class="grid">
        <div class="panel"><h2>By total bin</h2><div data-table="edge_total"></div></div>
        <div class="panel"><h2>By provider</h2><div data-table="edge_provider"></div></div>
      </div>
    </section>

    <section class="section" id="combos">
      <div class="panel">
        <h2>Combo research</h2>
        <p>The all-combo test is useful for finding candidate structures, but it can overstate returns because it grades many combinations from the same weekly slate. Treat it as a screening tool, not the final weekly process.</p>
        <div class="callout">The more realistic weekly-card test is the better guide for actual weekly use.</div>
      </div>
      <div class="panel"><h2>All-combo summary</h2><div data-table="combo_summary"></div></div>
      <div class="panel"><h2>All-combo season check</h2><div data-table="combo_season"></div></div>
    </section>

    <section class="section" id="weekly">
      <div class="panel">
        <h2>Weekly-card methodology</h2>
        <p>This is the bridge between research and what a real weekly process could look like. The model selects a limited number of top-ranked weekly cards instead of every possible combination.</p>
        <div class="grid3">
          <div class="rule"><strong>Primary card</strong><span>Top 1 weekly 2-leg combo from HGB under 3.5+ high-total games.</span></div>
          <div class="rule"><strong>Why top 1?</strong><span>It had a strong historical return with lower drawdown than broader weekly cards.</span></div>
          <div class="rule"><strong>Default combo cap</strong><span>0 or 1 weekly 2-leg card. Some weeks should have no combo.</span></div>
        </div>
      </div>
      <div class="panel"><h2>Weekly-card summary</h2><div data-table="weekly_cards"></div></div>
      <div class="grid">
        <div class="panel"><h2>Season-by-season weekly cards</h2><div data-table="weekly_cards_by_season"></div></div>
        <div class="panel"><h2>Same legs as straight plays</h2><div data-table="straight_equivalent"></div></div>
      </div>
    </section>

    <section class="section" id="method">
      <div class="grid">
        <div class="panel">
          <h2>Strategy moving forward</h2>
          <div class="rule-list">
            <div class="rule"><strong>1. Start with the target board</strong><span>Every game gets a classification: target candidate, lean/watch, paper only, no-play, or avoid.</span></div>
            <div class="rule"><strong>2. Let volume vary</strong><span>The number of targets should change week to week. Some weeks may have no qualifying combo.</span></div>
            <div class="rule"><strong>3. Track process quality</strong><span>Record line at decision time, closing line, weather snapshot, model edge, result, and closing-line value.</span></div>
            <div class="rule"><strong>4. Keep combo logic selective</strong><span>Default to only the top 1 weekly 2-leg card when two qualifying high-total under targets exist.</span></div>
          </div>
        </div>
        <div class="panel">
          <h2>No-play logic</h2>
          <div class="rule-list">
            <div class="rule"><strong class="bad">Edge below 3.5</strong><span>Does not clear the current historical threshold.</span></div>
            <div class="rule"><strong class="bad">Over side</strong><span>Overs did not validate as cleanly as unders in the current tests.</span></div>
            <div class="rule"><strong class="bad">Weather/line uncertainty</strong><span>Missing current line or uncertain forecast means no production target.</span></div>
            <div class="rule"><strong class="warn">Three-leg cards</strong><span>Positive in places, but season stability and drawdown make them paper-only by default.</span></div>
          </div>
        </div>
      </div>
      <div class="panel">
        <h2>What the Week 1 site should add next</h2>
        <p>The next production page should ingest a live weekly board with current totals and forecast weather, then apply the same thresholds shown here. This research dashboard is the explanation layer; the weekly target board will be the action layer.</p>
      </div>
    </section>

    <section class="section" id="tables">
      <div class="panel">
        <h2>Explore all output tables</h2>
        <p>Search, sort, and compare the generated research tables directly. Percent fields are stored as decimals in the source data.</p>
        <label for="tableSelect">Table </label>
        <select id="tableSelect"></select>
        <div id="tableExplorer"></div>
      </div>
    </section>
  </main>
  <footer class="shell footer">Generated from the latest repository outputs. Historical results require live tracking with current lines and forecast weather before confidence can be increased.</footer>

  <script>
    const DATA = {data_json};
    const TABLE_LABELS = {{
      model_bakeoff: 'Model bakeoff', edge_recent: 'Edge by recent period', edge_side: 'Edge by side', edge_total: 'Edge by total bin', edge_provider: 'Edge by provider', combo_summary: 'All-combo summary', combo_season: 'All-combo by season', weekly_cards: 'Weekly-card summary', weekly_cards_by_season: 'Weekly-card by season', straight_equivalent: 'Straight equivalent'
    }};
    function fmt(v, col='') {{
      if (v === null || v === undefined || Number.isNaN(v)) return '';
      if (typeof v === 'number') {{
        if (col.includes('rate') || col.includes('roi')) return (v * 100).toFixed(1) + '%';
        if (col.includes('units') || col.includes('drawdown')) return (v >= 0 ? '+' : '') + v.toFixed(1);
        return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
      }}
      return String(v);
    }}
    function renderTable(key, root, rows=null) {{
      const data = rows || DATA[key] || [];
      if (!data.length) {{ root.innerHTML = '<p>No data available.</p>'; return; }}
      const cols = Object.keys(data[0]);
      root.innerHTML = `
        <div class="toolbar"><input type="search" placeholder="Search table..." data-search="${{key}}"></div>
        <div class="table-wrap"><table><thead><tr>${{cols.map(c=>`<th data-col="${{c}}">${{c}}</th>`).join('')}}</tr></thead><tbody></tbody></table></div>`;
      const tbody = root.querySelector('tbody');
      let current = [...data];
      function draw(rowsToDraw) {{
        tbody.innerHTML = rowsToDraw.map(r => `<tr>${{cols.map(c=>`<td>${{fmt(r[c], c)}}</td>`).join('')}}</tr>`).join('');
      }}
      draw(current);
      root.querySelector('input').addEventListener('input', e => {{
        const q = e.target.value.toLowerCase();
        current = data.filter(r => JSON.stringify(r).toLowerCase().includes(q));
        draw(current);
      }});
      root.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {{
        const col = th.dataset.col;
        current.sort((a,b) => {{
          const av = a[col], bv = b[col];
          if (typeof av === 'number' && typeof bv === 'number') return bv - av;
          return String(av ?? '').localeCompare(String(bv ?? ''));
        }});
        draw(current);
      }}));
    }}
    function renderBars(rootId, rows, labelFn, valueFn) {{
      const root = document.getElementById(rootId);
      if (!root) return;
      const vals = rows.map(valueFn).filter(v => typeof v === 'number' && isFinite(v));
      const maxAbs = Math.max(...vals.map(v => Math.abs(v)), 0.01);
      root.innerHTML = rows.map(r => {{
        const v = valueFn(r) || 0;
        const w = Math.min(100, Math.abs(v) / maxAbs * 100);
        return `<div class="bar-row"><div class="bar-label" title="${{labelFn(r)}}">${{labelFn(r)}}</div><div class="bar-track"><div class="bar-fill" style="width:${{w}}%"></div></div><div class="bar-value">${{(v*100).toFixed(1)}}%</div></div>`;
      }}).join('');
    }}
    document.querySelectorAll('[data-table]').forEach(el => renderTable(el.dataset.table, el));
    const selector = document.getElementById('tableSelect');
    Object.keys(TABLE_LABELS).forEach(k => selector.insertAdjacentHTML('beforeend', `<option value="${{k}}">${{TABLE_LABELS[k]}}</option>`));
    function drawExplorer() {{ renderTable(selector.value, document.getElementById('tableExplorer')); }}
    selector.addEventListener('change', drawExplorer); drawExplorer();
    document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {{
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
      btn.classList.add('active'); document.getElementById(btn.dataset.target).classList.add('active');
    }}));
    renderBars('weeklyOverviewChart', (DATA.weekly_cards || []).slice().sort((a,b)=>(b.roi_per_card||0)-(a.roi_per_card||0)), r => r.card_strategy, r => r.roi_per_card);
    renderBars('sideChart', (DATA.edge_side || []).filter(r => r.model === 'hist_gradient_boosting' && [3.5,5.0].includes(Number(r.threshold))), r => `${{r.threshold}} ${{r.side}}`, r => r.roi_per_1u);
    renderBars('recentChart', (DATA.edge_recent || []).filter(r => r.model === 'hist_gradient_boosting' && [3.5,5.0].includes(Number(r.threshold))), r => `${{r.threshold}} ${{r.recent_period}}`, r => r.roi_per_1u);
  </script>
</body>
</html>"""


def main() -> None:
    data = {key: read_csv(path) for key, path in CSV_SOURCES.items()}
    payload = {key: clean_records(df) for key, df in data.items()}
    metrics = build_metrics(data)
    html = dashboard_html(payload, metrics)

    outputs = ensure_dir('outputs')
    docs = ensure_dir('docs')
    (outputs / 'research_dashboard.html').write_text(html, encoding='utf-8')
    (docs / 'index.html').write_text(html, encoding='utf-8')
    print(f'Wrote {(outputs / "research_dashboard.html")} and {(docs / "index.html")}')


if __name__ == '__main__':
    main()
