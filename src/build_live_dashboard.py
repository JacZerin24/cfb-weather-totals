from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .utils import ROOT, ensure_dir

CT = ZoneInfo('America/Chicago')


def read_csv(path: str) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def fmt_num(value: Any, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return '—'
        return f'{float(value):.{decimals}f}'
    except Exception:
        return '—'


def fmt_int(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return '—'
        return str(int(round(float(value))))
    except Exception:
        return '—'


def kickoff_ct(value: Any) -> str:
    try:
        ts = pd.to_datetime(value, utc=True)
        return ts.tz_convert(CT).strftime('%a %b %-d · %-I:%M %p CT')
    except Exception:
        return 'Kickoff TBD'


def status_class(status: str) -> str:
    return {
        'QUALIFIES': 'qualifies',
        'LEAN': 'lean',
        'WATCH': 'watch',
        'NO PLAY': 'no-play',
        'NO LINE': 'no-line',
    }.get(str(status), 'no-play')


def weather_text(row: pd.Series) -> str:
    if bool(row.get('game_indoors', False)):
        return 'Indoor / dome'
    status = str(row.get('nws_status', ''))
    if status != 'ok':
        return str(row.get('weather_summary') or 'NWS forecast unavailable')
    pieces = []
    if pd.notna(row.get('temperature_f')):
        pieces.append(f"{fmt_int(row.get('temperature_f'))}°F")
    if pd.notna(row.get('wind_mph')):
        wind = f"wind {fmt_int(row.get('wind_mph'))} mph"
        if pd.notna(row.get('wind_gust_mph')):
            wind += f", gust {fmt_int(row.get('wind_gust_mph'))}"
        pieces.append(wind)
    if pd.notna(row.get('humidity')):
        pieces.append(f"RH {fmt_int(row.get('humidity'))}%")
    if pd.notna(row.get('precip_probability_pct')):
        pieces.append(f"PoP {fmt_int(row.get('precip_probability_pct'))}%")
    return ' · '.join(pieces) if pieces else str(row.get('weather_summary') or 'NWS forecast available')


def model_text(row: pd.Series) -> str:
    if pd.isna(row.get('pred_market_residual')):
        return 'Not scored'
    residual = float(row['pred_market_residual'])
    direction = 'UNDER' if residual < 0 else 'OVER'
    return f"{direction} edge {abs(residual):.1f} · projected total {fmt_num(row.get('model_projected_total'))}"


def market_text(row: pd.Series) -> str:
    if pd.isna(row.get('closing_total')):
        return 'No current total'
    provider = str(row.get('line_provider') or 'unknown')
    providers = row.get('line_provider_count')
    provider_note = f" · {fmt_int(providers)} providers" if pd.notna(providers) else ''
    return f"O/U {fmt_num(row.get('closing_total'))} · {escape(provider)}{provider_note}"


def target_card(row: pd.Series, rank: int) -> str:
    tags = str(row.get('research_tags') or '').strip()
    tags_html = ''.join(f'<span class="tag">{escape(t.strip())}</span>' for t in tags.split(';') if t.strip())
    return f'''
      <article class="target-card">
        <div class="target-head">
          <span class="rank">#{rank}</span>
          <span class="badge qualifies">QUALIFIES</span>
        </div>
        <h3>{escape(str(row.get('away_team', '')))} <span>@</span> {escape(str(row.get('home_team', '')))}</h3>
        <div class="kickoff">{escape(kickoff_ct(row.get('start_date')))} · {escape(str(row.get('venue_name') or 'Venue TBD'))}</div>
        <div class="big-pick">UNDER {fmt_num(row.get('closing_total'))}</div>
        <div class="edge">Model edge: <strong>{fmt_num(row.get('abs_pred_edge'))} pts</strong> · model total {fmt_num(row.get('model_projected_total'))}</div>
        <div class="weather">{escape(weather_text(row))}</div>
        <p>{escape(str(row.get('decision_reason') or ''))}</p>
        <div class="tags">{tags_html}</div>
      </article>
    '''


def table_rows(board: pd.DataFrame) -> str:
    rows = []
    for _, row in board.iterrows():
        status = str(row.get('status') or 'NO PLAY')
        tags = str(row.get('research_tags') or '').strip()
        tags_html = '<br><span class="small">' + escape(tags) + '</span>' if tags else ''
        rows.append(f'''
          <tr data-status="{escape(status)}">
            <td><span class="badge {status_class(status)}">{escape(status)}</span></td>
            <td><strong>{escape(str(row.get('away_team', '')))} @ {escape(str(row.get('home_team', '')))}</strong><br><span class="small">{escape(kickoff_ct(row.get('start_date')))}</span></td>
            <td>{market_text(row)}</td>
            <td>{escape(weather_text(row))}</td>
            <td>{escape(model_text(row))}{tags_html}</td>
            <td>{escape(str(row.get('decision_reason') or ''))}</td>
          </tr>
        ''')
    return ''.join(rows)


def dashboard_html(board: pd.DataFrame, card: pd.DataFrame, snapshot: dict[str, Any]) -> str:
    generated_raw = snapshot.get('generated_at')
    try:
        generated = datetime.fromisoformat(str(generated_raw).replace('Z', '+00:00')).astimezone(CT).strftime('%b %-d, %Y · %-I:%M %p CT')
    except Exception:
        generated = datetime.now(timezone.utc).astimezone(CT).strftime('%b %-d, %Y · %-I:%M %p CT')

    targets = board[board.get('status', pd.Series(dtype=str)).eq('QUALIFIES')].copy() if not board.empty else pd.DataFrame()
    targets = targets.sort_values('abs_pred_edge', ascending=False) if 'abs_pred_edge' in targets.columns else targets
    target_html = ''.join(target_card(row, i) for i, (_, row) in enumerate(targets.iterrows(), start=1))
    if not target_html:
        target_html = '<div class="empty"><strong>No qualifying targets right now.</strong><br>The model is intentionally allowed to return zero plays when the slate does not meet the validated screen.</div>'

    if len(targets) >= 2:
        two = targets.head(2)
        legs = ' + '.join(f"UNDER {fmt_num(r.get('closing_total'))} ({escape(str(r.get('away_team')))} @ {escape(str(r.get('home_team')))})" for _, r in two.iterrows())
        card_html = f'<div class="parlay active-card"><span class="eyebrow">Top weekly 2-leg research card</span><h3>{legs}</h3><p>Built only from the two highest-edge qualifying high-total HGB under targets. No additional combinations are forced.</p></div>'
    else:
        card_html = '<div class="parlay"><span class="eyebrow">Top weekly 2-leg research card</span><h3>No card</h3><p>Fewer than two games currently qualify, so the model does not force a parlay.</p></div>'

    week = snapshot.get('week', '—')
    season = snapshot.get('season', '—')
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>CFB Weather Totals · Live Board</title>
  <style>
    :root {{ --bg:#08101d; --panel:#111c31; --panel2:#17243d; --text:#f1f6ff; --muted:#9fb0ce; --line:rgba(255,255,255,.11); --cyan:#7dd3fc; --green:#34d399; --yellow:#fbbf24; --red:#fb7185; --purple:#c4b5fd; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--text); background:radial-gradient(circle at 15% 0%,rgba(125,211,252,.17),transparent 28%),linear-gradient(145deg,#060b14,#0b1424 46%,#0e1728); line-height:1.45; }}
    a {{ color:var(--cyan); }} .shell {{ width:min(1380px,calc(100vw - 28px)); margin:0 auto; }}
    header {{ padding:36px 0 18px; }} .hero {{ display:grid; grid-template-columns:1.4fr .6fr; gap:18px; }}
    .panel,.hero-card,.metric,.target-card,.parlay,.empty {{ border:1px solid var(--line); background:linear-gradient(180deg,rgba(20,32,55,.96),rgba(13,24,43,.94)); border-radius:20px; box-shadow:0 18px 44px rgba(0,0,0,.28); }}
    .hero-card {{ padding:28px; }} .eyebrow {{ text-transform:uppercase; letter-spacing:.13em; font-size:12px; font-weight:900; color:#a7f3d0; }}
    h1 {{ font-size:clamp(35px,5vw,62px); line-height:.96; margin:10px 0 14px; }} h2 {{ margin:0 0 14px; font-size:26px; }} h3 {{ margin:0; }}
    p,.muted,.small {{ color:var(--muted); }} .hero-card p {{ max-width:850px; margin:0; }}
    .hero-side {{ display:grid; gap:10px; }} .hero-side div {{ padding:13px 15px; background:rgba(255,255,255,.045); border:1px solid var(--line); border-radius:14px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0; }} .metric {{ padding:16px; }} .metric .label {{ color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.09em; }} .metric .value {{ font-size:29px; font-weight:900; margin-top:4px; }}
    nav {{ position:sticky; top:0; z-index:20; backdrop-filter:blur(14px); background:rgba(6,11,20,.84); border-block:1px solid var(--line); }} nav .shell {{ display:flex; gap:18px; align-items:center; padding:11px 0; overflow:auto; }} nav a {{ text-decoration:none; font-weight:800; white-space:nowrap; }}
    main {{ padding:22px 0 54px; }} section {{ margin-bottom:24px; }} .section-head {{ display:flex; justify-content:space-between; gap:14px; align-items:end; margin-bottom:12px; }} .section-head p {{ margin:0; }}
    .target-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }} .target-card {{ padding:20px; }} .target-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }} .rank {{ font-weight:900; color:var(--cyan); }}
    .target-card h3 {{ font-size:23px; }} .target-card h3 span {{ color:var(--muted); font-weight:500; }} .kickoff {{ color:var(--muted); font-size:13px; margin-top:5px; }} .big-pick {{ font-size:31px; font-weight:950; margin:18px 0 3px; }} .edge {{ color:#cfe3ff; }} .weather {{ margin-top:12px; padding:11px 12px; background:rgba(125,211,252,.07); border:1px solid rgba(125,211,252,.18); border-radius:12px; }} .tags {{ display:flex; flex-wrap:wrap; gap:7px; }} .tag {{ font-size:11px; font-weight:800; color:#dbeafe; border:1px solid var(--line); border-radius:999px; padding:5px 8px; background:rgba(255,255,255,.045); }}
    .badge {{ display:inline-block; font-size:11px; font-weight:950; letter-spacing:.05em; border-radius:999px; padding:6px 9px; }} .qualifies {{ color:#bbf7d0; background:rgba(52,211,153,.13); border:1px solid rgba(52,211,153,.32); }} .lean {{ color:#ddd6fe; background:rgba(196,181,253,.12); border:1px solid rgba(196,181,253,.3); }} .watch {{ color:#fde68a; background:rgba(251,191,36,.12); border:1px solid rgba(251,191,36,.3); }} .no-play,.no-line {{ color:#fecdd3; background:rgba(251,113,133,.09); border:1px solid rgba(251,113,133,.22); }}
    .parlay {{ padding:20px; }} .parlay h3 {{ margin-top:7px; font-size:20px; }} .parlay p {{ margin-bottom:0; }} .active-card {{ border-color:rgba(52,211,153,.35); }} .empty {{ padding:24px; color:var(--muted); }}
    .panel {{ overflow:hidden; }} .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; padding:14px; border-bottom:1px solid var(--line); }} input,select {{ background:#0d1728; border:1px solid var(--line); color:var(--text); padding:9px 11px; border-radius:10px; }}
    .table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; min-width:1180px; }} th,td {{ padding:11px 12px; border-bottom:1px solid rgba(255,255,255,.07); text-align:left; vertical-align:top; font-size:12px; }} th {{ color:#dbeafe; background:#101b30; position:sticky; top:0; }} td {{ color:#dce7f7; }} tr:hover td {{ background:rgba(125,211,252,.045); }}
    footer {{ color:var(--muted); font-size:12px; padding:0 0 38px; }}
    @media(max-width:900px) {{ .hero,.metrics,.target-grid {{ grid-template-columns:1fr; }} .section-head {{ align-items:start; flex-direction:column; }} }}
  </style>
</head>
<body>
  <header class="shell">
    <div class="hero">
      <div class="hero-card">
        <div class="eyebrow">Live weekly board · NWS kickoff forecasts</div>
        <h1>CFB Weather Totals</h1>
        <p>Current market totals are scored with the same HistGradientBoosting residual model validated in the historical research. Outdoor weather comes from the National Weather Service forecast grid at the game venue and kickoff time.</p>
      </div>
      <div class="hero-card hero-side">
        <div><span class="eyebrow">Slate</span><br><strong>{escape(str(season))} · Week {escape(str(week))}</strong></div>
        <div><span class="eyebrow">Updated</span><br><strong>{escape(generated)}</strong></div>
        <div><span class="eyebrow">Mode</span><br><strong>Research / paper tracking</strong></div>
      </div>
    </div>
    <div class="metrics">
      <div class="metric"><div class="label">Games scanned</div><div class="value">{len(board)}</div></div>
      <div class="metric"><div class="label">Current totals</div><div class="value">{int(board['closing_total'].notna().sum()) if 'closing_total' in board.columns else 0}</div></div>
      <div class="metric"><div class="label">Qualifying targets</div><div class="value">{len(targets)}</div></div>
      <div class="metric"><div class="label">NWS-ready outdoor</div><div class="value">{int(board['nws_status'].eq('ok').sum()) if 'nws_status' in board.columns else 0}</div></div>
    </div>
  </header>
  <nav><div class="shell"><a href="#targets">Targets</a><a href="#card">2-leg card</a><a href="#board">Full board</a><a href="research.html">Historical research →</a></div></nav>
  <main class="shell">
    <section id="targets">
      <div class="section-head"><div><div class="eyebrow">Selective by design</div><h2>Qualifying under targets</h2></div><p>HGB under edge ≥3.5 + market total ≥56 + usable kickoff forecast/time.</p></div>
      <div class="target-grid">{target_html}</div>
    </section>
    <section id="card"><div class="section-head"><div><div class="eyebrow">Historical weekly-card method</div><h2>Top two-leg card</h2></div></div>{card_html}</section>
    <section id="board">
      <div class="section-head"><div><div class="eyebrow">Everything, including no-plays</div><h2>Full weekly board</h2></div><p>The site shows why games fail the screen instead of hiding them.</p></div>
      <div class="panel">
        <div class="toolbar"><input id="search" type="search" placeholder="Search matchup, weather, reason…"><select id="statusFilter"><option value="ALL">All statuses</option><option>QUALIFIES</option><option>LEAN</option><option>WATCH</option><option>NO PLAY</option><option>NO LINE</option></select></div>
        <div class="table-wrap"><table><thead><tr><th>Status</th><th>Game</th><th>Market</th><th>NWS kickoff forecast</th><th>Model</th><th>Why</th></tr></thead><tbody id="boardBody">{table_rows(board)}</tbody></table></div>
      </div>
    </section>
  </main>
  <footer class="shell">Historical results do not guarantee future outcomes. The production board remains in paper-tracking mode while live 2026 forecasts and market totals are accumulated and compared with closing lines/results.</footer>
  <script>
    const search = document.getElementById('search');
    const statusFilter = document.getElementById('statusFilter');
    const rows = [...document.querySelectorAll('#boardBody tr')];
    function filterRows() {{
      const q = search.value.toLowerCase(); const s = statusFilter.value;
      rows.forEach(row => {{ const showText = row.innerText.toLowerCase().includes(q); const showStatus = s === 'ALL' || row.dataset.status === s; row.style.display = showText && showStatus ? '' : 'none'; }});
    }}
    search.addEventListener('input', filterRows); statusFilter.addEventListener('change', filterRows);
  </script>
</body>
</html>'''


def main() -> None:
    board = read_csv('outputs/weekly_board.csv')
    card = read_csv('outputs/weekly_card.csv')
    snapshot_path = ROOT / 'outputs/weekly_snapshot.json'
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8')) if snapshot_path.exists() else {}

    docs = ensure_dir('docs')
    outputs = ensure_dir('outputs')
    research = docs / 'research.html'
    current_index = docs / 'index.html'
    if not research.exists() and current_index.exists():
        shutil.copyfile(current_index, research)

    html = dashboard_html(board, card, snapshot)
    current_index.write_text(html, encoding='utf-8')
    (outputs / 'live_dashboard.html').write_text(html, encoding='utf-8')
    print(f'Wrote {current_index} and {outputs / "live_dashboard.html"}')


if __name__ == '__main__':
    main()
