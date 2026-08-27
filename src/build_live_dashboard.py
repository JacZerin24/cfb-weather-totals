from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from html import escape
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .utils import ROOT, ensure_dir

CT = ZoneInfo('America/Chicago')

STATUS_COLORS = {
    'QUALIFIES': '#22c55e',
    'LEAN': '#8b5cf6',
    'WATCH': '#f59e0b',
    'NO PLAY': '#ef4444',
    'NO LINE': '#64748b',
}


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


def text_or(value: Any, fallback: str = '—') -> str:
    try:
        if value is None or pd.isna(value):
            return fallback
    except Exception:
        pass
    text = str(value).strip()
    return text if text and text.lower() != 'nan' else fallback


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        if value is None or pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


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
    if truthy(row.get('game_indoors', False)):
        return 'Indoor / dome'
    nws_status = text_or(row.get('nws_status'), '')
    if not nws_status and str(row.get('status') or '') == 'NO LINE':
        return 'NWS scoring waits until a current market total is available.'
    if nws_status != 'ok':
        return text_or(row.get('weather_summary'), 'NWS forecast unavailable')
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
    return ' · '.join(pieces) if pieces else text_or(row.get('weather_summary'), 'NWS forecast available')


def model_text(row: pd.Series) -> str:
    if pd.isna(row.get('pred_market_residual')):
        return 'Not scored'
    residual = float(row['pred_market_residual'])
    direction = 'UNDER' if residual < 0 else 'OVER'
    return f"{direction} edge {abs(residual):.1f} · projected total {fmt_num(row.get('model_projected_total'))}"


def market_text(row: pd.Series) -> str:
    if pd.isna(row.get('closing_total')):
        return 'No current total'
    provider = text_or(row.get('line_provider'), 'unknown')
    providers = row.get('line_provider_count')
    provider_note = f" · {fmt_int(providers)} providers" if pd.notna(providers) else ''
    return f"O/U {fmt_num(row.get('closing_total'))} · {escape(provider)}{provider_note}"


def target_card(row: pd.Series, rank: int) -> str:
    tags = text_or(row.get('research_tags'), '')
    tags_html = ''.join(f'<span class="tag">{escape(t.strip())}</span>' for t in tags.split(';') if t.strip())
    return f'''
      <article class="target-card">
        <div class="target-head">
          <span class="rank">#{rank}</span>
          <span class="badge qualifies">QUALIFIES</span>
        </div>
        <h3>{escape(text_or(row.get('away_team'), 'Away'))} <span>@</span> {escape(text_or(row.get('home_team'), 'Home'))}</h3>
        <div class="kickoff">{escape(kickoff_ct(row.get('start_date')))} · {escape(text_or(row.get('venue_name'), 'Venue TBD'))}</div>
        <div class="big-pick">UNDER {fmt_num(row.get('closing_total'))}</div>
        <div class="edge">Model edge: <strong>{fmt_num(row.get('abs_pred_edge'))} pts</strong> · model total {fmt_num(row.get('model_projected_total'))}</div>
        <div class="weather">{escape(weather_text(row))}</div>
        <p>{escape(text_or(row.get('decision_reason'), ''))}</p>
        <div class="tags">{tags_html}</div>
      </article>
    '''


def table_rows(board: pd.DataFrame) -> str:
    rows = []
    for _, row in board.iterrows():
        status = text_or(row.get('status'), 'NO PLAY')
        tags = text_or(row.get('research_tags'), '')
        tags_html = '<br><span class="small">' + escape(tags) + '</span>' if tags else ''
        rows.append(f'''
          <tr data-status="{escape(status)}">
            <td><span class="badge {status_class(status)}">{escape(status)}</span></td>
            <td><strong>{escape(text_or(row.get('away_team'), ''))} @ {escape(text_or(row.get('home_team'), ''))}</strong><br><span class="small">{escape(kickoff_ct(row.get('start_date')))}</span></td>
            <td>{market_text(row)}</td>
            <td>{escape(weather_text(row))}</td>
            <td>{escape(model_text(row))}{tags_html}</td>
            <td>{escape(text_or(row.get('decision_reason'), ''))}</td>
          </tr>
        ''')
    return ''.join(rows)


def map_records(board: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if board.empty:
        return records
    for _, row in board.iterrows():
        try:
            lat = float(row.get('venue_latitude'))
            lon = float(row.get('venue_longitude'))
            if pd.isna(lat) or pd.isna(lon):
                continue
        except Exception:
            continue

        city = text_or(row.get('venue_city'), '')
        state = text_or(row.get('venue_state'), '')
        location = ', '.join(v for v in [city, state] if v)
        records.append({
            'game_id': text_or(row.get('game_id'), ''),
            'lat': lat,
            'lon': lon,
            'status': text_or(row.get('status'), 'NO PLAY'),
            'away': text_or(row.get('away_team'), 'Away'),
            'home': text_or(row.get('home_team'), 'Home'),
            'kickoff': kickoff_ct(row.get('start_date')),
            'venue': text_or(row.get('venue_name'), 'Venue TBD'),
            'location': location,
            'market': 'No current total' if pd.isna(row.get('closing_total')) else f"O/U {fmt_num(row.get('closing_total'))} · {text_or(row.get('line_provider'), 'unknown')}",
            'model': model_text(row),
            'weather': weather_text(row),
            'reason': text_or(row.get('decision_reason'), ''),
        })
    return records


def map_filter_buttons(records: list[dict[str, Any]]) -> str:
    counts = {status: 0 for status in STATUS_COLORS}
    for record in records:
        status = str(record.get('status') or '')
        if status in counts:
            counts[status] += 1
    buttons = [
        f'<button type="button" class="map-filter active" data-map-status="ALL"><span class="map-dot all-dot"></span>All <strong>{len(records)}</strong></button>'
    ]
    for status, color in STATUS_COLORS.items():
        buttons.append(
            f'<button type="button" class="map-filter" data-map-status="{escape(status)}">'
            f'<span class="map-dot" style="background:{color}"></span>{escape(status)} <strong>{counts[status]}</strong></button>'
        )
    return ''.join(buttons)


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
        legs = ' + '.join(
            f"UNDER {fmt_num(r.get('closing_total'))} ({escape(text_or(r.get('away_team'), ''))} @ {escape(text_or(r.get('home_team'), ''))})"
            for _, r in two.iterrows()
        )
        card_html = f'<div class="parlay active-card"><span class="eyebrow">Top weekly 2-leg research card</span><h3>{legs}</h3><p>Built only from the two highest-edge qualifying high-total HGB under targets. No additional combinations are forced.</p></div>'
    else:
        card_html = '<div class="parlay"><span class="eyebrow">Top weekly 2-leg research card</span><h3>No card</h3><p>Fewer than two games currently qualify, so the model does not force a parlay.</p></div>'

    map_data = map_records(board)
    map_json = json.dumps(map_data, ensure_ascii=False).replace('<', '\\u003c')
    missing_coords = max(0, len(board) - len(map_data))
    missing_note = (
        f'{missing_coords} scheduled game(s) lack usable venue coordinates and remain available in the full board below.'
        if missing_coords else
        'Every scheduled game currently has usable venue coordinates.'
    )

    week = snapshot.get('week', '—')
    season = snapshot.get('season', '—')
    html = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>CFB Weather Totals · Live Board</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root { --bg:#08101d; --panel:#111c31; --panel2:#17243d; --text:#f1f6ff; --muted:#9fb0ce; --line:rgba(255,255,255,.11); --cyan:#7dd3fc; --green:#22c55e; --yellow:#f59e0b; --red:#ef4444; --purple:#8b5cf6; --gray:#64748b; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--text); background:radial-gradient(circle at 15% 0%,rgba(125,211,252,.17),transparent 28%),linear-gradient(145deg,#060b14,#0b1424 46%,#0e1728); line-height:1.45; }
    a { color:var(--cyan); } .shell { width:min(1380px,calc(100vw - 28px)); margin:0 auto; }
    header { padding:36px 0 18px; } .hero { display:grid; grid-template-columns:1.4fr .6fr; gap:18px; }
    .panel,.hero-card,.metric,.target-card,.parlay,.empty,.map-panel { border:1px solid var(--line); background:linear-gradient(180deg,rgba(20,32,55,.96),rgba(13,24,43,.94)); border-radius:20px; box-shadow:0 18px 44px rgba(0,0,0,.28); }
    .hero-card { padding:28px; } .eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:12px; font-weight:900; color:#a7f3d0; }
    h1 { font-size:clamp(35px,5vw,62px); line-height:.96; margin:10px 0 14px; } h2 { margin:0 0 14px; font-size:26px; } h3 { margin:0; }
    p,.muted,.small { color:var(--muted); } .hero-card p { max-width:850px; margin:0; }
    .hero-side { display:grid; gap:10px; } .hero-side div { padding:13px 15px; background:rgba(255,255,255,.045); border:1px solid var(--line); border-radius:14px; }
    .metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0; } .metric { padding:16px; } .metric .label { color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.09em; } .metric .value { font-size:29px; font-weight:900; margin-top:4px; }
    nav { position:sticky; top:0; z-index:1000; backdrop-filter:blur(14px); background:rgba(6,11,20,.84); border-block:1px solid var(--line); } nav .shell { display:flex; gap:18px; align-items:center; padding:11px 0; overflow:auto; } nav a { text-decoration:none; font-weight:800; white-space:nowrap; }
    main { padding:22px 0 54px; } section { margin-bottom:24px; } .section-head { display:flex; justify-content:space-between; gap:14px; align-items:end; margin-bottom:12px; } .section-head p { margin:0; }
    .map-panel { overflow:hidden; } .map-toolbar { padding:14px; display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; border-bottom:1px solid var(--line); }
    .map-filters { display:flex; gap:8px; flex-wrap:wrap; } .map-filter { min-height:38px; border:1px solid var(--line); background:#0d1728; color:var(--text); padding:7px 10px; border-radius:999px; cursor:pointer; font-weight:800; display:inline-flex; align-items:center; gap:6px; }
    .map-filter:hover,.map-filter:focus-visible { border-color:rgba(125,211,252,.55); } .map-filter.active { box-shadow:0 0 0 2px rgba(125,211,252,.28) inset; border-color:rgba(125,211,252,.7); }
    .map-dot { width:10px; height:10px; border-radius:999px; display:inline-block; flex:0 0 auto; } .all-dot { background:linear-gradient(135deg,var(--green) 0 20%,var(--purple) 20% 40%,var(--yellow) 40% 60%,var(--red) 60% 80%,var(--gray) 80%); }
    .map-count { color:var(--muted); font-size:12px; font-weight:800; } #weekMap { width:100%; height:clamp(430px,55vw,680px); background:#dbe7ef; } .map-note { padding:10px 14px 13px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); }
    .leaflet-popup-content-wrapper,.leaflet-popup-tip { background:#101b30; color:#eef6ff; } .leaflet-popup-content { margin:14px; min-width:245px; max-width:320px; } .leaflet-container a.leaflet-popup-close-button { color:#cbd5e1; }
    .map-popup .popup-status { font-size:10px; font-weight:950; letter-spacing:.08em; margin-bottom:5px; } .map-popup h3 { font-size:17px; line-height:1.2; margin:0 0 5px; } .map-popup .popup-sub { color:#aebed8; font-size:11px; margin-bottom:9px; } .map-popup .popup-line { padding:5px 0; border-top:1px solid rgba(255,255,255,.09); font-size:12px; } .map-popup .popup-reason { color:#cbd5e1; margin-top:7px; font-size:11px; }
    .target-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; } .target-card { padding:20px; } .target-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; } .rank { font-weight:900; color:var(--cyan); }
    .target-card h3 { font-size:23px; } .target-card h3 span { color:var(--muted); font-weight:500; } .kickoff { color:var(--muted); font-size:13px; margin-top:5px; } .big-pick { font-size:31px; font-weight:950; margin:18px 0 3px; } .edge { color:#cfe3ff; } .weather { margin-top:12px; padding:11px 12px; background:rgba(125,211,252,.07); border:1px solid rgba(125,211,252,.18); border-radius:12px; } .tags { display:flex; flex-wrap:wrap; gap:7px; } .tag { font-size:11px; font-weight:800; color:#dbeafe; border:1px solid var(--line); border-radius:999px; padding:5px 8px; background:rgba(255,255,255,.045); }
    .badge { display:inline-block; font-size:11px; font-weight:950; letter-spacing:.05em; border-radius:999px; padding:6px 9px; } .qualifies { color:#bbf7d0; background:rgba(34,197,94,.13); border:1px solid rgba(34,197,94,.35); } .lean { color:#ddd6fe; background:rgba(139,92,246,.13); border:1px solid rgba(139,92,246,.35); } .watch { color:#fde68a; background:rgba(245,158,11,.13); border:1px solid rgba(245,158,11,.35); } .no-play { color:#fecaca; background:rgba(239,68,68,.11); border:1px solid rgba(239,68,68,.3); } .no-line { color:#dbe4ef; background:rgba(100,116,139,.18); border:1px solid rgba(148,163,184,.3); }
    .parlay { padding:20px; } .parlay h3 { margin-top:7px; font-size:20px; } .parlay p { margin-bottom:0; } .active-card { border-color:rgba(34,197,94,.35); } .empty { padding:24px; color:var(--muted); }
    .panel { overflow:hidden; } .toolbar { display:flex; gap:8px; flex-wrap:wrap; padding:14px; border-bottom:1px solid var(--line); } input,select { background:#0d1728; border:1px solid var(--line); color:var(--text); padding:9px 11px; border-radius:10px; }
    .table-wrap { overflow:auto; } table { width:100%; border-collapse:collapse; min-width:1180px; } th,td { padding:11px 12px; border-bottom:1px solid rgba(255,255,255,.07); text-align:left; vertical-align:top; font-size:12px; } th { color:#dbeafe; background:#101b30; position:sticky; top:0; } td { color:#dce7f7; } tr:hover td { background:rgba(125,211,252,.045); }
    footer { color:var(--muted); font-size:12px; padding:0 0 38px; }
    @media(max-width:900px) { .hero,.metrics,.target-grid { grid-template-columns:1fr; } .section-head { align-items:start; flex-direction:column; } #weekMap { height:500px; } }
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
        <div><span class="eyebrow">Slate</span><br><strong>__SEASON__ · Week __WEEK__</strong></div>
        <div><span class="eyebrow">Updated</span><br><strong>__GENERATED__</strong></div>
        <div><span class="eyebrow">Mode</span><br><strong>Research / paper tracking</strong></div>
      </div>
    </div>
    <div class="metrics">
      <div class="metric"><div class="label">Games scheduled</div><div class="value">__GAMES_COUNT__</div></div>
      <div class="metric"><div class="label">Current totals</div><div class="value">__TOTAL_COUNT__</div></div>
      <div class="metric"><div class="label">Qualifying targets</div><div class="value">__TARGET_COUNT__</div></div>
      <div class="metric"><div class="label">NWS-ready outdoor</div><div class="value">__NWS_COUNT__</div></div>
    </div>
  </header>
  <nav><div class="shell"><a href="#map">Week map</a><a href="#targets">Targets</a><a href="#card">2-leg card</a><a href="#board">Full board</a><a href="research.html">Historical research →</a></div></nav>
  <main class="shell">
    <section id="map">
      <div class="section-head"><div><div class="eyebrow">Geographic slate overview</div><h2>Weekly status map</h2></div><p>Every scheduled game stays visible, including games without a current total.</p></div>
      <div class="map-panel">
        <div class="map-toolbar"><div class="map-filters">__MAP_BUTTONS__</div><div class="map-count" id="mapCount">Showing __MAPPED_COUNT__ mapped games</div></div>
        <div id="weekMap" role="region" aria-label="Interactive map of this week's college football games by model status"></div>
        <div class="map-note">Marker colors: green qualifies · purple lean · amber watch · red no play · gray no line. __MISSING_NOTE__</div>
      </div>
    </section>
    <section id="targets">
      <div class="section-head"><div><div class="eyebrow">Selective by design</div><h2>Qualifying under targets</h2></div><p>HGB under edge ≥3.5 + market total ≥56 + usable kickoff forecast/time.</p></div>
      <div class="target-grid">__TARGET_HTML__</div>
    </section>
    <section id="card"><div class="section-head"><div><div class="eyebrow">Historical weekly-card method</div><h2>Top two-leg card</h2></div></div>__CARD_HTML__</section>
    <section id="board">
      <div class="section-head"><div><div class="eyebrow">Everything, including no-plays and no-lines</div><h2>Full weekly board</h2></div><p>The site shows why games fail the screen instead of hiding them.</p></div>
      <div class="panel">
        <div class="toolbar"><input id="search" type="search" placeholder="Search matchup, weather, reason…"><select id="statusFilter"><option value="ALL">All statuses</option><option>QUALIFIES</option><option>LEAN</option><option>WATCH</option><option>NO PLAY</option><option>NO LINE</option></select></div>
        <div class="table-wrap"><table><thead><tr><th>Status</th><th>Game</th><th>Market</th><th>NWS kickoff forecast</th><th>Model</th><th>Why</th></tr></thead><tbody id="boardBody">__TABLE_ROWS__</tbody></table></div>
      </div>
    </section>
  </main>
  <footer class="shell">Historical results do not guarantee future outcomes. The production board remains in paper-tracking mode while live 2026 forecasts and market totals are accumulated and compared with closing lines/results.</footer>
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const gamePoints = __MAP_DATA__;
    const statusColors = { 'QUALIFIES':'#22c55e', 'LEAN':'#8b5cf6', 'WATCH':'#f59e0b', 'NO PLAY':'#ef4444', 'NO LINE':'#64748b' };

    function escapeHtml(value) {
      return String(value == null ? '' : value).replace(/[&<>'"]/g, function(ch) {
        return { '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;' }[ch];
      });
    }

    const mapElement = document.getElementById('weekMap');
    const mapCount = document.getElementById('mapCount');
    const mapFilters = [...document.querySelectorAll('.map-filter')];
    const mapMarkers = [];

    if (window.L && mapElement) {
      const weekMap = L.map('weekMap', { scrollWheelZoom:false, preferCanvas:true });
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom:19,
        attribution:'&copy; OpenStreetMap contributors'
      }).addTo(weekMap);

      gamePoints.forEach(function(game) {
        const color = statusColors[game.status] || '#64748b';
        const marker = L.circleMarker([game.lat, game.lon], {
          radius:8,
          color:'#f8fafc',
          weight:1.5,
          fillColor:color,
          fillOpacity:0.93
        });
        const place = game.location ? game.venue + ' · ' + game.location : game.venue;
        const popup = '<div class="map-popup">' +
          '<div class="popup-status" style="color:' + color + '">' + escapeHtml(game.status) + '</div>' +
          '<h3>' + escapeHtml(game.away) + ' @ ' + escapeHtml(game.home) + '</h3>' +
          '<div class="popup-sub">' + escapeHtml(game.kickoff) + '<br>' + escapeHtml(place) + '</div>' +
          '<div class="popup-line"><strong>Market:</strong> ' + escapeHtml(game.market) + '</div>' +
          '<div class="popup-line"><strong>Model:</strong> ' + escapeHtml(game.model) + '</div>' +
          '<div class="popup-line"><strong>Weather:</strong> ' + escapeHtml(game.weather) + '</div>' +
          '<div class="popup-reason">' + escapeHtml(game.reason) + '</div>' +
          '</div>';
        marker.bindPopup(popup, { maxWidth:340 });
        marker.bindTooltip(escapeHtml(game.away + ' @ ' + game.home), { direction:'top' });
        marker.addTo(weekMap);
        mapMarkers.push({ marker:marker, status:game.status });
      });

      if (mapMarkers.length) {
        const bounds = L.latLngBounds(mapMarkers.map(function(item) { return item.marker.getLatLng(); }));
        weekMap.fitBounds(bounds.pad(0.08), { maxZoom:6 });
      } else {
        weekMap.setView([38.5, -97], 4);
      }

      function filterMap(status) {
        let shown = 0;
        mapMarkers.forEach(function(item) {
          const visible = status === 'ALL' || item.status === status;
          if (visible && !weekMap.hasLayer(item.marker)) item.marker.addTo(weekMap);
          if (!visible && weekMap.hasLayer(item.marker)) weekMap.removeLayer(item.marker);
          if (visible) shown += 1;
        });
        mapCount.textContent = 'Showing ' + shown + ' mapped game' + (shown === 1 ? '' : 's');
      }

      mapFilters.forEach(function(button) {
        button.addEventListener('click', function() {
          mapFilters.forEach(function(other) { other.classList.remove('active'); });
          button.classList.add('active');
          filterMap(button.dataset.mapStatus || 'ALL');
        });
      });
    } else if (mapElement) {
      mapElement.innerHTML = '<div style="padding:28px;color:#334155">The interactive map library could not load. The complete weekly board is still available below.</div>';
    }

    const search = document.getElementById('search');
    const statusFilter = document.getElementById('statusFilter');
    const rows = [...document.querySelectorAll('#boardBody tr')];
    function filterRows() {
      const q = search.value.toLowerCase();
      const s = statusFilter.value;
      rows.forEach(function(row) {
        const showText = row.innerText.toLowerCase().includes(q);
        const showStatus = s === 'ALL' || row.dataset.status === s;
        row.style.display = showText && showStatus ? '' : 'none';
      });
    }
    search.addEventListener('input', filterRows);
    statusFilter.addEventListener('change', filterRows);
  </script>
</body>
</html>'''

    replacements = {
        '__SEASON__': escape(str(season)),
        '__WEEK__': escape(str(week)),
        '__GENERATED__': escape(generated),
        '__GAMES_COUNT__': str(len(board)),
        '__TOTAL_COUNT__': str(int(board['closing_total'].notna().sum()) if 'closing_total' in board.columns else 0),
        '__TARGET_COUNT__': str(len(targets)),
        '__NWS_COUNT__': str(int(board['nws_status'].eq('ok').sum()) if 'nws_status' in board.columns else 0),
        '__MAPPED_COUNT__': str(len(map_data)),
        '__MISSING_NOTE__': escape(missing_note),
        '__MAP_BUTTONS__': map_filter_buttons(map_data),
        '__TARGET_HTML__': target_html,
        '__CARD_HTML__': card_html,
        '__TABLE_ROWS__': table_rows(board),
        '__MAP_DATA__': map_json,
    }
    for token, value in replacements.items():
        html = html.replace(token, value)
    return html


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
