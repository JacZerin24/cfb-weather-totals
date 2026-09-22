from __future__ import annotations

import json
from html import escape
from typing import Any

import pandas as pd


MARKER = '/* NFL_SITE_PARITY */'

STATUS_CLASS = {
    'STRONG': 'strong',
    'QUALIFIES': 'qualifies',
    'NO PLAY': 'no-play',
    'EARLY LOOK': 'early',
    'PAST WINDOW': 'past',
    'WAITING': 'waiting',
    'NO LINE': 'no-line',
}


def _text(value: Any, fallback: str = '—') -> str:
    try:
        if value is None or pd.isna(value):
            return fallback
    except Exception:
        pass
    value = str(value).strip()
    return value if value and value.lower() not in {'nan', 'none'} else fallback


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        out = float(value)
        return out if pd.notna(out) else None
    except Exception:
        return None


def _fmt(value: Any, decimals: int = 1, suffix: str = '') -> str:
    number = _num(value)
    if number is None:
        return '—'
    return f'{number:.{decimals}f}{suffix}'


def _kickoff(value: Any) -> str:
    try:
        ts = pd.to_datetime(value, utc=True)
        return ts.tz_convert('America/Chicago').strftime('%a %b %-d · %-I:%M %p CT')
    except Exception:
        return 'Kickoff TBD'


def _sort_time(value: Any) -> str:
    try:
        return f'{pd.to_datetime(value, utc=True).timestamp():.0f}'
    except Exception:
        return ''


def weather_types(row: pd.Series) -> list[str]:
    """Weather tags supported by the frozen NFL decision-time dataset."""
    roof = _text(row.get('decision_roof'), '').lower()
    if roof in {'dome', 'retractable'}:
        return ['INDOOR']

    tags: list[str] = []
    temp = _num(row.get('forecast_temp_24h'))
    wind = _num(row.get('forecast_wind_24h'))
    if wind is not None and wind >= 15:
        tags.append('WIND')
    if temp is not None and temp >= 85:
        tags.append('HOT')
    if temp is not None and temp <= 40:
        tags.append('COLD')
    return tags


def _market(row: pd.Series) -> str:
    total = _num(row.get('closing_total'))
    if total is None:
        return 'No current total'
    price = _num(row.get('consensus_over_price'))
    book = _text(row.get('consensus_over_sportsbook'), '')
    price_text = ''
    if price is not None:
        price_int = int(round(price))
        price_text = f' · {price_int:+d}' if price_int > 0 else f' · {price_int:d}'
    book_text = f' · {book}' if book else ''
    return f'OVER {total:.1f}{price_text}{book_text}'


def _best_price(row: pd.Series) -> str:
    total = _num(row.get('closing_total'))
    price = _num(row.get('best_over_price_same_line'))
    book = _text(row.get('best_over_sportsbook_same_line'), '')
    if total is None or price is None:
        return '—'
    price_int = int(round(price))
    price_text = f'{price_int:+d}' if price_int > 0 else f'{price_int:d}'
    return f'{total:.1f} {price_text}' + (f' · {book}' if book else '')


def _weather(row: pd.Series) -> str:
    roof = _text(row.get('decision_roof'), 'unknown').lower()
    if roof == 'dome':
        return 'Dome · weather withheld'
    if roof == 'retractable':
        return 'Retractable roof · weather withheld'
    if roof != 'outdoors':
        return 'Weather unavailable'
    temp = _fmt(row.get('forecast_temp_24h'), 0, '°F')
    wind = _fmt(row.get('forecast_wind_24h'), 0, ' mph')
    if temp == '—' and wind == '—':
        return '24h JMA unavailable'
    return f'24h JMA · {temp} · wind {wind}'


def _model(row: pd.Series) -> str:
    edge = _num(row.get('pred_market_residual'))
    projected = _num(row.get('model_projected_total'))
    prob = _num(row.get('over_probability'))
    if edge is None:
        return 'Not scored'
    parts = [f'OVER edge {edge:+.1f}']
    if projected is not None:
        parts.append(f'model {projected:.1f}')
    if prob is not None:
        parts.append(f'P(OVER) {100 * prob:.1f}%')
    return ' · '.join(parts)


def _badge(value: str) -> str:
    css = STATUS_CLASS.get(value, 'no-play')
    return f'<span class="badge {css}">{escape(value)}</span>'


def _map_points(board: pd.DataFrame) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for _, row in board.iterrows():
        lat = _num(row.get('lat'))
        lon = _num(row.get('lon'))
        if lat is None or lon is None:
            continue
        points.append({
            'game_id': _text(row.get('game_id'), ''),
            'lat': lat,
            'lon': lon,
            'status': _text(row.get('status'), 'NO PLAY'),
            'signal': _text(row.get('model_signal'), 'NO PLAY'),
            'state': _text(row.get('decision_state'), ''),
            'away': _text(row.get('away_team'), 'Away'),
            'home': _text(row.get('home_team'), 'Home'),
            'kickoff': _kickoff(row.get('kickoff_utc')),
            'venue': _text(row.get('stadium'), 'Venue TBD'),
            'market': _market(row),
            'best': _best_price(row),
            'model': _model(row),
            'weather': _weather(row),
            'roof': _text(row.get('decision_roof'), 'unknown'),
            'weather_types': weather_types(row),
        })
    return points


def _map_buttons(points: list[dict[str, Any]]) -> str:
    order = ['STRONG', 'QUALIFIES', 'NO PLAY']
    counts: dict[str, int] = {}
    for point in points:
        signal = str(point.get('signal') or 'NO PLAY')
        counts[signal] = counts.get(signal, 0) + 1
    buttons = [
        f'<button type="button" class="map-filter active" data-map-signal="ALL">'
        f'<span class="map-dot all-dot"></span>All <strong>{len(points)}</strong></button>'
    ]
    for signal in order:
        count = counts.get(signal, 0)
        if not count:
            continue
        css = STATUS_CLASS.get(signal, 'no-play')
        buttons.append(
            f'<button type="button" class="map-filter" data-map-signal="{escape(signal)}">'
            f'<span class="map-dot {css}-dot"></span>{escape(signal)} <strong>{count}</strong></button>'
        )
    return ''.join(buttons)


def _table_rows(board: pd.DataFrame) -> str:
    rows: list[str] = []
    ordered = board.copy()
    if 'kickoff_utc' in ordered.columns:
        ordered['_sort_kickoff'] = pd.to_datetime(ordered['kickoff_utc'], utc=True, errors='coerce')
        ordered = ordered.sort_values('_sort_kickoff')

    for _, row in ordered.iterrows():
        status = _text(row.get('status'), 'NO PLAY')
        signal = _text(row.get('model_signal'), 'NO PLAY')
        tags = ' '.join(weather_types(row))
        matchup = f'{_text(row.get("away_team"), "")} @ {_text(row.get("home_team"), "")}'
        kickoff_sort = _sort_time(row.get('kickoff_utc'))
        temp = _num(row.get('forecast_temp_24h'))
        wind = _num(row.get('forecast_wind_24h'))
        edge = _num(row.get('pred_market_residual'))
        prob = _num(row.get('over_probability'))
        hours = _num(row.get('hours_to_kickoff'))
        total = _num(row.get('closing_total'))
        roof = _text(row.get('decision_roof'), '—').replace('_', ' ').title()
        state = _text(row.get('decision_state'), '—').replace('_', ' ')

        def num_td(value: float | None, display: str) -> str:
            sort = '' if value is None else f'{value:.6f}'
            return f'<td class="nfl-num" data-sort-value="{sort}">{escape(display)}</td>'

        rows.append(''.join([
            f'<tr data-status="{escape(status)}" data-signal="{escape(signal)}" data-weather-types="{escape(tags)}">',
            f'<td>{_badge(status)}</td>',
            f'<td>{_badge(signal)}</td>',
            f'<td><strong>{escape(matchup)}</strong><br><small>{escape(_text(row.get("stadium"), "Venue TBD"))}</small></td>',
            f'<td class="kickoff-num" data-sort-value="{kickoff_sort}">{escape(_kickoff(row.get("kickoff_utc")))}</td>',
            num_td(total, _market(row)),
            f'<td>{escape(_best_price(row))}</td>',
            num_td(temp, '—' if temp is None else f'{temp:.0f}°F'),
            num_td(wind, '—' if wind is None else f'{wind:.0f} mph'),
            f'<td>{escape(roof)}</td>',
            num_td(edge, '—' if edge is None else f'{edge:+.1f}'),
            num_td(prob, '—' if prob is None else f'{100 * prob:.1f}%'),
            f'<td>{escape(state)}</td>',
            num_td(hours, '—' if hours is None else f'{hours:.1f}h'),
            '</tr>',
        ]))
    if rows:
        return ''.join(rows)
    return '<tr><td colspan="13" class="empty-cell">No active NFL games are available.</td></tr>'


def _map_section(board: pd.DataFrame) -> tuple[str, str]:
    points = _map_points(board)
    payload = json.dumps(points, ensure_ascii=False).replace('<', '\\u003c')
    missing = max(0, len(board) - len(points))
    note = (
        f'{missing} game(s) lack usable venue coordinates and remain in the table.'
        if missing else
        'Every game in the current slate has usable venue coordinates.'
    )
    html = f'''
    <section id="map">
      <div class="section-head">
        <div><div class="eyebrow">Geographic slate overview</div><h2>Weekly signal map</h2></div>
        <p>Marker colors follow the current model signal; live decision status and frozen-window state remain visible in each popup.</p>
      </div>
      <div class="map-panel">
        <div class="map-toolbar">
          <div class="map-filters">{_map_buttons(points)}</div>
          <div class="map-controls">
            <button type="button" class="map-filter radar-toggle" id="radarToggle" aria-pressed="false">Radar: Off</button>
            <div class="map-count" id="mapCount">Showing {len(points)} mapped games</div>
          </div>
        </div>
        <div id="weekMap" role="region" aria-label="Interactive map of this week's NFL games by model signal"></div>
        <div class="map-note">Marker colors: lime strong · green qualifies · red no play. {escape(note)} Radar overlay: IEM / NWS CONUS NEXRAD base reflectivity. <span id="radarStatus">Radar off.</span></div>
      </div>
    </section>
    '''
    return html, payload


def _table_section(board: pd.DataFrame) -> str:
    return f'''
    <section id="board">
      <div class="section-head">
        <div><div class="eyebrow">Everything in one sortable view</div><h2>Full weekly table</h2></div>
        <p>Search and filter the entire slate without hiding NO PLAY, early-look, waiting, or roof-managed games.</p>
      </div>
      <div class="panel nfl-board-panel">
        <div class="toolbar">
          <input id="boardSearch" type="search" placeholder="Search matchup, venue, market, status…">
          <select id="boardStatusFilter">
            <option value="ALL">All live statuses</option>
            <option>STRONG</option><option>QUALIFIES</option><option>NO PLAY</option>
            <option>EARLY LOOK</option><option>PAST WINDOW</option><option>WAITING</option><option>NO LINE</option>
          </select>
          <select id="boardSignalFilter">
            <option value="ALL">All model signals</option>
            <option>STRONG</option><option>QUALIFIES</option><option>NO PLAY</option>
          </select>
          <select id="weatherFilter" title="Filter by fixed-lead weather / roof type">
            <option value="ALL">All weather</option>
            <option value="WIND">Windy (15+ mph)</option>
            <option value="HOT">Hot (85°F+)</option>
            <option value="COLD">Cold (40°F or lower)</option>
            <option value="INDOOR">Dome / retractable roof</option>
          </select>
        </div>
        <div class="table-wrap">
          <table id="nflBoardTable">
            <thead><tr>
              <th>Live status</th>
              <th>Model signal</th>
              <th>Game</th>
              <th><button class="sort-btn" type="button" data-nfl-sort="3">Kickoff <span>↕</span></button></th>
              <th><button class="sort-btn" type="button" data-nfl-sort="4">Market <span>↕</span></button></th>
              <th>Best price</th>
              <th><button class="sort-btn" type="button" data-nfl-sort="6">Temp <span>↕</span></button></th>
              <th><button class="sort-btn" type="button" data-nfl-sort="7">Wind <span>↕</span></button></th>
              <th>Roof</th>
              <th><button class="sort-btn" type="button" data-nfl-sort="9">Edge <span>↕</span></button></th>
              <th><button class="sort-btn" type="button" data-nfl-sort="10">P(OVER) <span>↕</span></button></th>
              <th>Decision state</th>
              <th><button class="sort-btn" type="button" data-nfl-sort="12">Hours to kick <span>↕</span></button></th>
            </tr></thead>
            <tbody id="nflBoardBody">{_table_rows(board)}</tbody>
          </table>
        </div>
      </div>
    </section>
    '''


CSS = r'''
    /* NFL_SITE_PARITY */
    .map-panel {
      overflow:hidden; border:1px solid var(--line);
      background:linear-gradient(180deg,rgba(17,30,50,.97),rgba(10,20,35,.96));
      border-radius:20px; box-shadow:0 18px 44px rgba(0,0,0,.24);
    }
    .map-toolbar {
      padding:14px; display:flex; justify-content:space-between; align-items:center;
      gap:12px; flex-wrap:wrap; border-bottom:1px solid var(--line);
    }
    .map-filters,.map-controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .map-filter {
      min-height:38px; border:1px solid var(--line); background:#0d1728;
      color:var(--text); padding:7px 10px; border-radius:999px; cursor:pointer;
      font-weight:850; display:inline-flex; align-items:center; gap:6px;
    }
    .map-filter:hover,.map-filter:focus-visible { border-color:rgba(103,232,249,.58); }
    .map-filter.active { box-shadow:0 0 0 2px rgba(103,232,249,.20) inset; border-color:rgba(103,232,249,.78); }
    .map-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
    .all-dot { background:linear-gradient(135deg,var(--lime) 0 33%,var(--green) 33% 66%,var(--red) 66%); }
    .strong-dot { background:var(--lime); } .qualifies-dot { background:var(--green); } .no-play-dot { background:var(--red); }
    .map-count,.map-note { color:var(--muted); font-size:12px; font-weight:750; }
    #weekMap { width:100%; height:clamp(430px,55vw,680px); background:#dbe7ef; }
    .map-note { padding:10px 14px 13px; border-top:1px solid var(--line); }
    #radarStatus { color:#bfdbfe; font-weight:800; }
    .radar-toggle { border-color:rgba(103,232,249,.40); }
    .radar-toggle.active { background:rgba(103,232,249,.14); border-color:rgba(103,232,249,.80); }
    .leaflet-popup-content-wrapper,.leaflet-popup-tip { background:#101b30; color:#eef6ff; }
    .leaflet-popup-content { margin:14px; min-width:250px; max-width:340px; }
    .leaflet-container a.leaflet-popup-close-button { color:#cbd5e1; }
    .map-popup .popup-status { font-size:10px; font-weight:950; letter-spacing:.07em; margin-bottom:5px; }
    .map-popup h3 { font-size:17px; line-height:1.2; margin:0 0 5px; }
    .map-popup .popup-sub { color:#aebed8; font-size:11px; margin-bottom:9px; }
    .map-popup .popup-line { padding:5px 0; border-top:1px solid rgba(255,255,255,.09); font-size:12px; }
    .sort-btn {
      appearance:none; border:0; background:transparent; color:#dbeafe; font:inherit;
      font-weight:900; padding:0; cursor:pointer; display:inline-flex; align-items:center; gap:5px;
    }
    .sort-btn:hover,.sort-btn:focus-visible { color:var(--cyan); }
    .sort-btn span { color:var(--muted); font-size:10px; }
    .nfl-num,.kickoff-num { white-space:nowrap; font-variant-numeric:tabular-nums; }
    .nfl-board-panel .toolbar { margin-bottom:0; border:0; border-bottom:1px solid var(--line); border-radius:0; }
    #nflBoardTable { min-width:1500px; }
    #nflBoardTable th { top:0; z-index:1; }
    @media(max-width:900px) { #weekMap { height:500px; } }
'''


JS_TEMPLATE = r'''
  <script>
    // NFL_SITE_PARITY
    const nflGamePoints = __NFL_MAP_DATA__;
    const nflSignalColors = {
      'STRONG':'#a3e635',
      'QUALIFIES':'#22c55e',
      'NO PLAY':'#ef4444'
    };

    function nflEscapeHtml(value) {
      return String(value == null ? '' : value).replace(/[&<>'"]/g, function(ch) {
        return { '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;' }[ch];
      });
    }

    const nflMapElement = document.getElementById('weekMap');
    const nflMapCount = document.getElementById('mapCount');
    const nflMapFilters = [...document.querySelectorAll('.map-filter[data-map-signal]')];
    const nflMapMarkers = [];

    if (window.L && nflMapElement) {
      const weekMap = L.map('weekMap', { scrollWheelZoom:true, preferCanvas:true });
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom:19,
        attribution:'&copy; OpenStreetMap contributors'
      }).addTo(weekMap);

      const radarTileUrl = function(validKey) {
        return 'https://mesonet.agron.iastate.edu/c/tile.py/1.0.0/ridge::USCOMP-N0Q-' + validKey + '/{z}/{x}/{y}.png';
      };
      const radarMetaUrl = 'https://mesonet.agron.iastate.edu/data/gis/images/4326/USCOMP/n0q_0.json';
      const radarToggle = document.getElementById('radarToggle');
      const radarStatus = document.getElementById('radarStatus');
      let radarLayer = null;
      let radarValid = null;
      let radarEnabled = false;
      let radarRefreshTimer = null;

      if (!weekMap.getPane('radarPane')) {
        weekMap.createPane('radarPane');
        weekMap.getPane('radarPane').style.zIndex = 250;
        weekMap.getPane('radarPane').style.pointerEvents = 'none';
      }

      function radarKeyFromIso(valid) {
        const dt = new Date(valid);
        if (Number.isNaN(dt.getTime())) return null;
        const y = dt.getUTCFullYear();
        const mo = String(dt.getUTCMonth() + 1).padStart(2, '0');
        const d = String(dt.getUTCDate()).padStart(2, '0');
        const h = String(dt.getUTCHours()).padStart(2, '0');
        const m = String(dt.getUTCMinutes()).padStart(2, '0');
        return '' + y + mo + d + h + m;
      }

      function fallbackRadarValid() {
        const dt = new Date(Date.now() - 10 * 60 * 1000);
        dt.setUTCMinutes(Math.floor(dt.getUTCMinutes() / 5) * 5, 0, 0);
        return dt.toISOString();
      }

      function formatRadarValid(valid, approximate) {
        const dt = new Date(valid);
        if (Number.isNaN(dt.getTime())) return approximate ? 'Radar time approximate.' : 'Radar time unavailable.';
        const label = new Intl.DateTimeFormat('en-US', {
          timeZone:'America/Chicago',
          month:'short', day:'numeric', hour:'numeric', minute:'2-digit',
          timeZoneName:'short'
        }).format(dt);
        return 'Radar valid ' + label + (approximate ? ' (approx.)' : '') + '.';
      }

      async function latestRadarValid() {
        try {
          const response = await fetch(radarMetaUrl + '?_=' + Date.now(), { cache:'no-store' });
          if (!response.ok) throw new Error('Radar metadata HTTP ' + response.status);
          const payload = await response.json();
          const valid = payload && payload.meta && payload.meta.valid;
          if (!valid || !radarKeyFromIso(valid)) throw new Error('Radar metadata missing valid time');
          return { valid:valid, approximate:false };
        } catch (error) {
          console.warn('Using conservative radar-time fallback:', error);
          return { valid:fallbackRadarValid(), approximate:true };
        }
      }

      function makeRadarLayer(valid) {
        return L.tileLayer(radarTileUrl(radarKeyFromIso(valid)), {
          opacity:0.58,
          pane:'radarPane',
          maxNativeZoom:12,
          maxZoom:19,
          noWrap:true,
          updateWhenIdle:true,
          keepBuffer:2,
          bounds:[[20, -130], [55, -60]],
          attribution:'NEXRAD: Iowa Environmental Mesonet / NWS'
        });
      }

      async function refreshRadar() {
        if (!radarEnabled) return;
        if (radarStatus) radarStatus.textContent = 'Checking latest radar scan…';
        const latest = await latestRadarValid();
        if (!radarEnabled) return;
        const latestKey = radarKeyFromIso(latest.valid);
        const currentKey = radarValid ? radarKeyFromIso(radarValid) : null;
        if (radarLayer && latestKey === currentKey) {
          if (!weekMap.hasLayer(radarLayer)) radarLayer.addTo(weekMap);
          if (radarStatus) radarStatus.textContent = formatRadarValid(latest.valid, latest.approximate);
          return;
        }
        const nextLayer = makeRadarLayer(latest.valid);
        const previousLayer = radarLayer;
        let swapped = false;
        function finishSwap() {
          if (swapped) return;
          swapped = true;
          if (!radarEnabled) {
            if (weekMap.hasLayer(nextLayer)) weekMap.removeLayer(nextLayer);
            return;
          }
          radarLayer = nextLayer;
          radarValid = latest.valid;
          if (previousLayer && previousLayer !== nextLayer && weekMap.hasLayer(previousLayer)) {
            weekMap.removeLayer(previousLayer);
          }
          if (radarStatus) radarStatus.textContent = formatRadarValid(latest.valid, latest.approximate);
        }
        nextLayer.once('load', finishSwap);
        nextLayer.addTo(weekMap);
        window.setTimeout(finishSwap, 3500);
      }

      function setRadarEnabled(enabled) {
        radarEnabled = enabled;
        try { localStorage.setItem('nflRadarEnabled', enabled ? '1' : '0'); } catch (e) {}
        if (radarToggle) {
          radarToggle.classList.toggle('active', enabled);
          radarToggle.setAttribute('aria-pressed', enabled ? 'true' : 'false');
          radarToggle.textContent = enabled ? 'Radar: On' : 'Radar: Off';
        }
        if (radarRefreshTimer) {
          window.clearInterval(radarRefreshTimer);
          radarRefreshTimer = null;
        }
        if (!enabled) {
          if (radarLayer && weekMap.hasLayer(radarLayer)) weekMap.removeLayer(radarLayer);
          if (radarStatus) radarStatus.textContent = 'Radar off.';
          return;
        }
        refreshRadar();
        radarRefreshTimer = window.setInterval(refreshRadar, 5 * 60 * 1000);
      }

      if (radarToggle) {
        radarToggle.addEventListener('click', function() {
          setRadarEnabled(!radarEnabled);
        });
      }
      let restoreRadar = false;
      try { restoreRadar = localStorage.getItem('nflRadarEnabled') === '1'; } catch (e) {}
      if (restoreRadar) setRadarEnabled(true);

      nflGamePoints.forEach(function(game) {
        const color = nflSignalColors[game.signal] || '#64748b';
        const marker = L.circleMarker([game.lat, game.lon], {
          radius:8, color:'#f8fafc', weight:1.5, fillColor:color, fillOpacity:0.93
        });
        const popup = '<div class="map-popup">' +
          '<div class="popup-status" style="color:' + color + '">' +
            nflEscapeHtml(game.signal + ' · ' + game.status) + '</div>' +
          '<h3>' + nflEscapeHtml(game.away) + ' @ ' + nflEscapeHtml(game.home) + '</h3>' +
          '<div class="popup-sub">' + nflEscapeHtml(game.kickoff) + '<br>' +
            nflEscapeHtml(game.venue) + '</div>' +
          '<div class="popup-line"><strong>Market:</strong> ' + nflEscapeHtml(game.market) + '</div>' +
          '<div class="popup-line"><strong>Best:</strong> ' + nflEscapeHtml(game.best) + '</div>' +
          '<div class="popup-line"><strong>Model:</strong> ' + nflEscapeHtml(game.model) + '</div>' +
          '<div class="popup-line"><strong>Weather:</strong> ' + nflEscapeHtml(game.weather) + '</div>' +
          '<div class="popup-line"><strong>Decision state:</strong> ' + nflEscapeHtml(game.state.replaceAll('_',' ')) + '</div>' +
          '</div>';
        marker.bindPopup(popup, { maxWidth:350 });
        marker.bindTooltip(nflEscapeHtml(game.away + ' @ ' + game.home), { direction:'top' });
        marker.addTo(weekMap);
        nflMapMarkers.push({ marker:marker, signal:game.signal });
      });

      if (nflMapMarkers.length) {
        const bounds = L.latLngBounds(nflMapMarkers.map(function(item) { return item.marker.getLatLng(); }));
        weekMap.fitBounds(bounds.pad(0.08), { maxZoom:6 });
      } else {
        weekMap.setView([38.5, -97], 4);
      }

      function filterNflMap(signal) {
        let shown = 0;
        nflMapMarkers.forEach(function(item) {
          const visible = signal === 'ALL' || item.signal === signal;
          if (visible && !weekMap.hasLayer(item.marker)) item.marker.addTo(weekMap);
          if (!visible && weekMap.hasLayer(item.marker)) weekMap.removeLayer(item.marker);
          if (visible) shown += 1;
        });
        if (nflMapCount) nflMapCount.textContent = 'Showing ' + shown + ' mapped game' + (shown === 1 ? '' : 's');
      }

      nflMapFilters.forEach(function(button) {
        button.addEventListener('click', function() {
          nflMapFilters.forEach(function(other) { other.classList.remove('active'); });
          button.classList.add('active');
          filterNflMap(button.dataset.mapSignal || 'ALL');
        });
      });
    } else if (nflMapElement) {
      nflMapElement.innerHTML = '<div style="padding:28px;color:#334155">The interactive map library could not load. The full NFL table remains available below.</div>';
    }

    const boardSearch = document.getElementById('boardSearch');
    const boardStatusFilter = document.getElementById('boardStatusFilter');
    const boardSignalFilter = document.getElementById('boardSignalFilter');
    const weatherFilter = document.getElementById('weatherFilter');
    const boardBody = document.getElementById('nflBoardBody');
    const boardRows = boardBody ? [...boardBody.querySelectorAll('tr[data-status]')] : [];

    function filterNflBoard() {
      const q = boardSearch ? (boardSearch.value || '').toLowerCase() : '';
      const status = boardStatusFilter ? boardStatusFilter.value : 'ALL';
      const signal = boardSignalFilter ? boardSignalFilter.value : 'ALL';
      const weather = weatherFilter ? weatherFilter.value : 'ALL';
      boardRows.forEach(function(row) {
        const textMatch = row.innerText.toLowerCase().includes(q);
        const statusMatch = status === 'ALL' || row.dataset.status === status;
        const signalMatch = signal === 'ALL' || row.dataset.signal === signal;
        const weatherTypes = (row.dataset.weatherTypes || '').split(/\s+/).filter(Boolean);
        const weatherMatch = weather === 'ALL' || weatherTypes.includes(weather);
        row.style.display = textMatch && statusMatch && signalMatch && weatherMatch ? '' : 'none';
      });
    }

    if (boardSearch) boardSearch.addEventListener('input', filterNflBoard);
    if (boardStatusFilter) boardStatusFilter.addEventListener('change', filterNflBoard);
    if (boardSignalFilter) boardSignalFilter.addEventListener('change', filterNflBoard);
    if (weatherFilter) weatherFilter.addEventListener('change', filterNflBoard);

    let nflSortState = { index:null, asc:true };
    document.querySelectorAll('[data-nfl-sort]').forEach(function(button) {
      button.addEventListener('click', function() {
        if (!boardBody) return;
        const index = Number(button.dataset.nflSort);
        const asc = nflSortState.index === index ? !nflSortState.asc : true;
        nflSortState = { index:index, asc:asc };
        const rows = [...boardBody.querySelectorAll('tr[data-status]')];
        rows.sort(function(a, b) {
          const av = a.children[index] ? a.children[index].dataset.sortValue : '';
          const bv = b.children[index] ? b.children[index].dataset.sortValue : '';
          const an = Number(av), bn = Number(bv);
          const aMissing = av === '' || Number.isNaN(an);
          const bMissing = bv === '' || Number.isNaN(bn);
          if (aMissing && bMissing) return 0;
          if (aMissing) return 1;
          if (bMissing) return -1;
          return asc ? an - bn : bn - an;
        });
        rows.forEach(function(row) { boardBody.appendChild(row); });
      });
    });
  </script>
'''


def enhance_html(html: str, board: pd.DataFrame) -> str:
    if MARKER in html:
        return html

    map_html, map_payload = _map_section(board)
    table_html = _table_section(board)

    leaflet_css = (
        '<link rel="stylesheet" '
        'href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">\n'
    )
    if 'leaflet@1.9.4/dist/leaflet.css' not in html:
        html = html.replace('</title>\n', '</title>\n  ' + leaflet_css, 1)

    if '</style>' not in html:
        raise RuntimeError('NFL page has no style insertion point.')
    html = html.replace('  </style>', CSS + '\n  </style>', 1)

    nav_old = '<a href="#live">Live cards</a>'
    if nav_old not in html:
        raise RuntimeError('NFL page live-card navigation link is missing.')
    html = html.replace(
        nav_old,
        '<a href="#map">Week map</a>\n    ' + nav_old + '\n    <a href="#board">Full table</a>',
        1,
    )

    live_anchor = '    <section id="live">'
    if live_anchor not in html:
        raise RuntimeError('NFL live section insertion point is missing.')
    html = html.replace(live_anchor, map_html + '\n' + live_anchor, 1)

    ledger_anchor = '    <section id="ledger">'
    if ledger_anchor not in html:
        raise RuntimeError('NFL ledger section insertion point is missing.')
    html = html.replace(ledger_anchor, table_html + '\n' + ledger_anchor, 1)

    leaflet_js = '<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>\n'
    if 'leaflet@1.9.4/dist/leaflet.js' not in html:
        first_script = html.find('  <script>')
        if first_script < 0:
            raise RuntimeError('NFL page script insertion point is missing.')
        html = html[:first_script] + '  ' + leaflet_js + html[first_script:]

    parity_js = JS_TEMPLATE.replace('__NFL_MAP_DATA__', map_payload)
    html = html.replace('</body>', parity_js + '\n</body>', 1)
    return html
