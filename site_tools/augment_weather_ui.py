from __future__ import annotations

import csv
import math
import re
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"
BOARD = ROOT / "outputs" / "weekly_board.csv"
CT = ZoneInfo("America/Chicago")

CSS_MARKER = "/* SITE_WEATHER_TOOLS */"
JS_MARKER = "// SITE_WEATHER_TOOLS"
RADAR_NOTE = "Radar overlay: latest CONUS NEXRAD base reflectivity from the Iowa Environmental Mesonet / NWS."


def text(value: object, fallback: str = "—") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return fallback if not s or s.lower() in {"nan", "none"} else s


def num(value: object) -> float | None:
    try:
        v = float(str(value).strip())
        return None if math.isnan(v) else v
    except Exception:
        return None


def fmt(value: object, decimals: int = 1) -> str:
    v = num(value)
    return "—" if v is None else f"{v:.{decimals}f}"


def fmt_int(value: object) -> str:
    v = num(value)
    return "—" if v is None else str(int(round(v)))


def kickoff_ct(value: object) -> str:
    raw = text(value, "")
    if not raw:
        return "Kickoff TBD"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(CT)
        return dt.strftime("%a %b %-d · %-I:%M %p CT")
    except Exception:
        return "Kickoff TBD"


def kickoff_sort_value(value: object) -> str:
    raw = text(value, "")
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return f"{dt.timestamp():.0f}"
    except Exception:
        return ""


def status_class(status: str) -> str:
    return {
        "QUALIFIES": "qualifies",
        "LEAN": "lean",
        "WATCH": "watch",
        "NO PLAY": "no-play",
        "NO LINE": "no-line",
    }.get(status, "no-play")


def division(row: dict[str, str]) -> str:
    existing = text(row.get("division_track"), "")
    if existing:
        return existing
    home = text(row.get("home_classification"), "").lower()
    away = text(row.get("away_classification"), "").lower()
    if home == "fbs" and away == "fbs":
        return "FBS"
    if home == "fcs" and away == "fcs":
        return "FCS"
    return "OTHER"


def market_html(row: dict[str, str]) -> str:
    total = num(row.get("closing_total"))
    if total is None:
        return "No current total"
    provider = escape(text(row.get("line_provider"), "unknown"))
    providers = num(row.get("line_provider_count"))
    note = f" · {int(round(providers))} providers" if providers is not None else ""
    return f"O/U {total:.1f} · {provider}{note}"


def model_html(row: dict[str, str]) -> str:
    residual = num(row.get("pred_market_residual"))
    if residual is None:
        base = "Not scored"
    else:
        direction = "UNDER" if residual < 0 else "OVER"
        projected = fmt(row.get("model_projected_total"))
        base = f"{direction} edge {abs(residual):.1f} · projected total {projected}"
    tags = [t.strip() for t in text(row.get("research_tags"), "").split(";") if t.strip()]
    if tags:
        base += '<br><span class="small">' + escape("; ".join(tags)) + "</span>"
    return base


def weather_cell(value: object, suffix: str = "") -> str:
    v = num(value)
    if v is None:
        return '<td class="weather-num" data-sort-value="">—</td>'
    display = f"{int(round(v))}{suffix}"
    return f'<td class="weather-num" data-sort-value="{v:.6f}">{escape(display)}</td>'


def row_html(row: dict[str, str]) -> str:
    status = text(row.get("status"), "NO PLAY")
    game = f"{text(row.get('away_team'), '')} @ {text(row.get('home_team'), '')}"
    reason = text(row.get("decision_reason"), "")
    kickoff_value = kickoff_sort_value(row.get("start_date"))
    return "".join([
        f'<tr data-status="{escape(status)}" data-division="{escape(division(row))}">',
        f'<td><span class="badge {status_class(status)}">{escape(status)}</span></td>',
        f'<td><strong>{escape(game)}</strong></td>',
        f'<td class="kickoff-num" data-sort-value="{kickoff_value}">{escape(kickoff_ct(row.get("start_date")))}</td>',
        f'<td>{market_html(row)}</td>',
        weather_cell(row.get("temperature_f"), "°F"),
        weather_cell(row.get("wind_mph"), " mph"),
        weather_cell(row.get("wind_gust_mph"), " mph"),
        weather_cell(row.get("humidity"), "%"),
        weather_cell(row.get("precip_probability_pct"), "%"),
        f'<td>{model_html(row)}</td>',
        f'<td>{escape(reason)}</td>',
        "</tr>",
    ])


def read_board() -> list[dict[str, str]]:
    with BOARD.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def replace_table(html: str, rows: list[dict[str, str]]) -> str:
    headers = (
        '<thead><tr><th>Status</th><th>Game</th>'
        '<th><button class="sort-btn" type="button" data-weather-sort="2">Kickoff <span>↕</span></button></th>'
        '<th>Market</th>'
        '<th><button class="sort-btn" type="button" data-weather-sort="4">Temp <span>↕</span></button></th>'
        '<th><button class="sort-btn" type="button" data-weather-sort="5">Wind <span>↕</span></button></th>'
        '<th><button class="sort-btn" type="button" data-weather-sort="6">Gust <span>↕</span></button></th>'
        '<th><button class="sort-btn" type="button" data-weather-sort="7">RH <span>↕</span></button></th>'
        '<th><button class="sort-btn" type="button" data-weather-sort="8" title="Probability of precipitation / chance of precipitation">PoP <span>↕</span></button></th>'
        '<th>Model</th><th>Why</th></tr></thead>'
    )
    body = '<tbody id="boardBody">' + "".join(row_html(r) for r in rows) + "</tbody>"
    pattern = re.compile(r'<thead><tr>.*?</tr></thead>\s*<tbody id="boardBody">.*?</tbody>', re.S)
    updated, count = pattern.subn(headers + body, html, count=1)
    if count != 1:
        raise RuntimeError("Could not locate the live-board table for weather-column augmentation.")
    updated = updated.replace(
        '<div class="section-head"><div><div class="eyebrow">Everything, including no-plays and no-lines</div><h2>Full weekly board</h2></div><p>The site shows why games fail the screen instead of hiding them.</p></div>',
        '<div class="section-head"><div><div class="eyebrow">Everything, including no-plays and no-lines</div><h2>Full weekly board</h2></div><p>Click Kickoff, Temp, Wind, Gust, RH, or PoP to sort. PoP is chance of precipitation.</p></div>',
    )
    updated = updated.replace(
        '<div class="section-head"><div><div class="eyebrow">Everything, including no-plays and no-lines</div><h2>Full weekly board</h2></div><p>Click Temp, Wind, Gust, RH, or PoP to sort. PoP is chance of precipitation.</p></div>',
        '<div class="section-head"><div><div class="eyebrow">Everything, including no-plays and no-lines</div><h2>Full weekly board</h2></div><p>Click Kickoff, Temp, Wind, Gust, RH, or PoP to sort. PoP is chance of precipitation.</p></div>',
    )
    return updated


def add_css(html: str) -> str:
    if CSS_MARKER in html:
        return html
    css = f"""
    {CSS_MARKER}
    .weather-num,.kickoff-num {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
    .sort-btn {{ appearance:none; border:0; background:transparent; color:#dbeafe; font:inherit; font-weight:900; padding:0; cursor:pointer; display:inline-flex; align-items:center; gap:5px; }}
    .sort-btn:hover,.sort-btn:focus-visible {{ color:var(--cyan); }}
    .sort-btn span {{ color:var(--muted); font-size:10px; }}
    .radar-toggle {{ border-color:rgba(125,211,252,.4); }}
    .radar-toggle.active {{ background:rgba(125,211,252,.14); border-color:rgba(125,211,252,.8); }}
    """
    return html.replace("  </style>", css + "\n  </style>", 1)


def add_radar_ui(html: str) -> str:
    if 'id="radarToggle"' not in html:
        html = html.replace(
            '<div class="map-count" id="mapCount">',
            '<button type="button" class="map-filter radar-toggle" id="radarToggle" aria-pressed="false">Radar: Off</button><div class="map-count" id="mapCount">',
            1,
        )
    if RADAR_NOTE not in html:
        html = html.replace(
            '</div>\n      </div>\n    </section>\n    <section id="targets">',
            f' {RADAR_NOTE}</div>\n      </div>\n    </section>\n    <section id="targets">',
            1,
        )
    if "const radarTileUrl" not in html:
        needle = """      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom:19,
        attribution:'&copy; OpenStreetMap contributors'
      }).addTo(weekMap);
"""
        radar = needle + """
      const radarTileUrl = 'https://mesonet.agron.iastate.edu/cache/tile.py/1.0.0/ridge::USCOMP-N0Q-0/{z}/{x}/{y}.png';
      const radarLayer = L.tileLayer(radarTileUrl, {
        opacity:0.58,
        maxZoom:12,
        attribution:'NEXRAD: Iowa Environmental Mesonet / NWS'
      });
      const radarToggle = document.getElementById('radarToggle');
      if (radarToggle) {
        radarToggle.addEventListener('click', function() {
          const turningOn = !weekMap.hasLayer(radarLayer);
          if (turningOn) radarLayer.addTo(weekMap); else weekMap.removeLayer(radarLayer);
          radarToggle.classList.toggle('active', turningOn);
          radarToggle.setAttribute('aria-pressed', turningOn ? 'true' : 'false');
          radarToggle.textContent = turningOn ? 'Radar: On' : 'Radar: Off';
        });
      }
"""
        if needle not in html:
            raise RuntimeError("Could not locate the Leaflet base layer for radar augmentation.")
        html = html.replace(needle, radar, 1)
    return html


def add_sort_js(html: str) -> str:
    if JS_MARKER in html:
        return html
    js = f"""

    {JS_MARKER}
    const weatherSortButtons = [...document.querySelectorAll('[data-weather-sort]')];
    let activeWeatherSortColumn = null;
    let activeWeatherSortAscending = true;
    weatherSortButtons.forEach(function(button) {{
      button.addEventListener('click', function() {{
        const column = Number(button.dataset.weatherSort);
        if (activeWeatherSortColumn === column) {{
          activeWeatherSortAscending = !activeWeatherSortAscending;
        }} else {{
          activeWeatherSortColumn = column;
          activeWeatherSortAscending = true;
        }}
        const tbody = document.getElementById('boardBody');
        const currentRows = [...tbody.querySelectorAll('tr')];
        currentRows.sort(function(a, b) {{
          const av = Number(a.children[column].dataset.sortValue);
          const bv = Number(b.children[column].dataset.sortValue);
          const aMissing = Number.isNaN(av) || a.children[column].dataset.sortValue === '';
          const bMissing = Number.isNaN(bv) || b.children[column].dataset.sortValue === '';
          if (aMissing && bMissing) return 0;
          if (aMissing) return 1;
          if (bMissing) return -1;
          return activeWeatherSortAscending ? av - bv : bv - av;
        }});
        currentRows.forEach(function(row) {{ tbody.appendChild(row); }});
        weatherSortButtons.forEach(function(other) {{ other.querySelector('span').textContent = '↕'; }});
        button.querySelector('span').textContent = activeWeatherSortAscending ? '▲' : '▼';
      }});
    }});
"""
    return html.replace("  </script>", js + "\n  </script>", 1)


def main() -> None:
    if not INDEX.exists() or not BOARD.exists():
        raise RuntimeError("Existing live site or weekly_board.csv is missing.")
    rows = read_board()
    html = INDEX.read_text(encoding="utf-8")
    html = replace_table(html, rows)
    html = add_css(html)
    html = add_radar_ui(html)
    html = add_sort_js(html)
    INDEX.write_text(html, encoding="utf-8")
    print(f"Updated {INDEX} using {len(rows)} existing board rows. No model scoring was run.")


if __name__ == "__main__":
    main()
