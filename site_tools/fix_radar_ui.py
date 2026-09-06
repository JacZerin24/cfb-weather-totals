from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"
CSS_MARKER = "/* SITE_RADAR_SYNC */"
JS_MARKER = "// SITE_RADAR_SYNC"
OLD_NOTE = "Radar overlay: latest CONUS NEXRAD base reflectivity from the Iowa Environmental Mesonet / NWS."
NEW_NOTE = (
    "Radar overlay: IEM / NWS CONUS NEXRAD base reflectivity. "
    "All visible tiles are pinned to one radar valid time so neighboring tiles cannot mix scans. "
    '<span id="radarStatus">Radar off.</span>'
)


def add_css(html: str) -> str:
    if CSS_MARKER in html:
        return html
    css = f"""
    {CSS_MARKER}
    #radarStatus {{ color:#bfdbfe; font-weight:800; }}
    """
    return html.replace("  </style>", css + "\n  </style>", 1)


def add_status_text(html: str) -> str:
    if 'id="radarStatus"' in html:
        return html
    if OLD_NOTE in html:
        return html.replace(OLD_NOTE, NEW_NOTE, 1)
    # Fallback for a regenerated map note whose wording changed.
    return html.replace(
        "</div>\n      </div>\n    </section>\n    <section id=\"targets\">",
        f" {NEW_NOTE}</div>\n      </div>\n    </section>\n    <section id=\"targets\">",
        1,
    )


def radar_block() -> str:
    return r'''
      // SITE_RADAR_SYNC
      // Use a timestamp-pinned IEM layer rather than the moving "-0" layer.
      // This prevents adjacent cached tiles from representing different 5-minute scans.
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
        // IEM mosaics are available every 5 minutes and generally finish shortly
        // afterward. Ten minutes back is a conservative fallback if metadata is
        // temporarily unavailable or blocked by the browser.
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
        const key = radarKeyFromIso(valid);
        return L.tileLayer(radarTileUrl(key), {
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
        // Leaflet can wait on an off-screen/failed tile before emitting load. Do not
        // leave two complete scans stacked indefinitely in that edge case.
        window.setTimeout(finishSwap, 3500);
      }

      function setRadarEnabled(enabled) {
        radarEnabled = enabled;
        try { localStorage.setItem('cfbRadarEnabled', enabled ? '1' : '0'); } catch (e) {}
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
        // Re-check metadata while the overlay is on. A new timestamp causes the
        // entire tile layer to swap together, so the map never intentionally mixes scans.
        radarRefreshTimer = window.setInterval(refreshRadar, 5 * 60 * 1000);
      }

      if (radarToggle) {
        radarToggle.addEventListener('click', function() {
          setRadarEnabled(!radarEnabled);
        });
      }

      let restoreRadar = false;
      try { restoreRadar = localStorage.getItem('cfbRadarEnabled') === '1'; } catch (e) {}
      if (restoreRadar) setRadarEnabled(true);
'''


def replace_radar_js(html: str) -> str:
    # Replace either the original moving latest-tile block or a previous version of
    # this synchronized block. Keep the game marker code immediately afterward.
    pattern = re.compile(
        r"\n\s*(?:" + re.escape(JS_MARKER) + r"\n\s*)?const radarTileUrl\s*=.*?\n\s*gamePoints\.forEach",
        flags=re.S,
    )
    replacement = "\n" + radar_block().rstrip() + "\n\n      gamePoints.forEach"
    updated, count = pattern.subn(lambda _: replacement, html, count=1)
    if count != 1:
        raise RuntimeError("Could not locate the existing radar block in docs/index.html.")
    return updated


def main() -> None:
    if not INDEX.exists():
        raise RuntimeError("docs/index.html is missing.")
    html = INDEX.read_text(encoding="utf-8")
    html = add_css(html)
    html = add_status_text(html)
    html = replace_radar_js(html)
    INDEX.write_text(html, encoding="utf-8")
    print("Replaced moving latest radar tiles with one-time-pinned, auto-refreshing radar layers.")


if __name__ == "__main__":
    main()
