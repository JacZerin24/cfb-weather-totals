from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

import pandas as pd

from .fcs_model import FCS_QUALIFY_EDGE, FCS_QUALIFY_TOTAL
from .utils import ROOT, ensure_dir


def _game_id(value) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return str(value or '').strip()


def _division_buttons(board: pd.DataFrame) -> str:
    mapped = board[board['venue_latitude'].notna() & board['venue_longitude'].notna()].copy()
    counts = mapped.get('division_track', pd.Series('OTHER', index=mapped.index)).fillna('OTHER').value_counts().to_dict()
    return ''.join([
        f'<button type="button" class="division-filter active" data-division="ALL">All <strong>{len(mapped)}</strong></button>',
        f'<button type="button" class="division-filter" data-division="FBS">FBS <strong>{int(counts.get("FBS", 0))}</strong></button>',
        f'<button type="button" class="division-filter" data-division="FCS">FCS <strong>{int(counts.get("FCS", 0))}</strong></button>',
        f'<button type="button" class="division-filter" data-division="OTHER">Other <strong>{int(counts.get("OTHER", 0))}</strong></button>',
    ])


def _add_division_to_map_json(html: str, board: pd.DataFrame) -> str:
    match = re.search(r"const gamePoints = (\[.*?\]);\n    const statusColors", html, flags=re.S)
    if not match:
        return html
    try:
        points = json.loads(match.group(1))
    except json.JSONDecodeError:
        return html

    info = {}
    for _, row in board.iterrows():
        info[_game_id(row.get('game_id'))] = {
            'division': str(row.get('division_track') or 'OTHER'),
            'model_track': str(row.get('model_track') or 'GENERAL HGB'),
        }
    for point in points:
        meta = info.get(_game_id(point.get('game_id')), {'division': 'OTHER', 'model_track': 'GENERAL HGB'})
        point.update(meta)
    payload = json.dumps(points, ensure_ascii=False).replace('<', '\\u003c')
    return html[:match.start(1)] + payload + html[match.end(1):]


def _add_table_divisions(html: str, board: pd.DataFrame) -> str:
    divisions = [str(v or 'OTHER') for v in board.get('division_track', pd.Series('OTHER', index=board.index)).tolist()]
    index = 0

    def repl(match: re.Match) -> str:
        nonlocal index
        division = divisions[index] if index < len(divisions) else 'OTHER'
        index += 1
        return f'{match.group(0)[:-1]} data-division="{escape(division)}">'

    return re.sub(r'<tr data-status="[^"]+">', repl, html)


def augment_html(html: str, board: pd.DataFrame) -> str:
    if 'data-division="FCS"' in html and 'division-filter' in html:
        return html

    html = _add_division_to_map_json(html, board)
    html = _add_table_divisions(html, board)

    css = '''
    .division-filters { display:flex; gap:8px; flex-wrap:wrap; width:100%; padding-bottom:2px; }
    .division-filter { min-height:38px; border:1px solid var(--line); background:#142137; color:var(--text); padding:7px 11px; border-radius:10px; cursor:pointer; font-weight:900; }
    .division-filter:hover,.division-filter:focus-visible { border-color:rgba(125,211,252,.55); }
    .division-filter.active { background:rgba(125,211,252,.13); border-color:rgba(125,211,252,.75); box-shadow:0 0 0 2px rgba(125,211,252,.18) inset; }
    .track-note { margin:10px 0 0; color:var(--muted); font-size:12px; }
'''
    html = html.replace('    footer { color:var(--muted);', css + '    footer { color:var(--muted);')

    html = html.replace(
        'Current market totals are scored with the same HistGradientBoosting residual model validated in the historical research. Outdoor weather comes from the National Weather Service forecast grid at the game venue and kickoff time.',
        'Current totals are scored with division-aware research tracks: the existing CFB HGB workflow plus a dedicated FCS-vs-FCS HGB model. Outdoor weather comes from the National Weather Service forecast grid at the game venue and kickoff time.',
    )
    html = html.replace(
        '<a href="research.html">Historical research →</a>',
        '<a href="research.html">Historical research →</a><a href="fcs-research.html">FCS research →</a>',
    )
    html = html.replace(
        '<div class="map-toolbar"><div class="map-filters">',
        '<div class="map-toolbar"><div class="division-filters" id="divisionFilters">' + _division_buttons(board) + '</div><div class="map-filters">',
    )
    html = html.replace(
        'HGB under edge ≥3.5 + market total ≥56 + usable kickoff forecast/time.',
        f'FBS/general screen: HGB under edge ≥3.5 + total ≥56. FCS screen: FCS-only HGB under edge ≥{FCS_QUALIFY_EDGE:.1f} + total ≥{FCS_QUALIFY_TOTAL:.0f}. Both require a usable kickoff forecast/time.',
    )

    old_toolbar = '<div class="toolbar"><input id="search" type="search" placeholder="Search matchup, weather, reason…"><select id="statusFilter"><option value="ALL">All statuses</option><option>QUALIFIES</option><option>LEAN</option><option>WATCH</option><option>NO PLAY</option><option>NO LINE</option></select></div>'
    new_toolbar = '<div class="toolbar"><input id="search" type="search" placeholder="Search matchup, weather, reason…"><select id="divisionFilter"><option value="ALL">All divisions</option><option value="FBS">FBS vs FBS</option><option value="FCS">FCS vs FCS</option><option value="OTHER">Other / cross-division</option></select><select id="statusFilter"><option value="ALL">All statuses</option><option>QUALIFIES</option><option>LEAN</option><option>WATCH</option><option>NO PLAY</option><option>NO LINE</option></select></div>'
    html = html.replace(old_toolbar, new_toolbar)

    html = html.replace(
        "    const mapFilters = [...document.querySelectorAll('.map-filter')];\n    const mapMarkers = [];",
        "    const mapFilters = [...document.querySelectorAll('.map-filter')];\n    const divisionFilters = [...document.querySelectorAll('.division-filter')];\n    let activeMapStatus = 'ALL';\n    let activeDivision = 'ALL';\n    const mapMarkers = [];",
    )
    html = html.replace(
        "        mapMarkers.push({ marker:marker, status:game.status });",
        "        mapMarkers.push({ marker:marker, status:game.status, division:game.division || 'OTHER' });",
    )
    html = html.replace(
        "          '<div class=\"popup-status\" style=\"color:' + color + '\">' + escapeHtml(game.status) + '</div>' +",
        "          '<div class=\"popup-status\" style=\"color:' + color + '\">' + escapeHtml(game.status + ' · ' + (game.division || 'OTHER')) + '</div>' +",
    )
    html = html.replace(
        "          '<div class=\"popup-line\"><strong>Model:</strong> ' + escapeHtml(game.model) + '</div>' +",
        "          '<div class=\"popup-line\"><strong>Model:</strong> ' + escapeHtml(game.model) + '<br><span style=\"color:#94a3b8\">' + escapeHtml(game.model_track || '') + '</span></div>' +",
    )

    filter_pattern = re.compile(
        r"      function filterMap\(status\) \{.*?      \}\n\n      mapFilters\.forEach",
        flags=re.S,
    )
    new_filter = '''      function filterMap(status) {
        activeMapStatus = status;
        let shown = 0;
        mapMarkers.forEach(function(item) {
          const statusVisible = activeMapStatus === 'ALL' || item.status === activeMapStatus;
          const divisionVisible = activeDivision === 'ALL' || item.division === activeDivision;
          const visible = statusVisible && divisionVisible;
          if (visible && !weekMap.hasLayer(item.marker)) item.marker.addTo(weekMap);
          if (!visible && weekMap.hasLayer(item.marker)) weekMap.removeLayer(item.marker);
          if (visible) shown += 1;
        });
        mapCount.textContent = 'Showing ' + shown + ' mapped game' + (shown === 1 ? '' : 's');
      }

      mapFilters.forEach'''
    html = filter_pattern.sub(new_filter, html, count=1)

    old_map_listener_end = '''      mapFilters.forEach(function(button) {
        button.addEventListener('click', function() {
          mapFilters.forEach(function(other) { other.classList.remove('active'); });
          button.classList.add('active');
          filterMap(button.dataset.mapStatus || 'ALL');
        });
      });
    } else if (mapElement) {'''
    new_map_listener_end = '''      mapFilters.forEach(function(button) {
        button.addEventListener('click', function() {
          mapFilters.forEach(function(other) { other.classList.remove('active'); });
          button.classList.add('active');
          filterMap(button.dataset.mapStatus || 'ALL');
        });
      });
      divisionFilters.forEach(function(button) {
        button.addEventListener('click', function() {
          divisionFilters.forEach(function(other) { other.classList.remove('active'); });
          button.classList.add('active');
          activeDivision = button.dataset.division || 'ALL';
          filterMap(activeMapStatus);
        });
      });
    } else if (mapElement) {'''
    html = html.replace(old_map_listener_end, new_map_listener_end)

    html = html.replace(
        "    const statusFilter = document.getElementById('statusFilter');\n    const rows = [...document.querySelectorAll('#boardBody tr')];",
        "    const divisionFilter = document.getElementById('divisionFilter');\n    const statusFilter = document.getElementById('statusFilter');\n    const rows = [...document.querySelectorAll('#boardBody tr')];",
    )
    html = html.replace(
        "      const s = statusFilter.value;\n      rows.forEach(function(row) {\n        const showText = row.innerText.toLowerCase().includes(q);\n        const showStatus = s === 'ALL' || row.dataset.status === s;\n        row.style.display = showText && showStatus ? '' : 'none';",
        "      const d = divisionFilter ? divisionFilter.value : 'ALL';\n      const s = statusFilter.value;\n      rows.forEach(function(row) {\n        const showText = row.innerText.toLowerCase().includes(q);\n        const showDivision = d === 'ALL' || row.dataset.division === d;\n        const showStatus = s === 'ALL' || row.dataset.status === s;\n        row.style.display = showText && showDivision && showStatus ? '' : 'none';",
    )
    html = html.replace(
        "    search.addEventListener('input', filterRows);\n    statusFilter.addEventListener('change', filterRows);",
        "    search.addEventListener('input', filterRows);\n    if (divisionFilter) divisionFilter.addEventListener('change', filterRows);\n    statusFilter.addEventListener('change', filterRows);",
    )
    return html


def fcs_research_page() -> str:
    by_season_path = ROOT / 'outputs/fcs_selected_screen_by_season.csv'
    diag_path = ROOT / 'outputs/fcs_model_diagnostics.csv'
    grid_path = ROOT / 'outputs/fcs_threshold_grid.csv'
    data_path = ROOT / 'outputs/fcs_data_summary.csv'
    if not all(p.exists() for p in [by_season_path, diag_path, grid_path, data_path]):
        return '<!doctype html><html><body><h1>FCS research output not available yet.</h1></body></html>'

    by_season = pd.read_csv(by_season_path)
    diag = pd.read_csv(diag_path)
    grid = pd.read_csv(grid_path)
    data = pd.read_csv(data_path)
    selected = grid[
        grid['under_edge_threshold'].eq(FCS_QUALIFY_EDGE)
        & grid['minimum_total'].eq(FCS_QUALIFY_TOTAL)
    ].iloc[0]

    rows = ''.join(
        f'<tr><td>{int(r.season)}</td><td>{int(r.wins)}-{int(r.losses)}</td><td>{r.hit_rate:.1%}</td><td>{r.roi_per_1u:.1%}</td></tr>'
        for _, r in by_season.iterrows()
    )
    diag_rows = ''.join(
        f'<tr><td>{int(r.test_season)}</td><td>{int(r.train_games)}</td><td>{int(r.test_games)}</td><td>{r.model_mae:.2f}</td><td>{r.zero_residual_baseline_mae:.2f}</td></tr>'
        for _, r in diag.iterrows()
    )
    total_games = int(data['games_with_totals'].sum())
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FCS Totals Research</title>
<style>body{{margin:0;background:#08101d;color:#eef6ff;font-family:Inter,system-ui,sans-serif;line-height:1.5}}main{{width:min(1000px,calc(100vw - 28px));margin:0 auto;padding:34px 0 60px}}a{{color:#7dd3fc}}.card{{background:#111c31;border:1px solid rgba(255,255,255,.11);border-radius:18px;padding:20px;margin:16px 0}}h1{{font-size:clamp(34px,6vw,58px);margin:8px 0}}.eyebrow{{color:#a7f3d0;text-transform:uppercase;font-size:12px;font-weight:900;letter-spacing:.12em}}.big{{font-size:30px;font-weight:900}}p,small{{color:#a9bad5}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid rgba(255,255,255,.1);text-align:left}}th{{color:#dbeafe}}.warn{{border-color:rgba(245,158,11,.35)}}</style></head>
<body><main><a href="index.html">← Live board</a><div class="eyebrow">Dedicated division track</div><h1>FCS Weather Totals Research</h1><p>FCS-vs-FCS games only. This remains a research and paper-tracking model.</p>
<div class="card"><div class="eyebrow">Current conservative candidate</div><div class="big">UNDER edge ≥ {FCS_QUALIFY_EDGE:.1f} · total ≥ {FCS_QUALIFY_TOTAL:.0f}</div><p>{int(selected.wins)}-{int(selected.losses)} walk-forward record · {selected.hit_rate:.1%} hit rate · {selected.roi_per_1u:.1%} paper ROI per graded play at -110.</p><small>{total_games:,} historical FCS games with usable totals/results are available in the current training dataset.</small></div>
<div class="card"><h2>Walk-forward by season</h2><table><thead><tr><th>Season</th><th>Record</th><th>Hit rate</th><th>Paper ROI</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="card warn"><h2>Model diagnostic guardrail</h2><p>The FCS HGB's overall residual MAE does not beat a zero-residual baseline in these seasons. That is why the site does <strong>not</strong> treat every prediction as actionable; it only promotes the strict selective under subset above.</p><table><thead><tr><th>Test season</th><th>Train</th><th>Test</th><th>HGB MAE</th><th>Zero baseline MAE</th></tr></thead><tbody>{diag_rows}</tbody></table></div>
<div class="card"><h2>Interpretation</h2><p>The historical FCS market sample begins in 2022, so this evidence is useful but younger than the broader CFB research. The 2026 FCS board should be paper tracked before increasing confidence, and no FCS over strategy is currently promoted.</p></div>
</main></body></html>'''


def main() -> None:
    board_path = ROOT / 'outputs/weekly_board.csv'
    if not board_path.exists():
        raise RuntimeError('weekly_board.csv must exist before dashboard augmentation.')
    board = pd.read_csv(board_path)

    for relative in ['docs/index.html', 'outputs/live_dashboard.html']:
        path = ROOT / relative
        if not path.exists():
            continue
        html = path.read_text(encoding='utf-8')
        path.write_text(augment_html(html, board), encoding='utf-8')

    docs = ensure_dir('docs')
    (docs / 'fcs-research.html').write_text(fcs_research_page(), encoding='utf-8')
    print('Added division controls and FCS research page to the live dashboard.')


if __name__ == '__main__':
    main()
