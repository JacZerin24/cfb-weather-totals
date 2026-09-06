from __future__ import annotations

import csv
import json
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"
BOARD = ROOT / "outputs" / "weekly_board.csv"
OVERRIDES = ROOT / "site_tools" / "game_status_overrides.csv"
STALE_STATUS = "POSTPONED / STALE"
STALE_COLOR = "#f97316"
CSS_MARKER = "/* SITE_STALE_STATUS */"


def load_overrides() -> list[dict[str, str]]:
    if not OVERRIDES.exists():
        return []
    with OVERRIDES.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if str(row.get("game_id") or "").strip()]


def load_board_statuses() -> dict[str, str]:
    if not BOARD.exists():
        return {}
    with BOARD.open(newline="", encoding="utf-8") as f:
        return {str(row.get("game_id") or "").strip(): str(row.get("status") or "").strip() for row in csv.DictReader(f)}


def add_css(html: str) -> str:
    if CSS_MARKER in html:
        return html
    css = f"""
    {CSS_MARKER}
    .stale {{ color:#fed7aa; background:rgba(249,115,22,.14); border:1px solid rgba(249,115,22,.42); }}
    """
    return html.replace("  </style>", css + "\n  </style>", 1)


def add_table_filter_option(html: str) -> str:
    if f'<option>{STALE_STATUS}</option>' in html:
        return html
    return html.replace(
        '<option>WATCH</option><option>NO PLAY</option>',
        f'<option>WATCH</option><option>{STALE_STATUS}</option><option>NO PLAY</option>',
        1,
    )


def update_table_row(html: str, matchup: str, status: str, reason: str) -> str:
    matchup_html = escape(matchup)
    pattern = re.compile(
        rf'(<tr\s+[^>]*>.*?<strong>{re.escape(matchup_html)}</strong>.*?</tr>)',
        flags=re.S,
    )
    match = pattern.search(html)
    if not match:
        return html
    row = match.group(1)
    row = re.sub(r'data-status="[^"]*"', f'data-status="{escape(status)}"', row, count=1)
    row = re.sub(
        r'<span class="badge [^"]+">[^<]+</span>',
        f'<span class="badge stale">{escape(status)}</span>',
        row,
        count=1,
    )
    # The final table cell is the explanation / Why column.
    row = re.sub(r'<td>[^<]*(?:<[^>]+>[^<]*</[^>]+>[^<]*)*</td>\s*</tr>$', f'<td>{escape(reason)}</td></tr>', row)
    return html[:match.start(1)] + row + html[match.end(1):]


def update_map_json(html: str, overrides: list[dict[str, str]]) -> str:
    match = re.search(r"const gamePoints = (\[.*?\]);\n\s*const statusColors", html, flags=re.S)
    if not match:
        return html
    try:
        points = json.loads(match.group(1))
    except json.JSONDecodeError:
        return html
    by_id = {str(o["game_id"]).strip(): o for o in overrides}
    for point in points:
        game_id = str(point.get("game_id") or "").strip()
        override = by_id.get(game_id)
        if override:
            point["status"] = override.get("status") or STALE_STATUS
            point["reason"] = override.get("reason") or "Original model status is stale after postponement."
    payload = json.dumps(points, ensure_ascii=False).replace("<", "\\u003c")
    return html[:match.start(1)] + payload + html[match.end(1):]


def add_map_color(html: str) -> str:
    if f"'{STALE_STATUS}':'{STALE_COLOR}'" in html or f"'{STALE_STATUS}': '{STALE_COLOR}'" in html:
        return html
    return html.replace(
        "const statusColors = { ",
        f"const statusColors = {{ '{STALE_STATUS}':'{STALE_COLOR}', ",
        1,
    )


def add_map_filter(html: str, count: int) -> str:
    if f'data-map-status="{STALE_STATUS}"' in html:
        return html
    button = (
        f'<button type="button" class="map-filter" data-map-status="{STALE_STATUS}">'
        f'<span class="map-dot" style="background:{STALE_COLOR}"></span>{STALE_STATUS} <strong>{count}</strong></button>'
    )
    pattern = re.compile(r'(<div class="map-filters">)(.*?)(</div>)', flags=re.S)
    match = pattern.search(html)
    if not match:
        return html
    return html[:match.start()] + match.group(1) + match.group(2) + button + match.group(3) + html[match.end():]


def decrement_qualifies_map_count(html: str, amount: int) -> str:
    if amount <= 0:
        return html
    pattern = re.compile(r'(data-map-status="QUALIFIES"[^>]*>.*?<strong>)(\d+)(</strong>)', flags=re.S)
    match = pattern.search(html)
    if not match:
        return html
    value = max(0, int(match.group(2)) - amount)
    return html[:match.start(2)] + str(value) + html[match.end(2):]


def remove_stale_target_card(html: str, matchup: str) -> str:
    away, _, home = matchup.partition(" @ ")
    if not away or not home:
        return html
    pattern = re.compile(
        rf'<article class="target-card">.*?<h3>{re.escape(escape(away))} <span>@</span> {re.escape(escape(home))}</h3>.*?</article>',
        flags=re.S,
    )
    return pattern.sub('', html, count=1)


def renumber_target_cards(html: str) -> str:
    counter = 0
    def repl(match: re.Match) -> str:
        nonlocal counter
        counter += 1
        return f'<span class="rank">#{counter}</span>'
    return re.sub(r'<span class="rank">#\d+</span>', repl, html)


def patch_research_card(html: str, stale_matchups: list[str]) -> str:
    if not stale_matchups:
        return html
    pattern = re.compile(r'<div class="parlay active-card">.*?</div>', flags=re.S)
    match = pattern.search(html)
    if not match:
        return html
    current = match.group(0)
    if not any(m in current for m in stale_matchups):
        return html
    replacement = (
        '<div class="parlay"><span class="eyebrow">Top weekly 2-leg research card</span>'
        '<h3>No current card</h3><p>The original card included a qualifier that was postponed and is now stale. '
        'The official prospective record is preserved, but the site does not treat that leg as a current actionable qualifier.</p></div>'
    )
    return html[:match.start()] + replacement + html[match.end():]


def decrement_target_metric(html: str, amount: int) -> str:
    if amount <= 0:
        return html
    pattern = re.compile(r'(<div class="label">Qualifying targets</div><div class="value">)(\d+)(</div>)')
    match = pattern.search(html)
    if not match:
        return html
    value = max(0, int(match.group(2)) - amount)
    return html[:match.start(2)] + str(value) + html[match.end(2):]


def main() -> None:
    if not INDEX.exists():
        raise RuntimeError("docs/index.html is missing.")
    overrides = load_overrides()
    if not overrides:
        print("No site status overrides to apply.")
        return

    board_statuses = load_board_statuses()
    stale_qualifiers = sum(
        1 for o in overrides
        if (o.get("status") or STALE_STATUS) == STALE_STATUS
        and board_statuses.get(str(o.get("game_id") or "").strip()) == "QUALIFIES"
    )

    html = INDEX.read_text(encoding="utf-8")
    html = add_css(html)
    html = add_table_filter_option(html)
    html = update_map_json(html, overrides)
    html = add_map_color(html)
    html = add_map_filter(html, len(overrides))
    html = decrement_qualifies_map_count(html, stale_qualifiers)

    stale_matchups: list[str] = []
    for override in overrides:
        status = override.get("status") or STALE_STATUS
        reason = override.get("reason") or "Original model status is stale after postponement."
        matchup = override.get("matchup") or ""
        if matchup:
            html = update_table_row(html, matchup, status, reason)
            if status == STALE_STATUS:
                stale_matchups.append(matchup)
                html = remove_stale_target_card(html, matchup)

    html = renumber_target_cards(html)
    html = patch_research_card(html, stale_matchups)
    html = decrement_target_metric(html, stale_qualifiers)
    INDEX.write_text(html, encoding="utf-8")
    print(f"Applied {len(overrides)} site-only status override(s); official board/prospective files were not modified.")


if __name__ == "__main__":
    main()
