from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"
RESULTS_DIR = ROOT / "diagnostics" / "results"
OVERRIDES = ROOT / "site_tools" / "game_status_overrides.csv"
CSS_MARKER = "/* SITE_POSTPONEMENT_DIAGNOSTICS */"
CENTRAL = ZoneInfo("America/Chicago")


def load_stale_game_ids() -> set[str]:
    if not OVERRIDES.exists():
        return set()
    with OVERRIDES.open(newline="", encoding="utf-8") as f:
        return {
            str(row.get("game_id") or "").strip()
            for row in csv.DictReader(f)
            if str(row.get("status") or "").strip() == "POSTPONED / STALE"
        }


def load_diagnostics(stale_game_ids: set[str]) -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    out: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        game_id = str(data.get("game_id") or "").strip()
        if game_id not in stale_game_ids:
            continue
        if not data.get("new_kickoff_utc") or not isinstance(data.get("fresh_ticket_line_rescore"), dict):
            continue
        out.append(data)
    return out


def fmt_num(value, digits: int = 0, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}{suffix}"


def fmt_kickoff(value: str) -> str:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(CENTRAL)
    except (TypeError, ValueError):
        return str(value or "—")
    hour = dt.hour % 12 or 12
    return f"{dt.strftime('%a %b')} {dt.day} · {hour}:{dt.minute:02d} {dt.strftime('%p')} CT"


def add_css(html: str) -> str:
    if CSS_MARKER in html:
        return html
    css = f"""
    {CSS_MARKER}
    .postponement-diagnostic-row td {{ padding:0 12px 14px; background:rgba(249,115,22,.035); }}
    .postponement-diagnostic {{ margin:0; padding:14px 15px; border:1px solid rgba(249,115,22,.34); border-radius:13px; background:linear-gradient(135deg,rgba(249,115,22,.11),rgba(125,211,252,.045)); }}
    .postponement-diagnostic-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; margin-bottom:10px; }}
    .postponement-diagnostic-title {{ font-weight:950; color:#ffedd5; }}
    .postponement-diagnostic-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; }}
    .postponement-diagnostic-item {{ border:1px solid rgba(255,255,255,.08); background:rgba(255,255,255,.035); border-radius:10px; padding:9px 10px; }}
    .postponement-diagnostic-item span {{ display:block; color:var(--muted); font-size:9px; font-weight:900; letter-spacing:.07em; text-transform:uppercase; }}
    .postponement-diagnostic-item strong {{ display:block; margin-top:3px; font-size:12px; color:#f8fbff; }}
    .postponement-diagnostic-note {{ margin-top:10px; color:#c9d7ea; font-size:11px; }}
    @media(max-width:900px) {{ .postponement-diagnostic-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    """
    return html.replace("  </style>", css + "\n  </style>", 1)


def remove_existing_diagnostic_rows(html: str) -> str:
    return re.sub(
        r'<tr class="postponement-diagnostic-row" data-diagnostic-game="[^"]+">.*?</tr>',
        "",
        html,
        flags=re.S,
    )


def diagnostic_summary(data: dict) -> dict[str, str]:
    weather = data.get("fresh_nws_weather") or {}
    score = data.get("fresh_ticket_line_rescore") or {}
    side = str(score.get("model_side") or "—").upper()
    edge = fmt_num(score.get("abs_edge"), 1)
    projected = fmt_num(score.get("projected_total"), 1)
    classification = str(score.get("classification_if_scored_fresh") or "—")
    ticket_total = fmt_num(data.get("ticket_total"), 1)
    weather_text = (
        f"{fmt_num(weather.get('temperature_f'), 0, '°F')} · "
        f"wind {fmt_num(weather.get('wind_mph'), 0, ' mph')}, "
        f"gust {fmt_num(weather.get('wind_gust_mph'), 0, ' mph')} · "
        f"RH {fmt_num(weather.get('humidity_pct'), 0, '%')} · "
        f"PoP {fmt_num(weather.get('pop_pct'), 0, '%')}"
    )
    return {
        "kickoff": fmt_kickoff(data.get("new_kickoff_utc")),
        "weather": weather_text,
        "ticket": f"O/U {ticket_total}",
        "model": f"{side} edge {edge} · projected {projected}",
        "classification": classification,
    }


def diagnostic_row(data: dict, colspan: int) -> str:
    game_id = escape(str(data.get("game_id") or ""))
    summary = diagnostic_summary(data)
    classification = escape(summary["classification"])
    badge_class = "no-play" if classification == "NO PLAY" else "watch"
    return (
        f'<tr class="postponement-diagnostic-row" data-diagnostic-game="{game_id}">'
        f'<td colspan="{colspan}"><div class="postponement-diagnostic">'
        '<div class="postponement-diagnostic-head">'
        '<div><div class="postponement-diagnostic-title">Updated postponement re-score</div>'
        '<div class="small">Fresh kickoff/weather diagnostic for the rescheduled game</div></div>'
        f'<span class="badge {badge_class}">{classification}</span></div>'
        '<div class="postponement-diagnostic-grid">'
        f'<div class="postponement-diagnostic-item"><span>Updated kickoff</span><strong>{escape(summary["kickoff"])}</strong></div>'
        f'<div class="postponement-diagnostic-item"><span>Updated weather</span><strong>{escape(summary["weather"])}</strong></div>'
        f'<div class="postponement-diagnostic-item"><span>Ticket line</span><strong>{escape(summary["ticket"])}</strong></div>'
        f'<div class="postponement-diagnostic-item"><span>Updated model</span><strong>{escape(summary["model"])}</strong></div>'
        f'<div class="postponement-diagnostic-item"><span>Decision status</span><strong>{classification}</strong></div>'
        '</div>'
        '<div class="postponement-diagnostic-note"><strong>Decision-support diagnostic only.</strong> '
        'This re-score uses the rescheduled kickoff and fresh NWS weather. It does not overwrite the original qualifier, '
        'official prospective record, or any other game prediction.</div>'
        '</div></td></tr>'
    )


def table_column_count(html: str) -> int:
    match = re.search(r'<table><thead><tr>(.*?)</tr></thead>', html, flags=re.S)
    if not match:
        return 11
    count = len(re.findall(r'<th(?:\s|>)', match.group(1)))
    return count or 11


def insert_diagnostic_after_game(html: str, data: dict, colspan: int) -> str:
    matchup = escape(str(data.get("matchup") or ""))
    if not matchup:
        return html
    pattern = re.compile(
        rf'(<tr\s+[^>]*>.*?<strong>{re.escape(matchup)}</strong>.*?</tr>)',
        flags=re.S,
    )
    match = pattern.search(html)
    if not match:
        return html
    row = diagnostic_row(data, colspan)
    return html[:match.end(1)] + row + html[match.end(1):]


def update_map_reason(html: str, diagnostics: list[dict]) -> str:
    match = re.search(r"const gamePoints = (\[.*?\]);\n\s*const statusColors", html, flags=re.S)
    if not match:
        return html
    try:
        points = json.loads(match.group(1))
    except json.JSONDecodeError:
        return html
    by_id = {str(d.get("game_id") or "").strip(): d for d in diagnostics}
    for point in points:
        game_id = str(point.get("game_id") or "").strip()
        data = by_id.get(game_id)
        if not data:
            continue
        summary = diagnostic_summary(data)
        base_reason = str(point.get("reason") or "").strip()
        diagnostic_reason = (
            f"Updated re-score for {summary['kickoff']}: {summary['model']}; "
            f"{summary['classification']}. {summary['weather']}."
        )
        if "Updated re-score for" in base_reason:
            base_reason = re.sub(r"\s*Updated re-score for .*?$", "", base_reason).strip()
        point["reason"] = (base_reason + " " + diagnostic_reason).strip()
    payload = json.dumps(points, ensure_ascii=False).replace("<", "\\u003c")
    return html[:match.start(1)] + payload + html[match.end(1):]


def main() -> None:
    if not INDEX.exists():
        raise RuntimeError("docs/index.html is missing.")
    stale_game_ids = load_stale_game_ids()
    diagnostics = load_diagnostics(stale_game_ids)
    if not diagnostics:
        print("No stale-game postponement diagnostics to display.")
        return

    html = INDEX.read_text(encoding="utf-8")
    html = add_css(html)
    html = remove_existing_diagnostic_rows(html)
    colspan = table_column_count(html)
    for data in diagnostics:
        html = insert_diagnostic_after_game(html, data, colspan)
    html = update_map_reason(html, diagnostics)
    INDEX.write_text(html, encoding="utf-8")
    print(f"Displayed {len(diagnostics)} postponement diagnostic(s) without modifying model/prospective data.")


if __name__ == "__main__":
    main()
