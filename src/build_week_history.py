"""Read-only historical CFB weekly pages built from immutable pregame snapshots.

This is a presentation-only derived product. It never edits immutable snapshots,
official entries, model forecasts, thresholds, or the operational weekly board.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .augment_dashboard_divisions import augment_html
from .build_live_dashboard import dashboard_html
from .cfbd_client import CFBDClient
from .fcs_card_guard import card_html, eligible_card_targets
from .fcs_research_labels import label_html
from .utils import ROOT
from site_tools import augment_weather_ui as weather_ui

CT = ZoneInfo("America/Chicago")
SOURCE = ROOT / "outputs/prospective/2026"
DEST = ROOT / "docs/history"
NAV_START = "<!-- WEEK_HISTORY_NAV_START -->"
NAV_END = "<!-- WEEK_HISTORY_NAV_END -->"


def game_key(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return ""


def completed_results(season: int) -> pd.DataFrame:
    """Completed results for *all* games, not just official betting entries.

    On missing API credentials/outages the already graded immutable-entry
    results are used as a partial fallback. Never infer that 0-0 is final.
    """
    cols = ["game_id", "final_away_points", "final_home_points", "actual_total_points"]
    if not os.getenv("CFBD_API_KEY"):
        return pd.DataFrame(columns=cols)
    records: list[dict[str, Any]] = []
    try:
        client = CFBDClient()
        for season_type in ("regular", "postseason"):
            records.extend(client.get("/games", {"year": season, "seasonType": season_type}))
    except Exception as exc:
        print(f"History results refresh unavailable: {type(exc).__name__}: {exc}")
        if not records:
            return pd.DataFrame(columns=cols)
    if not records:
        return pd.DataFrame(columns=cols)
    games = pd.json_normalize(records).rename(columns={
        "id": "game_id",
        "homePoints": "final_home_points",
        "awayPoints": "final_away_points",
    })
    for col in cols:
        if col not in games:
            games[col] = pd.NA
    completed = games.get("completed", pd.Series(False, index=games.index))
    completed = completed.astype(str).str.lower().isin({"true", "1", "yes"})
    for col in ("final_home_points", "final_away_points"):
        games[col] = pd.to_numeric(games[col], errors="coerce")
    games["actual_total_points"] = games["final_home_points"] + games["final_away_points"]
    games = games[completed & games["actual_total_points"].notna()].copy()
    games["game_id"] = games["game_id"].map(game_key)
    return games[cols].drop_duplicates("game_id")


def read_snapshots() -> pd.DataFrame:
    chunks = []
    for path in sorted((SOURCE / "snapshots").glob("board_*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if not frame.empty and {"game_id", "start_date", "snapshot_timestamp_utc"}.issubset(frame):
            chunks.append(frame)
    if not chunks:
        return pd.DataFrame()
    frames = pd.concat(chunks, ignore_index=True)
    frames["game_id"] = frames["game_id"].map(game_key)
    frames["start_date"] = pd.to_datetime(frames["start_date"], utc=True, errors="coerce")
    frames["snapshot_timestamp_utc"] = pd.to_datetime(
        frames["snapshot_timestamp_utc"], utc=True, errors="coerce"
    )
    frames = frames[
        frames["game_id"].ne("")
        & frames["start_date"].notna()
        & frames["snapshot_timestamp_utc"].notna()
        & frames["snapshot_timestamp_utc"].lt(frames["start_date"])
    ].copy()
    if frames.empty:
        return frames
    kickoff_ct = frames["start_date"].dt.tz_convert(CT)
    week_start_ct = kickoff_ct.dt.normalize() - pd.to_timedelta(kickoff_ct.dt.weekday, unit="D")
    frames["week_start"] = week_start_ct.dt.strftime("%Y-%m-%d")
    frames = frames.sort_values(["snapshot_timestamp_utc", "game_id"])
    # An exact pre-kickoff archived row, not a retrospectively re-scored game.
    return frames.drop_duplicates(["game_id", "week_start"], keep="last")


def read_grades() -> pd.DataFrame:
    path = SOURCE / "graded_entries.csv"
    if not path.exists():
        return pd.DataFrame()
    grades = pd.read_csv(path, low_memory=False)
    if grades.empty:
        return grades
    grades["game_id"] = grades["game_id"].map(game_key)
    grades = grades[grades["game_id"].ne("")].copy()
    if "snapshot_timestamp_utc" in grades:
        grades = grades.sort_values("snapshot_timestamp_utc")
    return grades.drop_duplicates("game_id", keep="last").set_index("game_id")


def is_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1"}


def value_or_none(value: Any) -> float | None:
    try:
        number = float(value)
        return None if pd.isna(number) else number
    except (ValueError, TypeError, OverflowError):
        return None


def outcome_cell(row: pd.Series, grade: pd.Series | None) -> str:
    away = value_or_none(row.get("final_away_points"))
    home = value_or_none(row.get("final_home_points"))
    total = value_or_none(row.get("actual_total_points"))
    if away is None or home is None or total is None:
        return '<td class="history-result"><span class="small">Final not yet verified</span></td>'
    score = f'{int(away)}–{int(home)}'
    parts = [f'<strong>{escape(score)}</strong> <span class="small">away–home</span>',
             f'<div>Total: <strong>{int(total)}</strong></div>']
    line = value_or_none(row.get("closing_total"))
    side = str(row.get("model_side", "")).lower()
    if side not in ("over", "under"):
        residual = value_or_none(row.get("pred_market_residual"))
        side = "under" if residual is not None and residual < 0 else (
            "over" if residual is not None else ""
        )
    if side and line is not None:
        verdict = "PUSH" if total == line else (
            "HIT" if (total < line if side == "under" else total > line) else "MISS"
        )
        parts.append(
            f'<div class="small">Snapshot {escape(side.upper())} {line:.1f}: '
            f'<strong>{verdict}</strong> (hypothetical)</div>'
        )
    else:
        parts.append('<div class="small">No scorable snapshot prediction</div>')
    if grade is not None and is_yes(grade.get("paper_qualifier")):
        official = str(grade.get("paper_result", "")).strip().lower()
        if official in ("win", "loss", "push"):
            label = official.upper()
        elif official == "excluded":
            label = "EXCLUDED"
        else:
            label = "PENDING"
        parts.append(f'<div class="history-official">Official paper entry: {escape(label)}</div>')
        official_line = value_or_none(grade.get("closing_total"))
        if official_line is not None and (line is None or abs(line - official_line) > 0.001):
            parts.append(f'<div class="small">Official entry line: {official_line:.1f}</div>')
    return '<td class="history-result">' + "".join(parts) + "</td>"


def finalized_board(week: pd.DataFrame, grades: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    board = week.copy()
    if not scores.empty:
        board = board.merge(scores, on="game_id", how="left", suffixes=("", "_score"))
    for col in ("final_away_points", "final_home_points", "actual_total_points"):
        if col not in board:
            board[col] = pd.NA
    for col in ("final_away_points", "final_home_points", "actual_total_points"):
        if col not in grades.columns:
            continue
        missing = board[col].isna()
        board.loc[missing, col] = board.loc[missing, "game_id"].map(grades[col])
    if "status_rank" in board:
        board = board.sort_values(["status_rank", "start_date", "game_id"])
    else:
        board = board.sort_values(["start_date", "game_id"])
    return board.reset_index(drop=True)


def week_description(frame: pd.DataFrame, start: str) -> str:
    dates = pd.Timestamp(start)
    end = dates + pd.Timedelta(days=6)
    label = "Week"
    values = pd.to_numeric(frame.get("week", pd.Series(dtype=float)), errors="coerce").dropna()
    if not values.empty:
        label = f'Week {int(values.mode().iloc[0])}'
    return f'{label} · {dates.strftime("%b %-d")}–{end.strftime("%b %-d")}'


def navigation(weeks: list[tuple[str, str]], selected: str | None, history: bool) -> str:
    live = "../index.html" if history else "index.html"
    option_list = [
        f'<option value="{live}"{" selected" if selected is None else ""}>Current week · Live</option>'
    ]
    for name, label in weeks:
        href = f'{name}.html' if history else f'history/{name}.html'
        option_list.append(
            f'<option value="{escape(href)}"{" selected" if selected == name else ""}>'
            f'{escape(label)}</option>'
        )
    return (
        NAV_START +
        '<div class="shell week-history-nav"><label for="weekHistorySelect">'
        'Season / week</label><select id="weekHistorySelect" aria-label="Select college football week" '
        'onchange="if(this.value) window.location.href=this.value;">' +
        "".join(option_list) + "</select></div>" + NAV_END
    )


def add_navigation(html: str, weeks: list[tuple[str, str]], selected: str | None, history: bool) -> str:
    html = re.sub(re.escape(NAV_START) + r".*?" + re.escape(NAV_END), "", html, flags=re.S)
    css = """
    .week-history-nav { display:flex; gap:12px; align-items:center; padding:12px 0; }
    .week-history-nav label { color:var(--muted); font-size:12px; font-weight:900; }
    .week-history-nav select { max-width:min(100%,360px); }
    .history-result { min-width:184px; line-height:1.6; }
    .history-official { color:#bbf7d0; font-weight:800; }
    .history-note { margin:0 0 18px; padding:13px 17px; border:1px solid var(--line);
      background:#101b30; border-radius:12px; color:#cbd5e1; font-size:13px; }
    """
    if "    .week-history-nav {" not in html:
        html = html.replace("  </style>", css + "\n  </style>", 1)
    html = html.replace("</nav>", "</nav>\n" + navigation(weeks, selected, history), 1)
    if not history:
        html = html.replace('<a href="#board">Full board</a>',
                            '<a href="#board">Full board</a><a href="history/">Previous weeks</a>', 1)
    return html


def render_archive(
    board: pd.DataFrame, grades: pd.DataFrame, weeks: list[tuple[str, str]], name: str,
    description: str,
) -> str:
    final_stamp = board["snapshot_timestamp_utc"].max().isoformat()
    season = int(pd.to_numeric(board["season"], errors="coerce").mode().iloc[0])
    week_values = pd.to_numeric(board.get("week", pd.Series(dtype=float)), errors="coerce").dropna()
    number = int(week_values.mode().iloc[0]) if not week_values.empty else "—"
    html = dashboard_html(board, pd.DataFrame(), {
        "generated_at": final_stamp, "season": season, "week": number
    })
    html = augment_html(html, board)
    html = weather_ui.replace_table(html, [])
    rows = []
    for _, row in board.iterrows():
        data = {k: "" if pd.isna(v) else str(v) for k, v in row.items()}
        at = row["snapshot_timestamp_utc"].tz_convert(CT)
        tags = data.get("research_tags", "")
        data["research_tags"] = (tags + "; " if tags else "") + (
            "Last pregame snapshot: " + at.strftime("%b %-d, %-I:%M %p CT")
        )
        grade = grades.loc[row["game_id"]] if row["game_id"] in grades.index else None
        base = weather_ui.row_html(data)
        rows.append(base.replace("</tr>", outcome_cell(row, grade) + "</tr>", 1))
    html = html.replace(
        "<th>Model</th><th>Why</th></tr></thead>",
        "<th>Model</th><th>Why</th><th>Final result</th></tr></thead>",
        1,
    )
    html = html.replace('<tbody id="boardBody"></tbody>',
                        '<tbody id="boardBody">' + "".join(rows) + "</tbody>", 1)
    html = weather_ui.add_weather_filter_ui(html)
    html = weather_ui.add_css(html)
    html = weather_ui.add_weather_filter_js(html)
    html = weather_ui.add_sort_js(html)
    html = label_html(html, board)
    eligible = eligible_card_targets(board)
    fcs_qualifiers = int(
        (board["status"].eq("QUALIFIES") & board["division_track"].astype(str).str.upper().eq("FCS")).sum()
    )
    html = re.sub(r'<section id="card">.*?</section>',
                  card_html(eligible, fcs_qualifiers), html, count=1, flags=re.S)
    html = html.replace("CFB Weather Totals · Live Board", "CFB Weather Totals · Archived Board", 1)
    html = html.replace("Live weekly board · NWS kickoff forecasts",
                        "Archived pre-kickoff board · NWS forecasts", 1)
    html = html.replace("Current totals are scored with division-aware research tracks:",
                        "Recorded pregame totals were scored with division-aware research tracks:", 1)
    html = html.replace("Current market totals are scored with the same",
                        "Archived market totals were scored with the same", 1)
    html = html.replace("Weekly status map", "Archived weekly status map", 1)
    html = html.replace(
        '<main class="shell">',
        '<main class="shell"><div class="history-note">'
        + escape(description) +
        ' · Each game shows its last recorded pre-kickoff snapshot, which may have a different '
        'capture time than other games. A hypothetical model-side result is NOT an official play. '
        'Only entries marked official count toward the frozen prospective ledger. '
        'The map shows archived forecasts; live radar is intentionally unavailable on historical pages.'
        '</div>',
        1,
    )
    html = html.replace('href="research.html"', 'href="../research.html"')
    html = html.replace('href="fcs-research.html"', 'href="../fcs-research.html"')
    html = add_navigation(html, weeks, name, history=True)
    return html


def build_history(now: pd.Timestamp | None = None, fetch_scores: bool = True) -> dict[str, int]:
    snapshots = read_snapshots()
    if snapshots.empty:
        print("No archived pregame snapshots yet; leaving live dashboard unchanged.")
        return {}
    now = now if now is not None else pd.Timestamp.now(tz=CT)
    now_ct = pd.Timestamp(now).tz_convert(CT)
    this_week = (now_ct.normalize() - pd.Timedelta(days=now_ct.weekday())).strftime("%Y-%m-%d")
    snapshots = snapshots[snapshots["week_start"].lt(this_week)].copy()
    if snapshots.empty:
        print("No prior completed calendar weeks available.")
        return {}
    grades = read_grades()
    seasons = pd.to_numeric(snapshots["season"], errors="coerce").dropna().unique()
    scores = pd.concat([completed_results(int(s)) for s in seasons], ignore_index=True) if fetch_scores else pd.DataFrame()
    weeks: list[tuple[str, str]] = []
    for start, group in snapshots.groupby("week_start", sort=True):
        season = int(pd.to_numeric(group["season"], errors="coerce").mode().iloc[0])
        name = f"week-{season}-{start}"
        weeks.append((name, week_description(group, start)))
    DEST.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for (start, group), (name, label) in zip(snapshots.groupby("week_start", sort=True), weeks):
        board = finalized_board(group, grades, scores)
        path = DEST / f"{name}.html"
        path.write_text(render_archive(board, grades, weeks, name, label), encoding="utf-8")
        counts[name] = len(board)
    # A simple directory landing page supports a stable "Previous weeks" URL.
    (DEST / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0;url=' + weeks[-1][0] + '.html">'
        '<title>CFB Weekly History</title><body>'
        '<a href="' + weeks[-1][0] + '.html">View most recent archived week</a>'
        '</body></html>', encoding="utf-8"
    )
    live_path = ROOT / "docs/index.html"
    if live_path.exists():
        html = live_path.read_text(encoding="utf-8")
        live_path.write_text(add_navigation(html, weeks, None, history=False), encoding="utf-8")
    print(f"Built {len(counts)} historical weekly dashboards: {sum(counts.values())} game rows.")
    return counts


if __name__ == "__main__":
    build_history()
