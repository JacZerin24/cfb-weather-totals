"""Website-only same-week final score overlay and read-only official verification.

Runs exclusively in GitHub Pages deployment. Reads immutable archives and CFBD
final scores, computes paper grading IN MEMORY, and changes only docs/index.html.
It does not run or modify any operational model or write to outputs/.
"""
from __future__ import annotations

import json
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from . import build_week_history as history
from .prospective_ledger import (
    SNAPSHOT_DIR, _read_immutable, grade_official_entries, load_protocol,
    select_official_entries,
)
from .prospective_integrity_rebuild import (
    apply_grading_exclusions, apply_official_eligibility_corrections,
)
from .utils import ROOT

CURRENT = ROOT / "docs/index.html"
BOARD = ROOT / "outputs/weekly_board.csv"
START = "<!-- CURRENT_WEEK_FINAL_RESULTS_START -->"
END = "<!-- CURRENT_WEEK_FINAL_RESULTS_END -->"


def current_calendar_start(now: pd.Timestamp | None = None) -> str:
    current = pd.Timestamp.now(tz=history.CT) if now is None else pd.Timestamp(now).tz_convert(history.CT)
    day = current.date()
    return (day - timedelta(days=day.weekday())).isoformat()


def read_only_grade(scores: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen official selection and grading logic without writing ledgers."""
    raw = _read_immutable(SNAPSHOT_DIR)
    if raw.empty:
        return pd.DataFrame()
    protocol = load_protocol()
    corrected = apply_official_eligibility_corrections(raw, protocol)
    official = select_official_entries(corrected, protocol)
    if official.empty:
        return pd.DataFrame()
    official["game_id"] = pd.to_numeric(official["game_id"], errors="coerce").astype("Int64")
    finals = scores.copy()
    if not finals.empty:
        finals["game_id"] = pd.to_numeric(finals["game_id"], errors="coerce").astype("Int64")
        finals = finals[finals["game_id"].notna()].copy()
    grades = grade_official_entries(official, pd.DataFrame(), finals, protocol)
    grades = apply_grading_exclusions(grades, protocol)
    grades["game_id"] = grades["game_id"].map(history.game_key)
    return grades[grades["game_id"].ne("")].drop_duplicates("game_id", keep="last").set_index("game_id")


def score_index(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    result = scores.copy()
    result["game_id"] = result["game_id"].map(history.game_key)
    return result.drop_duplicates("game_id").set_index("game_id")


def apply_current_table(
    html: str, board: pd.DataFrame, snapshots: pd.DataFrame,
    scores: pd.DataFrame, grades: pd.DataFrame,
) -> tuple[str, int]:
    """Add a single extra final-result column without rebuilding the live board."""
    if START in html:
        return html, 0
    original_header = "<th>Model</th><th>Why</th></tr></thead>"
    if original_header not in html:
        raise RuntimeError("Cannot find current production board headers; refusing to alter page.")
    html = html.replace(
        original_header,
        "<th>Model</th><th>Why</th><th>Final result</th></tr></thead>", 1,
    )
    if snapshots.empty:
        index = pd.DataFrame()
    else:
        index = snapshots.copy()
        index["game_id"] = index["game_id"].map(history.game_key)
        index = index.drop_duplicates("game_id", keep="last").set_index("game_id")
    valid_scores = score_index(scores)
    valid_grades = grades
    rendered = 0
    finals = 0
    def update_body(match: re.Match[str]) -> str:
        nonlocal rendered, finals
        tbody = match.group(1)
        def update_row(rowmatch: re.Match[str]) -> str:
            nonlocal rendered, finals
            row_html = rowmatch.group(0)
            if rendered >= len(board):
                raise RuntimeError("Found more displayed game rows than the live source board.")
            record = board.iloc[rendered]
            rendered += 1
            game_id = history.game_key(record.get("game_id"))
            snapshot = index.loc[game_id].copy() if game_id in index.index else record.copy()
            # Never compute a hypothetical result using a postgame / nonarchived model.
            if game_id not in index.index:
                snapshot["closing_total"] = pd.NA
                snapshot["model_side"] = ""
                snapshot["pred_market_residual"] = pd.NA
            if game_id in valid_scores.index:
                result = valid_scores.loc[game_id]
                for col in ("final_away_points", "final_home_points", "actual_total_points"):
                    snapshot[col] = result.get(col)
                finals += 1
            else:
                for col in ("final_away_points", "final_home_points", "actual_total_points"):
                    snapshot[col] = pd.NA
                # A verified previously graded final may be shown during API failure.
                if game_id in valid_grades.index:
                    cached = valid_grades.loc[game_id]
                    if history.value_or_none(cached.get("actual_total_points")) is not None:
                        for col in ("final_away_points", "final_home_points", "actual_total_points"):
                            snapshot[col] = cached.get(col)
                        finals += 1
            grade = valid_grades.loc[game_id] if game_id in valid_grades.index else None
            return row_html.replace("</tr>", history.outcome_cell(snapshot, grade) + "</tr>", 1)
        updated = re.sub(r'<tr data-status="[^"]+"[^>]*>.*?</tr>', update_row, tbody, flags=re.S)
        return '<tbody id="boardBody">' + updated + "</tbody>"
    html, count = re.subn(r'<tbody id="boardBody">(.*?)</tbody>', update_body, html, count=1, flags=re.S)
    if count != 1 or rendered != len(board):
        raise RuntimeError(f"Live board row count mismatch: HTML rows={rendered}, CSV rows={len(board)}.")
    # Do not repeat current-week week review banner on re-render.
    note = (
        f'{START}<div class="shell" style="padding:8px 0 0;color:var(--muted);font-size:12px">'
        f'Final scores: {finals} of {len(board)} displayed games verified. '
        'Archived pregame predictions and official paper entries are labeled separately.'
        f'</div>{END}'
    )
    html = html.replace('<main class="shell">',note+'<main class="shell">',1)
    return html, finals


def week_verification_summary(grades: pd.DataFrame, start: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "week_start_ct": start, "official_qualifiers": 0,
        "wins": 0, "losses": 0, "pushes": 0,
        "excluded": 0, "pending": 0, "verified_final_games": 0,
        "net_units_1u_at_-110": 0.0, "by_track": {},
    }
    if grades.empty:
        return result
    current = grades.copy()
    kickoff = pd.to_datetime(current["start_date"], utc=True, errors="coerce").dt.tz_convert(history.CT)
    week = kickoff.dt.date.map(
        lambda day: (day - timedelta(days=day.weekday())).isoformat() if pd.notna(day) else ""
    )
    current = current[week.eq(start)].copy()
    qualifiers = current[current["paper_qualifier"].map(history.is_yes)].copy()
    result["official_qualifiers"] = len(qualifiers)
    result["verified_final_games"] = int(current.get("actual_total_points", pd.Series(index=current.index, dtype=float)).notna().sum())
    for outcome in ("win", "loss", "push", "excluded"):
        result[{"win":"wins","loss":"losses","push":"pushes","excluded":"excluded"}[outcome]] = int(
            qualifiers["paper_result"].eq(outcome).sum()
        )
    result["pending"] = int(
        (~qualifiers["paper_result"].isin(["win", "loss", "push", "excluded"])).sum()
    )
    # The official frozen protocol uses 1u risk at -110, not realized book returns.
    result["net_units_1u_at_-110"] = round(
        float(pd.to_numeric(qualifiers["paper_units_1u"], errors="coerce").sum()), 4
    )
    if "model_track" in qualifiers:
        for name, group in qualifiers.groupby("model_track", dropna=False):
            result["by_track"][str(name)] = {
                "wins": int(group["paper_result"].eq("win").sum()),
                "losses": int(group["paper_result"].eq("loss").sum()),
                "pushes": int(group["paper_result"].eq("push").sum()),
                "pending": int((~group["paper_result"].isin(["win","loss","push","excluded"])).sum()),
            }
    return result


def publish_current_week_results(
    now: pd.Timestamp | None = None,
    scores: pd.DataFrame | None = None,
    grades: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if not CURRENT.exists() or not BOARD.exists():
        raise RuntimeError("Live site and board files required for read-only results overlay.")
    board = pd.read_csv(BOARD)
    if board.empty:
        raise RuntimeError("Live board has no games.")
    season = int(pd.to_numeric(board["season"], errors="coerce").dropna().mode().iloc[0])
    finals = history.completed_results(season) if scores is None else scores
    snapshots = history.read_snapshots()
    if grades is None:
        current_grades = read_only_grade(finals)
    else:
        current_grades = grades
    html = CURRENT.read_text(encoding="utf-8")
    new, final_count = apply_current_table(html, board, snapshots, finals, current_grades)
    CURRENT.write_text(new, encoding="utf-8")
    week_start = current_calendar_start(now)
    summary = week_verification_summary(current_grades, week_start)
    summary["live_board_games"] = len(board)
    summary["live_board_verified_finals"] = final_count
    summary["verified_as_of_ct"] = pd.Timestamp.now(tz=history.CT).isoformat()
    dest = CURRENT.parent / "history/current-week-review.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("READ-ONLY CURRENT WEEK VERIFICATION: " + json.dumps(summary, sort_keys=True))
    return summary


if __name__ == "__main__":
    publish_current_week_results()
