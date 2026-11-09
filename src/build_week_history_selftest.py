"""Offline checks for historical weekly presentation and provenance."""
from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from . import build_week_history as history


def sample_row(at: str, projected: float) -> dict:
    return {
        "record_kind": "board_snapshot", "snapshot_timestamp_utc": at,
        "game_id": 401234567, "start_date": "2026-09-05T20:00:00Z",
        "season": 2026, "week": 1, "status": "QUALIFIES",
        "away_team": "Away Test", "home_team": "Home Test",
        "decision_reason": "Archived screen", "research_tags": "",
        "division_track": "FBS", "model_track": "GENERAL HGB",
        "venue_name": "Test Stadium", "venue_city": "Test", "venue_state": "TX",
        "venue_latitude": 31.0, "venue_longitude": -97.0, "game_indoors": False,
        "closing_total": 56.5, "line_provider": "DraftKings", "line_provider_count": 1,
        "model_projected_total": projected,
        "pred_market_residual": projected - 56.5,
        "abs_pred_edge": abs(projected - 56.5), "model_side": "under",
        "temperature_f": 80, "wind_mph": 4, "wind_gust_mph": 8,
        "humidity": 60, "precip_probability_pct": 5,
        "nws_status": "ok", "weather_summary": "Dry",
    }


def main() -> None:
    old = history.SOURCE, history.DEST, history.ROOT
    try:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            history.ROOT = base
            history.SOURCE = base / "outputs/prospective/2026"
            history.DEST = base / "docs/history"
            snap = history.SOURCE / "snapshots"
            snap.mkdir(parents=True)
            (base / "docs").mkdir(parents=True)
            pd.DataFrame([
                sample_row("2026-09-05T08:00:00Z", 42.0),
                sample_row("2026-09-05T16:00:00Z", 41.0),
                sample_row("2026-09-05T21:00:00Z", 12.0),  # postgame; never select
            ]).to_csv(snap / "board_fixture.csv", index=False)
            pd.DataFrame([{
                "game_id": 401234567, "paper_qualifier": True,
                "paper_result": "win", "closing_total": 56.5,
                "final_away_points": 27, "final_home_points": 17,
                "actual_total_points": 44,
            }]).to_csv(history.SOURCE / "graded_entries.csv", index=False)
            (base / "docs/index.html").write_text(
                '<html><head><style></style></head><body>'
                '<header><h1>CFB Weather Totals</h1></header><nav>'
                '<a href="#board">Full board</a></nav></body></html>',
                encoding="utf-8",
            )
            counts = history.build_history(
                now=pd.Timestamp("2026-09-14T12:00:00Z"), fetch_scores=False
            )
            assert sum(counts.values()) == 1, counts
            pages = list(history.DEST.glob("week-*.html"))
            assert len(pages) == 1
            html = pages[0].read_text(encoding="utf-8")
            assert re.search(r'<tr data-status="QUALIFIES"', html)
            assert "Final result" in html
            assert "27–17" in html
            assert "Total: <strong>44</strong>" in html
            assert "Official paper entry: WIN" in html
            assert "projected total 41.0" in html, "Must select latest PREgame model"
            assert "projected total 12.0" not in html, "Postgame forecasts are leakage"
            assert "Last pregame snapshot:" in html
            assert 'href="../research.html"' in html
            assert 'data-weather-sort="8"' in html
            assert '<option value="../index.html"' in html
            assert "Latest recorded" not in html

            # Running the exporter repeatedly must not accumulate navigation links.
            history.build_history(
                now=pd.Timestamp("2026-09-14T12:00:00Z"), fetch_scores=False
            )
            live = (base / "docs/index.html").read_text(encoding="utf-8")
            assert live.count(history.NAV_START) == 1
            assert live.count('href="history/"') == 1
            assert live.index(history.NAV_START) > live.index("<h1>CFB Weather Totals</h1>")
            assert live.index(history.NAV_START) < live.index("</header>")
            assert "Browse weeks" in live
            assert "Previous week: Week 1" in live
            assert 'aria-label="Select college football week"' in live

            # An early Sunday refresh can display NEXT week's board already.
            # In that case Saturday's completed games must be visible in history
            # on Sunday, without waiting for the Monday calendar rollover.
            pd.DataFrame([{
                "start_date": "2026-09-12T20:00:00Z",
            }]).to_csv(base / "outputs/weekly_board.csv", index=False)
            assert history.live_board_week_start() == "2026-09-07"
            sunday = history.build_history(
                now=pd.Timestamp("2026-09-06T12:00:00Z"), fetch_scores=False
            )
            assert sum(sunday.values()) == 1, sunday
    finally:
        history.SOURCE, history.DEST, history.ROOT = old
    print("Historical week viewer self-test passed: pregame snapshot, final result, "
          "official outcome, filters and navigation.")


if __name__ == "__main__":
    main()
