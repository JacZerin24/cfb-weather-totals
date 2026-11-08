"""Offline test: Sunday finals appear on the CURRENT week, not only in history."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from . import preview_week_results as preview
from . import build_week_history as history


def main() -> None:
    with TemporaryDirectory() as d:
        root = Path(d)
        (root / "docs").mkdir()
        (root / "outputs").mkdir()
        (root / "docs/index.html").write_text(
            '<html><head><style></style></head><body>'
            '<nav></nav><main class="shell">'
            '<section><table><thead><tr><th>Status</th><th>Game</th>'
            '<th>Model</th><th>Why</th></tr></thead>'
            '<tbody id="boardBody">'
            '<tr data-status="QUALIFIES" data-division="FBS"><td>QUALIFIES</td></tr>'
            '<tr data-status="NO PLAY" data-division="FBS"><td>NO PLAY</td></tr>'
            '</tbody></table></section></main></body></html>', encoding="utf-8",
        )
        rows = pd.DataFrame([
            {"game_id": 1234, "season": 2026, "start_date": "2026-11-07T20:00:00Z",
             "status": "QUALIFIES", "model_side": "under", "closing_total": 56.5},
            {"game_id": 2345, "season": 2026, "start_date": "2026-11-07T21:00:00Z",
             "status": "NO PLAY", "model_side": "over", "closing_total": 44.5},
        ])
        rows.to_csv(root / "outputs/weekly_board.csv", index=False)
        snap = rows.iloc[[0]].copy()
        snap["game_id"] = snap["game_id"].map(history.game_key)
        scores = pd.DataFrame([{
            "game_id": "1234", "final_away_points": 17,
            "final_home_points": 20, "actual_total_points": 37,
        }])
        grades = pd.DataFrame([{
            "game_id": "1234", "start_date": "2026-11-07T20:00:00Z",
            "paper_qualifier": True, "paper_result": "win",
            "paper_units_1u": 100/110, "model_track": "GENERAL HGB",
            "closing_total": 56.5, "actual_total_points": 37,
        }]).set_index("game_id")
        html = (root / "docs/index.html").read_text(encoding="utf-8")
        updated, count = preview.apply_current_table(html, rows, snap, scores, grades)
        assert count == 1
        assert updated.count("<th>Final result</th>") == 1
        assert updated.count('class="history-result"') == 2
        assert "17–20" in updated
        assert "Total: <strong>37</strong>" in updated
        assert "Snapshot UNDER 56.5" in updated
        assert "Official paper entry: WIN" in updated
        assert "Final not yet verified" in updated
        assert "Snapshot OVER 44.5" not in updated
        assert preview.current_calendar_start(
            pd.Timestamp("2026-11-08T12:00:00Z")
        ) == "2026-11-02"
        summary = preview.week_verification_summary(grades, "2026-11-02")
        assert summary["wins"] == 1 and summary["official_qualifiers"] == 1
        assert summary["net_units_1u_at_-110"] > 0.9
        assert summary["pending"] == 0
        assert preview.week_verification_summary(grades, "2026-11-09")["wins"] == 0
    print("Sunday current-week results self-test passed: final, pending, official, hypothetical, CT week.")


if __name__ == "__main__":
    main()
