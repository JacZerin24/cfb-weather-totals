from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.fcs_model import classify_division_row, score_fcs_rows
from src.predict_week import add_live_categories, fetch_nws_weather

ROOT = Path(__file__).resolve().parents[1]
GAME_ID = "401866625"
NEW_KICKOFF_UTC = pd.Timestamp("2026-09-06T15:00:00Z")  # Sunday 10:00 AM CT / 11:00 AM ET
TICKET_TOTAL = 67.5

WEATHER_COLUMNS = [
    "temperature_f",
    "dewpoint_f",
    "humidity",
    "wind_mph",
    "wind_gust_mph",
    "precip_probability_pct",
    "precipitation",
    "snowfall",
    "pressure",
    "weather_summary",
    "nws_status",
    "nws_office",
]


def n(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def main() -> None:
    board_path = ROOT / "outputs" / "weekly_board.csv"
    board = pd.read_csv(board_path, dtype={"game_id": str})
    match = board[board["game_id"].astype(str).eq(GAME_ID)].copy()
    if len(match) != 1:
        raise RuntimeError(f"Expected exactly one Campbell game row for {GAME_ID}; found {len(match)}")

    row = match.iloc[[0]].copy()
    stored = row.iloc[0]

    # These context fields were used by the original FCS model but are not part of
    # weekly_board.csv's display schema. This game is a non-neutral, non-conference game.
    row["neutral_site"] = False
    row["conference_game"] = False
    row["game_indoors_bool"] = False
    row["start_time_tbd"] = False
    row["division_track"] = "FCS"

    # Reproduce the original stored score first as an environment/integrity check.
    original_prepared = add_live_categories(row.copy())
    original_rescored = score_fcs_rows(original_prepared).iloc[0]

    # Ticket-line diagnostic: preserve the user's already-bet 67.5 line while changing
    # only the kickoff time/weather context to the rescheduled Sunday game.
    fresh = row.copy()
    fresh["start_date"] = NEW_KICKOFF_UTC
    fresh["closing_total"] = TICKET_TOTAL
    fresh["line_provider"] = str(stored.get("line_provider") or "DraftKings")

    for col in WEATHER_COLUMNS:
        if col in fresh.columns:
            fresh[col] = np.nan

    weather = fetch_nws_weather(fresh)
    if weather.empty:
        raise RuntimeError("NWS kickoff weather returned no row.")
    weather_row = weather.iloc[0]
    for col in weather.columns:
        if col != "game_id":
            fresh[col] = weather_row[col]

    fresh = add_live_categories(fresh)
    scored = score_fcs_rows(fresh).iloc[0]
    status, reason = classify_division_row(scored)

    result = {
        "game_id": GAME_ID,
        "matchup": "Western Carolina @ Campbell",
        "diagnostic_type": "postponement_ticket_line_rescore",
        "official_record_unchanged": True,
        "new_kickoff_utc": NEW_KICKOFF_UTC.isoformat(),
        "ticket_total": TICKET_TOTAL,
        "line_provider_anchor": str(stored.get("line_provider") or "DraftKings"),
        "stored_original": {
            "kickoff_utc": n(stored.get("start_date")),
            "pred_market_residual": n(stored.get("pred_market_residual")),
            "projected_total": n(stored.get("model_projected_total")),
            "status": n(stored.get("status")),
        },
        "original_reproduction_check": {
            "pred_market_residual": n(original_rescored.get("pred_market_residual")),
            "projected_total": n(original_rescored.get("model_projected_total")),
            "absolute_residual_difference_vs_stored": abs(
                float(original_rescored.get("pred_market_residual")) - float(stored.get("pred_market_residual"))
            ),
        },
        "fresh_nws_weather": {
            "temperature_f": n(scored.get("temperature_f")),
            "dewpoint_f": n(scored.get("dewpoint_f")),
            "humidity_pct": n(scored.get("humidity")),
            "wind_mph": n(scored.get("wind_mph")),
            "wind_gust_mph": n(scored.get("wind_gust_mph")),
            "pop_pct": n(scored.get("precip_probability_pct")),
            "precipitation_in": n(scored.get("precipitation")),
            "snowfall_in": n(scored.get("snowfall")),
            "nws_status": n(scored.get("nws_status")),
            "nws_office": n(scored.get("nws_office")),
            "weather_summary": n(scored.get("weather_summary")),
        },
        "fresh_ticket_line_rescore": {
            "pred_market_residual": n(scored.get("pred_market_residual")),
            "projected_total": n(scored.get("model_projected_total")),
            "model_side": n(scored.get("model_side")),
            "abs_edge": n(scored.get("abs_pred_edge")),
            "classification_if_scored_fresh": status,
            "classification_reason": reason,
        },
    }

    out_dir = ROOT / "outputs" / "postponement_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "campbell_20260906.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("CAMPBELL_DIAGNOSTIC_JSON=" + json.dumps(result, sort_keys=True))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

# This comment intentionally triggers the isolated diagnostic workflow after workflow registration.
