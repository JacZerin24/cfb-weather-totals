from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .pregame_forecast_pilot import (
    AVAILABILITY_LAG_HOURS,
    LEADS,
    MAX_VALID_OFFSET_MINUTES,
    MOS_ENDPOINT,
    build_verification,
    extract_lead,
    prepare_payload,
)

AUDIT_DIR = Path("outputs/pregame_forecast_realism/station_audit")
OUTPUT_DIR = Path("outputs/pregame_forecast_realism/full_acquisition")
CACHE_PATH = Path("data/processed/pregame_forecast_full_cache.csv")
FULL_PATH = Path("data/processed/pregame_forecast_full.csv")
MAX_STATION_DISTANCE_MILES = 30.0
CORE_FIELDS = ["temperature_f", "dewpoint_f", "wind_mph", "wind_direction_degrees"]


def _bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        .fillna(False)
        .astype(bool)
    )


def fetch_game_window(
    session: requests.Session,
    station: str,
    kickoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # All three frozen lead cutoffs can be reconstructed from this bounded runtime window.
    start = kickoff - pd.Timedelta(hours=40)
    end = kickoff - pd.Timedelta(hours=8)
    params = {
        "station": station,
        "model": "NBS",
        "sts": start.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    started = time.monotonic()
    try:
        response = session.get(MOS_ENDPOINT, params=params, timeout=60)
    except Exception as exc:
        return pd.DataFrame(), {
            "station": station,
            "status": "request_exception",
            "http_status": np.nan,
            "response_bytes": 0,
            "elapsed_seconds": time.monotonic() - started,
            "rows": 0,
            "unique_runtimes": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    log: dict[str, Any] = {
        "station": station,
        "status": "ok" if response.ok else "http_error",
        "http_status": response.status_code,
        "response_bytes": len(response.content),
        "elapsed_seconds": time.monotonic() - started,
        "rows": 0,
        "unique_runtimes": 0,
        "error": "",
    }
    if not response.ok:
        log["error"] = response.text[:500]
        return pd.DataFrame(), log
    try:
        payload = response.json()
    except Exception as exc:
        log["status"] = "json_error"
        log["error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), log
    if not isinstance(payload, list):
        log["status"] = "payload_error"
        log["error"] = f"Unexpected payload type {type(payload).__name__}"
        return pd.DataFrame(), log
    frame = prepare_payload(payload)
    log["rows"] = len(frame)
    log["unique_runtimes"] = int(frame["runtime"].nunique()) if not frame.empty else 0
    if frame.empty:
        log["status"] = "empty_payload"
    return frame, log


def complete_game_ids(rows: pd.DataFrame) -> set[int]:
    if rows.empty:
        return set()
    frame = rows.copy()
    frame["game_id"] = pd.to_numeric(frame["game_id"], errors="coerce")
    frame["lead_hours"] = pd.to_numeric(frame["lead_hours"], errors="coerce")
    out: set[int] = set()
    for game_id, group in frame.dropna(subset=["game_id"]).groupby("game_id", observed=True):
        if set(group["lead_hours"].dropna().astype(int)) == set(LEADS) and len(group) == len(LEADS):
            if group["nbm_station_id"].astype(str).nunique() == 1:
                out.add(int(game_id))
    return out


def merge_cache(existing: pd.DataFrame, additions: list[dict[str, Any]]) -> pd.DataFrame:
    new = pd.DataFrame(additions)
    if existing.empty:
        combined = new
    elif new.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    if combined.empty:
        return combined
    combined["game_id"] = pd.to_numeric(combined["game_id"], errors="coerce")
    combined["lead_hours"] = pd.to_numeric(combined["lead_hours"], errors="coerce")
    combined = combined.dropna(subset=["game_id", "lead_hours"])
    combined["game_id"] = combined["game_id"].astype(int)
    combined["lead_hours"] = combined["lead_hours"].astype(int)
    combined = combined.drop_duplicates(["game_id", "lead_hours"], keep="last")
    return combined.sort_values(["season", "game_id", "lead_hours"]).reset_index(drop=True)


def verification_by_scope(rows: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    overall, _ = build_verification(rows)
    if not overall.empty:
        overall.insert(0, "scope", "overall")
        parts.append(overall)
    for season, group in rows.groupby("season", observed=True):
        result, _ = build_verification(group)
        if result.empty:
            continue
        result.insert(0, "scope", str(int(season)))
        parts.append(result)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def qc_tables(scope: pd.DataFrame, rows: pd.DataFrame, requests_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible_ids = set(pd.to_numeric(scope["game_id"], errors="coerce").dropna().astype(int))
    complete_ids = complete_game_ids(rows)
    common = rows[pd.to_numeric(rows["game_id"], errors="coerce").isin(complete_ids)].copy() if not rows.empty else pd.DataFrame()

    coverage_rows: list[dict[str, Any]] = []
    for label, group in [("overall", scope)] + [
        (str(int(season)), group) for season, group in scope.groupby("season", observed=True)
    ]:
        ids = set(pd.to_numeric(group["game_id"], errors="coerce").dropna().astype(int))
        n_complete = len(ids & complete_ids)
        coverage_rows.append(
            {
                "scope": label,
                "eligible_station_mapped_games": len(ids),
                "complete_three_lead_games": n_complete,
                "coverage_pct": n_complete / len(ids) if ids else np.nan,
                "missing_games": len(ids) - n_complete,
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    if common.empty:
        core_complete_pct = 0.0
        availability_violations = 0
        valid_offset_violations = 0
        station_distance_violations = 0
        duplicate_rows = 0
    else:
        core_complete_pct = float(common[CORE_FIELDS].notna().all(axis=1).mean())
        availability = pd.to_datetime(common["forecast_availability_utc"], utc=True, errors="coerce")
        cutoff = pd.to_datetime(common["decision_cutoff_utc"], utc=True, errors="coerce")
        availability_violations = int((availability > cutoff).fillna(True).sum())
        offset = pd.to_numeric(common["forecast_valid_offset_minutes"], errors="coerce")
        valid_offset_violations = int((offset.abs() > MAX_VALID_OFFSET_MINUTES).fillna(True).sum())
        distance = pd.to_numeric(common["station_distance_miles"], errors="coerce")
        station_distance_violations = int((distance > MAX_STATION_DISTANCE_MILES).fillna(True).sum())
        duplicate_rows = int(common.duplicated(["game_id", "lead_hours"]).sum())

    overall_pct = float(coverage.loc[coverage["scope"].eq("overall"), "coverage_pct"].iloc[0])
    season_rows = coverage[~coverage["scope"].eq("overall")]
    min_season_pct = float(season_rows["coverage_pct"].min()) if not season_rows.empty else 0.0
    request_failures = 0
    if not requests_log.empty and "status" in requests_log.columns:
        request_failures = int((requests_log["status"].astype(str) != "ok").sum())

    ready = bool(
        overall_pct >= 0.95
        and min_season_pct >= 0.90
        and core_complete_pct >= 0.99
        and availability_violations == 0
        and valid_offset_violations == 0
        and station_distance_violations == 0
        and duplicate_rows == 0
    )
    validation = pd.DataFrame(
        [
            {
                "eligible_station_mapped_games": len(eligible_ids),
                "complete_three_lead_games": len(complete_ids & eligible_ids),
                "overall_coverage_pct": overall_pct,
                "minimum_season_coverage_pct": min_season_pct,
                "complete_lead_rows": len(common),
                "core_continuous_complete_pct": core_complete_pct,
                "availability_violations": availability_violations,
                "valid_offset_violations": valid_offset_violations,
                "station_distance_violations": station_distance_violations,
                "duplicate_game_lead_rows": duplicate_rows,
                "request_attempt_failures": request_failures,
                "ready_for_modeling": ready,
            }
        ]
    )
    return coverage, validation, common


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire leak-safe archived NBM pregame forecasts.")
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--request-delay", type=float, default=0.10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    scope = pd.read_csv(AUDIT_DIR / "research_scope.csv", low_memory=False)
    mapping = pd.read_csv(AUDIT_DIR / "venue_season_station_mapping.csv", low_memory=False)
    candidates = pd.read_csv(AUDIT_DIR / "venue_season_station_candidates.csv", low_memory=False)
    scope["kickoff_utc"] = pd.to_datetime(scope["kickoff_utc"], utc=True, errors="coerce")
    mapping["station_mapped"] = _bool_series(mapping["station_mapped"])
    candidates["within_30_miles"] = _bool_series(candidates["within_30_miles"])

    scope = scope.merge(
        mapping[["season", "venue_id", "station_mapped"]],
        on=["season", "venue_id"],
        how="left",
        validate="many_to_one",
    )
    scope = scope[scope["station_mapped"].fillna(False)].copy()
    scope["game_id"] = pd.to_numeric(scope["game_id"], errors="coerce").astype("Int64")
    scope = scope.dropna(subset=["game_id", "kickoff_utc"]).copy()
    scope["game_id"] = scope["game_id"].astype(int)
    scope = scope.drop_duplicates("game_id").sort_values(["season", "kickoff_utc", "game_id"]).reset_index(drop=True)

    existing = pd.read_csv(CACHE_PATH, low_memory=False) if CACHE_PATH.exists() else pd.DataFrame()
    starting_complete = complete_game_ids(existing)
    remaining = scope[~scope["game_id"].isin(starting_complete)].head(args.max_games).copy()

    print(
        f"Full NBM acquisition start: eligible={len(scope):,}; "
        f"already_complete={len(starting_complete):,}; selected={len(remaining):,}"
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "cfb-weather-totals archived NBM forecast realism research"})
    additions: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    game_rows: list[dict[str, Any]] = []

    for position, (_, game) in enumerate(remaining.iterrows(), start=1):
        game_candidates = candidates[
            candidates["season"].eq(game["season"])
            & candidates["venue_id"].eq(game["venue_id"])
            & candidates["within_30_miles"]
        ].sort_values("candidate_rank")

        success = False
        selected_station = ""
        selected_rank = np.nan
        selected_distance = np.nan
        last_error = ""
        attempts = 0

        for _, candidate in game_candidates.iterrows():
            attempts += 1
            station = str(candidate["nbm_station_id"]).strip().upper()
            frame, req = fetch_game_window(session, station, pd.Timestamp(game["kickoff_utc"]))
            req.update(
                {
                    "game_id": int(game["game_id"]),
                    "season": int(game["season"]),
                    "venue_id": game["venue_id"],
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "station_distance_miles": float(candidate["distance_miles"]),
                }
            )
            request_rows.append(req)
            if frame.empty:
                last_error = str(req.get("error") or req.get("status") or "empty payload")
                time.sleep(args.request_delay)
                continue

            try:
                extracted = [extract_lead(frame, pd.Timestamp(game["kickoff_utc"]), lead) for lead in LEADS]
            except Exception as exc:
                last_error = f"extract: {type(exc).__name__}: {exc}"
                time.sleep(args.request_delay)
                continue
            if not all(item is not None for item in extracted):
                last_error = "station did not provide all three frozen lead reconstructions"
                time.sleep(args.request_delay)
                continue

            selected_station = station
            selected_rank = int(candidate["candidate_rank"])
            selected_distance = float(candidate["distance_miles"])
            hist = {
                "hist_temperature_f": game.get("temperature_f"),
                "hist_dewpoint_f": game.get("dewpoint_f"),
                "hist_humidity": game.get("humidity"),
                "hist_wind_mph": game.get("wind_mph"),
                "hist_wind_direction_degrees": game.get("wind_direction_degrees"),
                "hist_precipitation": game.get("precipitation"),
                "hist_snowfall": game.get("snowfall"),
            }
            for item in extracted:
                assert item is not None
                additions.append(
                    {
                        "game_id": int(game["game_id"]),
                        "season": int(game["season"]),
                        "kickoff_utc": game["kickoff_utc"],
                        "venue_id": game["venue_id"],
                        "venue_name": game.get("venue_name"),
                        "venue_latitude": game.get("venue_latitude"),
                        "venue_longitude": game.get("venue_longitude"),
                        "nbm_station_id": station,
                        "station_candidate_rank": selected_rank,
                        "station_distance_miles": selected_distance,
                        **item,
                        **hist,
                    }
                )
            success = True
            break

        game_rows.append(
            {
                "game_id": int(game["game_id"]),
                "season": int(game["season"]),
                "venue_id": game["venue_id"],
                "status": "ok" if success else "incomplete",
                "selected_station": selected_station,
                "selected_candidate_rank": selected_rank,
                "selected_station_distance_miles": selected_distance,
                "candidate_attempts": attempts,
                "last_error": last_error,
            }
        )

        if position % 25 == 0 or position == len(remaining):
            print(
                f"[{position:,}/{len(remaining):,}] game={int(game['game_id'])} "
                f"status={'ok' if success else 'incomplete'} station={selected_station or '-'} attempts={attempts}"
            )
        if args.checkpoint_every > 0 and position % args.checkpoint_every == 0:
            checkpoint = merge_cache(existing, additions)
            checkpoint.to_csv(CACHE_PATH, index=False)
        time.sleep(args.request_delay)

    combined = merge_cache(existing, additions)
    combined.to_csv(CACHE_PATH, index=False)
    combined.to_csv(FULL_PATH, index=False)
    requests_log = pd.DataFrame(request_rows)
    game_status = pd.DataFrame(game_rows)
    requests_log.to_csv(OUTPUT_DIR / "request_log.csv", index=False)
    game_status.to_csv(OUTPUT_DIR / "game_attempt_status.csv", index=False)

    coverage, validation, common = qc_tables(scope, combined, requests_log)
    coverage.to_csv(OUTPUT_DIR / "coverage.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    common.to_csv(OUTPUT_DIR / "common_three_lead_forecasts.csv", index=False)

    verification = verification_by_scope(common)
    verification.to_csv(OUTPUT_DIR / "verification_metrics.csv", index=False)

    end_complete = complete_game_ids(combined)
    progress = pd.DataFrame(
        [
            {
                "eligible_station_mapped_games": len(scope),
                "starting_complete_games": len(starting_complete),
                "ending_complete_games": len(end_complete),
                "newly_complete_games": len(end_complete - starting_complete),
                "remaining_incomplete_games": len(set(scope["game_id"]) - end_complete),
                "selected_games_this_run": len(remaining),
                "http_request_attempts_this_run": len(requests_log),
                "failed_request_attempts_this_run": int((requests_log.get("status", pd.Series(dtype=str)).astype(str) != "ok").sum()) if not requests_log.empty else 0,
                "successful_games_this_run": int((game_status.get("status", pd.Series(dtype=str)).astype(str) == "ok").sum()) if not game_status.empty else 0,
                "ready_for_modeling": bool(validation.iloc[0]["ready_for_modeling"]),
            }
        ]
    )
    progress.to_csv(OUTPUT_DIR / "progress.csv", index=False)

    print("--- acquisition progress ---")
    print(progress.to_string(index=False))
    print("--- coverage ---")
    print(coverage.to_string(index=False))
    print("--- validation ---")
    print(validation.to_string(index=False))
    if not verification.empty:
        print("--- verification overall ---")
        print(verification[verification["scope"].eq("overall")].to_string(index=False))

    all_attempted = len(remaining) >= len(scope) - len(starting_complete)
    if args.strict and all_attempted and not bool(validation.iloc[0]["ready_for_modeling"]):
        raise SystemExit("ERROR: Full NBM acquisition did not pass the frozen model-readiness QC gate.")


if __name__ == "__main__":
    main()
