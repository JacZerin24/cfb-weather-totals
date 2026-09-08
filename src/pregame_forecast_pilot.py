from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

MOS_ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
AUDIT_DIR = Path("outputs/pregame_forecast_realism/station_audit")
OUTPUT_DIR = Path("outputs/pregame_forecast_realism/pilot")
AVAILABILITY_LAG_HOURS = 2.0
LEADS = [24, 12, 6]
MAX_VALID_OFFSET_MINUTES = 90.0
KNOT_TO_MPH = 1.150779448


def relative_humidity_from_temp_dewpoint(temp_f: Any, dew_f: Any) -> float:
    try:
        tc = (float(temp_f) - 32.0) * 5.0 / 9.0
        dc = (float(dew_f) - 32.0) * 5.0 / 9.0
    except (TypeError, ValueError):
        return np.nan
    a, b = 17.625, 243.04
    rh = 100.0 * math.exp((a * dc / (b + dc)) - (a * tc / (b + tc)))
    return float(np.clip(rh, 0.0, 100.0))


def circular_abs_error_deg(forecast: Any, observed: Any) -> float:
    try:
        a = float(forecast) % 360.0
        b = float(observed) % 360.0
    except (TypeError, ValueError):
        return np.nan
    return abs((a - b + 180.0) % 360.0 - 180.0)


def prepare_payload(payload: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    frame["runtime"] = pd.to_datetime(frame.get("runtime"), utc=True, errors="coerce")
    frame["ftime"] = pd.to_datetime(frame.get("ftime"), utc=True, errors="coerce")
    for col in ["tmp", "dpt", "wdr", "wsp", "gst", "p06", "q06", "s06"]:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["runtime", "ftime"]).copy()


def select_run_and_kickoff_row(
    frame: pd.DataFrame,
    kickoff: pd.Timestamp,
    lead_hours: int,
    availability_lag_hours: float = AVAILABILITY_LAG_HOURS,
    max_valid_offset_minutes: float = MAX_VALID_OFFSET_MINUTES,
) -> tuple[pd.Series | None, pd.DataFrame | None, dict[str, Any]]:
    cutoff = kickoff - pd.Timedelta(hours=lead_hours)
    max_runtime = cutoff - pd.Timedelta(hours=availability_lag_hours)
    eligible_runtimes = sorted(frame.loc[frame["runtime"] <= max_runtime, "runtime"].dropna().unique(), reverse=True)

    detail = {
        "decision_cutoff_utc": cutoff,
        "max_eligible_runtime_utc": max_runtime,
        "eligible_runtime_count": len(eligible_runtimes),
    }
    for runtime in eligible_runtimes:
        run = frame[frame["runtime"].eq(runtime)].copy()
        run["valid_offset_minutes"] = (run["ftime"] - kickoff).abs().dt.total_seconds() / 60.0
        candidates = run[
            run[["tmp", "dpt", "wsp", "wdr"]].notna().all(axis=1)
            & (run["valid_offset_minutes"] <= max_valid_offset_minutes)
        ].sort_values(["valid_offset_minutes", "ftime"])
        if candidates.empty:
            continue
        selected = candidates.iloc[0].copy()
        return selected, run, detail
    return None, None, detail


def select_six_hour_period(run: pd.DataFrame, kickoff: pd.Timestamp) -> pd.Series | None:
    period = run[run[["p06", "q06", "s06"]].notna().any(axis=1)].copy()
    if period.empty:
        return None
    # NBS P06/Q06/S06 are six-hour period fields reported at the period endpoint.
    # Select the first endpoint at/after kickoff, no more than six hours later.
    period["hours_after_kickoff"] = (period["ftime"] - kickoff).dt.total_seconds() / 3600.0
    covering = period[(period["hours_after_kickoff"] >= 0) & (period["hours_after_kickoff"] <= 6.0)]
    if covering.empty:
        return None
    return covering.sort_values(["ftime"]).iloc[0]


def extract_lead(
    frame: pd.DataFrame,
    kickoff: pd.Timestamp,
    lead_hours: int,
) -> dict[str, Any] | None:
    selected, run, detail = select_run_and_kickoff_row(frame, kickoff, lead_hours)
    if selected is None or run is None:
        return None

    runtime = pd.Timestamp(selected["runtime"])
    ftime = pd.Timestamp(selected["ftime"])
    cutoff = pd.Timestamp(detail["decision_cutoff_utc"])
    availability = runtime + pd.Timedelta(hours=AVAILABILITY_LAG_HOURS)
    if availability > cutoff:
        raise RuntimeError("Leak guard violated: selected NBM run was not eligible by decision cutoff.")

    period = select_six_hour_period(run, kickoff)
    p06 = float(period["p06"]) if period is not None and pd.notna(period["p06"]) else np.nan
    q06 = float(period["q06"]) if period is not None and pd.notna(period["q06"]) else np.nan
    s06 = float(period["s06"]) if period is not None and pd.notna(period["s06"]) else np.nan

    temp_f = float(selected["tmp"])
    dew_f = float(selected["dpt"])
    wind_mph = float(selected["wsp"]) * KNOT_TO_MPH
    gust_mph = float(selected["gst"]) * KNOT_TO_MPH if pd.notna(selected["gst"]) else np.nan
    direction = float(selected["wdr"])

    return {
        "lead_hours": lead_hours,
        "decision_cutoff_utc": cutoff,
        "forecast_runtime_utc": runtime,
        "forecast_availability_utc": availability,
        "forecast_valid_time_utc": ftime,
        "forecast_runtime_age_at_cutoff_hours": (cutoff - runtime).total_seconds() / 3600.0,
        "forecast_age_at_kickoff_hours": (kickoff - runtime).total_seconds() / 3600.0,
        "forecast_valid_offset_minutes": (ftime - kickoff).total_seconds() / 60.0,
        "temperature_f": temp_f,
        "dewpoint_f": dew_f,
        "humidity": relative_humidity_from_temp_dewpoint(temp_f, dew_f),
        "wind_mph": wind_mph,
        "wind_direction_degrees": direction,
        "wind_gust_mph": gust_mph,
        "precip_probability_pct": p06,
        "precipitation": q06 / 100.0 if pd.notna(q06) else np.nan,
        "snowfall": s06 / 10.0 if pd.notna(s06) else np.nan,
        "precip_period_end_utc": pd.Timestamp(period["ftime"]) if period is not None else pd.NaT,
    }


def stable_geographic_pilot(scope: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    merged = scope.merge(
        mapping[["season", "venue_id", "station_mapped", "distance_miles"]],
        on=["season", "venue_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_map"),
    )
    merged = merged[merged["station_mapped"].fillna(False)].copy()
    selected_parts: list[pd.DataFrame] = []
    for season in [2019, 2021, 2023, 2025]:
        group = merged[merged["season"].eq(season)].copy()
        group["lat_bin"] = pd.qcut(group["venue_latitude"], q=4, labels=False, duplicates="drop")
        group["lon_bin"] = pd.qcut(group["venue_longitude"], q=4, labels=False, duplicates="drop")
        sample = (
            group.sort_values(["lat_bin", "lon_bin", "kickoff_utc", "game_id"])
            .groupby(["lat_bin", "lon_bin"], observed=True, as_index=False)
            .head(1)
        )
        selected_parts.append(sample)
    pilot = pd.concat(selected_parts, ignore_index=True)
    pilot = pilot.sort_values(["season", "venue_latitude", "venue_longitude", "game_id"]).reset_index(drop=True)
    return pilot


def fetch_station_payload(session: requests.Session, station: str, kickoff: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = kickoff - pd.Timedelta(hours=84)
    end = kickoff
    params = {
        "station": station,
        "model": "NBS",
        "sts": start.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    response = session.get(MOS_ENDPOINT, params=params, timeout=60)
    log = {
        "station": station,
        "http_status": response.status_code,
        "ok": bool(response.ok),
        "response_bytes": len(response.content),
        "error": "",
    }
    if not response.ok:
        log["error"] = response.text[:500]
        return pd.DataFrame(), log
    try:
        payload = response.json()
    except Exception as exc:
        log["error"] = f"JSON decode: {type(exc).__name__}: {exc}"
        return pd.DataFrame(), log
    if not isinstance(payload, list):
        log["error"] = f"Unexpected payload type: {type(payload).__name__}"
        return pd.DataFrame(), log
    frame = prepare_payload(payload)
    log["rows"] = len(frame)
    log["unique_runtimes"] = int(frame["runtime"].nunique()) if not frame.empty else 0
    return frame, log


def build_verification(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    errors = rows.copy()
    continuous = [
        ("temperature_f", "hist_temperature_f"),
        ("dewpoint_f", "hist_dewpoint_f"),
        ("humidity", "hist_humidity"),
        ("wind_mph", "hist_wind_mph"),
        ("precipitation", "hist_precipitation"),
        ("snowfall", "hist_snowfall"),
    ]
    for lead, group in errors.groupby("lead_hours", observed=True):
        for fcst, obs in continuous:
            if fcst not in group.columns or obs not in group.columns:
                continue
            pair = group[[fcst, obs]].apply(pd.to_numeric, errors="coerce").dropna()
            if pair.empty:
                continue
            err = pair[fcst] - pair[obs]
            abs_err = err.abs()
            corr = pair[fcst].corr(pair[obs]) if len(pair) >= 3 else np.nan
            metrics.append(
                {
                    "lead_hours": int(lead),
                    "variable": fcst,
                    "paired_n": len(pair),
                    "bias": float(err.mean()),
                    "mae": float(abs_err.mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(err)))),
                    "pearson_r": float(corr) if pd.notna(corr) else np.nan,
                    "median_abs_error": float(abs_err.median()),
                    "p90_abs_error": float(abs_err.quantile(0.9)),
                }
            )

        if {"wind_direction_degrees", "hist_wind_direction_degrees"} <= set(group.columns):
            circ = group.apply(
                lambda r: circular_abs_error_deg(r["wind_direction_degrees"], r["hist_wind_direction_degrees"]),
                axis=1,
            ).dropna()
            if not circ.empty:
                metrics.append(
                    {
                        "lead_hours": int(lead),
                        "variable": "wind_direction_degrees",
                        "paired_n": len(circ),
                        "bias": np.nan,
                        "mae": float(circ.mean()),
                        "rmse": float(np.sqrt(np.mean(np.square(circ)))),
                        "pearson_r": np.nan,
                        "median_abs_error": float(circ.median()),
                        "p90_abs_error": float(circ.quantile(0.9)),
                    }
                )

        if {"precip_probability_pct", "hist_precipitation"} <= set(group.columns):
            pop = pd.to_numeric(group["precip_probability_pct"], errors="coerce") / 100.0
            obs = (pd.to_numeric(group["hist_precipitation"], errors="coerce") > 0).astype(float)
            valid = pop.notna() & pd.to_numeric(group["hist_precipitation"], errors="coerce").notna()
            if valid.any():
                brier = float(np.mean(np.square(pop[valid] - obs[valid])))
                metrics.append(
                    {
                        "lead_hours": int(lead),
                        "variable": "precip_probability_brier",
                        "paired_n": int(valid.sum()),
                        "bias": np.nan,
                        "mae": brier,
                        "rmse": np.nan,
                        "pearson_r": np.nan,
                        "median_abs_error": np.nan,
                        "p90_abs_error": np.nan,
                    }
                )
    return pd.DataFrame(metrics), errors


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scope = pd.read_csv(AUDIT_DIR / "research_scope.csv", low_memory=False)
    mapping = pd.read_csv(AUDIT_DIR / "venue_season_station_mapping.csv", low_memory=False)
    candidates = pd.read_csv(AUDIT_DIR / "venue_season_station_candidates.csv", low_memory=False)
    scope["kickoff_utc"] = pd.to_datetime(scope["kickoff_utc"], utc=True, errors="coerce")
    pilot = stable_geographic_pilot(scope, mapping)
    pilot.to_csv(OUTPUT_DIR / "pilot_games.csv", index=False)

    session = requests.Session()
    session.headers.update({"User-Agent": "cfb-weather-totals pregame forecast realism pilot"})
    output_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    game_status: list[dict[str, Any]] = []

    for idx, game in pilot.iterrows():
        game_candidates = candidates[
            candidates["season"].eq(game["season"])
            & candidates["venue_id"].eq(game["venue_id"])
            & candidates["within_30_miles"].astype(bool)
        ].sort_values("candidate_rank")
        chosen_station = None
        chosen_distance = np.nan
        chosen_leads: list[dict[str, Any]] | None = None
        attempts = 0

        for _, candidate in game_candidates.iterrows():
            attempts += 1
            station = str(candidate["nbm_station_id"]).strip().upper()
            frame, req = fetch_station_payload(session, station, pd.Timestamp(game["kickoff_utc"]))
            req.update({"game_id": game["game_id"], "season": game["season"], "candidate_rank": candidate["candidate_rank"]})
            request_rows.append(req)
            if frame.empty:
                time.sleep(0.15)
                continue
            extracted = [extract_lead(frame, pd.Timestamp(game["kickoff_utc"]), lead) for lead in LEADS]
            if all(item is not None for item in extracted):
                chosen_station = station
                chosen_distance = float(candidate["distance_miles"])
                chosen_leads = [item for item in extracted if item is not None]
                break
            time.sleep(0.15)

        status = "ok" if chosen_leads is not None else "no_station_with_all_leads"
        game_status.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "venue_id": game["venue_id"],
                "venue_name": game.get("venue_name"),
                "status": status,
                "station": chosen_station,
                "station_distance_miles": chosen_distance,
                "candidate_attempts": attempts,
            }
        )
        if chosen_leads is None:
            print(f"[{idx + 1}/{len(pilot)}] game={game['game_id']} FAILED after {attempts} station candidates")
            continue

        hist_values = {
            "hist_temperature_f": game.get("temperature_f"),
            "hist_dewpoint_f": game.get("dewpoint_f"),
            "hist_humidity": game.get("humidity"),
            "hist_wind_mph": game.get("wind_mph"),
            "hist_wind_direction_degrees": game.get("wind_direction_degrees"),
            "hist_precipitation": game.get("precipitation"),
            "hist_snowfall": game.get("snowfall"),
        }
        for extracted in chosen_leads:
            output_rows.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "kickoff_utc": game["kickoff_utc"],
                    "venue_id": game["venue_id"],
                    "venue_name": game.get("venue_name"),
                    "venue_latitude": game.get("venue_latitude"),
                    "venue_longitude": game.get("venue_longitude"),
                    "nbm_station_id": chosen_station,
                    "station_distance_miles": chosen_distance,
                    **extracted,
                    **hist_values,
                }
            )
        print(f"[{idx + 1}/{len(pilot)}] game={game['game_id']} station={chosen_station} attempts={attempts}")
        time.sleep(0.15)

    rows = pd.DataFrame(output_rows)
    statuses = pd.DataFrame(game_status)
    requests_log = pd.DataFrame(request_rows)
    rows.to_csv(OUTPUT_DIR / "pilot_forecasts.csv", index=False)
    statuses.to_csv(OUTPUT_DIR / "pilot_game_status.csv", index=False)
    requests_log.to_csv(OUTPUT_DIR / "pilot_request_log.csv", index=False)

    if rows.empty:
        raise SystemExit("ERROR: Pilot returned no usable forecast rows.")

    coverage_rows: list[dict[str, Any]] = []
    for lead in LEADS:
        n = int(rows["lead_hours"].eq(lead).sum())
        coverage_rows.append(
            {
                "lead_hours": lead,
                "pilot_games": len(pilot),
                "complete_games": n,
                "coverage_pct": n / len(pilot),
                "unique_stations": int(rows.loc[rows["lead_hours"].eq(lead), "nbm_station_id"].nunique()),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUTPUT_DIR / "pilot_coverage.csv", index=False)
    metrics, _ = build_verification(rows)
    metrics.to_csv(OUTPUT_DIR / "pilot_verification_metrics.csv", index=False)

    runtime = pd.to_datetime(rows["forecast_runtime_utc"], utc=True, errors="coerce")
    availability = pd.to_datetime(rows["forecast_availability_utc"], utc=True, errors="coerce")
    cutoff = pd.to_datetime(rows["decision_cutoff_utc"], utc=True, errors="coerce")
    if (availability > cutoff).any():
        raise SystemExit("ERROR: Pilot contains a forecast unavailable by its decision cutoff.")
    if (pd.to_numeric(rows["forecast_valid_offset_minutes"], errors="coerce").abs() > MAX_VALID_OFFSET_MINUTES).any():
        raise SystemExit("ERROR: Pilot contains a kickoff valid-time offset beyond 90 minutes.")
    if runtime.isna().any() or availability.isna().any() or cutoff.isna().any():
        raise SystemExit("ERROR: Pilot contains invalid timing metadata.")

    print("--- pilot coverage ---")
    print(coverage.to_string(index=False))
    print("--- pilot verification metrics ---")
    print(metrics.to_string(index=False))
    print("--- game status ---")
    print(statuses["status"].value_counts(dropna=False).to_string())

    if float(coverage["coverage_pct"].min()) < 0.90:
        raise SystemExit("ERROR: Pilot lead-time coverage is below the frozen 90% source-feasibility gate.")


if __name__ == "__main__":
    main()
