from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .deep_research import prep

MODELING_PATH = Path("data/processed/modeling_dataset.csv")
LOCATION_PATH = Path("data/reference/stadium_locations.csv")
OUTPUT_DIR = Path("outputs/pregame_forecast_realism/station_audit")
NETWORK_ENDPOINT = "https://mesonet.agron.iastate.edu/geojson/network.py"
MAX_DISTANCE_MILES = 30.0

US_STATE_CODES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR",
]


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.7613
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nbm_station_id(sid: str) -> str:
    sid = str(sid).strip().upper()
    if len(sid) == 3:
        return "K" + sid
    return sid


def fetch_station_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    session = requests.Session()
    session.headers.update({"User-Agent": "cfb-weather-totals pregame forecast station audit"})
    stations: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []

    for state in US_STATE_CODES:
        network = f"{state}_ASOS"
        try:
            response = session.get(
                NETWORK_ENDPOINT,
                params={"network": network},
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            features = payload.get("features", []) if isinstance(payload, dict) else []
            request_rows.append(
                {"network": network, "status": "ok", "http_status": response.status_code, "features": len(features), "error": ""}
            )
            for feature in features:
                props = feature.get("properties") or {}
                geom = feature.get("geometry") or {}
                coords = geom.get("coordinates") or []
                if len(coords) < 2:
                    continue
                sid = props.get("sid") or feature.get("id")
                if not sid:
                    continue
                lon, lat = coords[:2]
                try:
                    lat = float(lat)
                    lon = float(lon)
                except (TypeError, ValueError):
                    continue
                stations.append(
                    {
                        "iem_station_id": str(sid).strip().upper(),
                        "nbm_station_id": nbm_station_id(str(sid)),
                        "station_name": props.get("sname") or props.get("name") or "",
                        "station_state": props.get("state") or state,
                        "station_network": props.get("network") or network,
                        "station_latitude": lat,
                        "station_longitude": lon,
                        "archive_begin": props.get("archive_begin"),
                        "archive_end": props.get("archive_end"),
                        "online": props.get("online"),
                    }
                )
        except Exception as exc:
            request_rows.append(
                {"network": network, "status": "error", "http_status": getattr(getattr(exc, "response", None), "status_code", np.nan), "features": 0, "error": f"{type(exc).__name__}: {exc}"}
            )

    inv = pd.DataFrame(stations)
    if inv.empty:
        raise RuntimeError("IEM ASOS station inventory is empty.")
    inv = inv.drop_duplicates(["iem_station_id", "station_latitude", "station_longitude"]).reset_index(drop=True)
    inv["archive_begin"] = pd.to_datetime(inv["archive_begin"], utc=True, errors="coerce")
    inv["archive_end"] = pd.to_datetime(inv["archive_end"], utc=True, errors="coerce")
    return inv, pd.DataFrame(request_rows)


def prepare_scope() -> pd.DataFrame:
    raw = pd.read_csv(MODELING_PATH, low_memory=False)
    df = prep(raw)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    df["venue_id"] = pd.to_numeric(df.get("venue_id"), errors="coerce")
    if "season_type" in df.columns:
        season_type = df["season_type"].astype(str).str.lower()
        df = df[season_type.eq("regular") | season_type.eq("regular season")].copy()
    scope = df[
        df["season"].between(2019, 2025, inclusive="both")
        & df["outdoor"].astype(bool)
        & df["fbs_vs_fbs"].astype(bool)
    ].copy()
    scope["kickoff_utc"] = pd.to_datetime(scope.get("start_date"), utc=True, errors="coerce")
    scope = scope[scope["kickoff_utc"].notna()].copy()

    locations = pd.read_csv(LOCATION_PATH, low_memory=False)
    locations["venue_id"] = pd.to_numeric(locations["venue_id"], errors="coerce")
    locations = locations.drop_duplicates("venue_id")
    keep = [
        "venue_id", "venue_name", "venue_latitude", "venue_longitude",
        "venue_city", "venue_state", "venue_timezone",
    ]
    loc = locations[[c for c in keep if c in locations.columns]].copy()
    loc = loc.rename(columns={c: f"ref_{c}" for c in loc.columns if c != "venue_id"})
    scope = scope.merge(loc, on="venue_id", how="left", validate="many_to_one")

    for field in ["venue_name", "venue_latitude", "venue_longitude", "venue_city", "venue_state", "venue_timezone"]:
        ref = f"ref_{field}"
        if field not in scope.columns:
            scope[field] = scope.get(ref)
        elif ref in scope.columns:
            scope[field] = scope[field].where(scope[field].notna(), scope[ref])

    scope["venue_latitude"] = pd.to_numeric(scope["venue_latitude"], errors="coerce")
    scope["venue_longitude"] = pd.to_numeric(scope["venue_longitude"], errors="coerce")
    scope = scope.drop_duplicates("game_id").sort_values(["season", "kickoff_utc", "game_id"]).reset_index(drop=True)
    return scope


def station_active_for_season(stations: pd.DataFrame, season: int) -> pd.Series:
    season_start = pd.Timestamp(f"{season}-08-01", tz="UTC")
    season_end = pd.Timestamp(f"{season}-12-31 23:59:59", tz="UTC")
    began = stations["archive_begin"].isna() | (stations["archive_begin"] <= season_end)
    ended = stations["archive_end"].isna() | (stations["archive_end"] >= season_start)
    return began & ended


def build_mapping(scope: pd.DataFrame, stations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    venue_seasons = scope[
        ["season", "venue_id", "venue_name", "venue_city", "venue_state", "venue_latitude", "venue_longitude"]
    ].drop_duplicates(["season", "venue_id"])

    for _, venue in venue_seasons.iterrows():
        base = venue.to_dict()
        if pd.isna(venue["venue_latitude"]) or pd.isna(venue["venue_longitude"]):
            mapping_rows.append({**base, "station_mapped": False, "mapping_reason": "missing_venue_coordinates"})
            continue

        eligible = stations[station_active_for_season(stations, int(venue["season"]))].copy()
        if eligible.empty:
            mapping_rows.append({**base, "station_mapped": False, "mapping_reason": "no_active_station_inventory"})
            continue

        distances = eligible.apply(
            lambda row: haversine_miles(
                float(venue["venue_latitude"]), float(venue["venue_longitude"]),
                float(row["station_latitude"]), float(row["station_longitude"]),
            ),
            axis=1,
        )
        nearest = eligible.assign(distance_miles=distances).nsmallest(5, "distance_miles")
        for rank, (_, station) in enumerate(nearest.iterrows(), start=1):
            candidates.append(
                {
                    **base,
                    "candidate_rank": rank,
                    "iem_station_id": station["iem_station_id"],
                    "nbm_station_id": station["nbm_station_id"],
                    "station_name": station["station_name"],
                    "station_state": station["station_state"],
                    "station_network": station["station_network"],
                    "station_latitude": station["station_latitude"],
                    "station_longitude": station["station_longitude"],
                    "distance_miles": station["distance_miles"],
                    "within_30_miles": bool(station["distance_miles"] <= MAX_DISTANCE_MILES),
                }
            )

        first = nearest.iloc[0]
        mapped = bool(first["distance_miles"] <= MAX_DISTANCE_MILES)
        mapping_rows.append(
            {
                **base,
                "station_mapped": mapped,
                "mapping_reason": "within_30_miles" if mapped else "nearest_station_over_30_miles",
                "iem_station_id": first["iem_station_id"],
                "nbm_station_id": first["nbm_station_id"],
                "station_name": first["station_name"],
                "station_state": first["station_state"],
                "station_network": first["station_network"],
                "station_latitude": first["station_latitude"],
                "station_longitude": first["station_longitude"],
                "distance_miles": first["distance_miles"],
            }
        )

    return pd.DataFrame(mapping_rows), pd.DataFrame(candidates)


def summarize(scope: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_map = scope.merge(
        mapping[["season", "venue_id", "station_mapped", "mapping_reason", "nbm_station_id", "distance_miles"]],
        on=["season", "venue_id"],
        how="left",
        validate="many_to_one",
    )
    game_map["station_mapped"] = game_map["station_mapped"].fillna(False).astype(bool)
    game_map["distance_band"] = pd.cut(
        game_map["distance_miles"],
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=["<=10", ">10-20", ">20-30", ">30"],
    )

    rows: list[dict[str, Any]] = []
    for label, group in [("overall", game_map)] + [
        (str(int(season)), group) for season, group in game_map.groupby("season", observed=True)
    ]:
        mapped = int(group["station_mapped"].sum())
        rows.append(
            {
                "scope": label,
                "eligible_games": len(group),
                "games_with_venue_coordinates": int(group[["venue_latitude", "venue_longitude"]].notna().all(axis=1).sum()),
                "games_station_within_30_miles": mapped,
                "coverage_pct": mapped / len(group) if len(group) else np.nan,
                "unique_venues": int(group["venue_id"].nunique(dropna=True)),
                "mapped_unique_venues": int(group.loc[group["station_mapped"], "venue_id"].nunique(dropna=True)),
                "median_station_distance_miles": float(group.loc[group["station_mapped"], "distance_miles"].median()) if mapped else np.nan,
                "p90_station_distance_miles": float(group.loc[group["station_mapped"], "distance_miles"].quantile(0.9)) if mapped else np.nan,
            }
        )
    coverage = pd.DataFrame(rows)

    bands = (
        game_map.groupby(["season", "distance_band"], observed=True)
        .size()
        .rename("games")
        .reset_index()
    )
    return coverage, bands


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scope = prepare_scope()
    if scope.empty:
        raise RuntimeError("Pregame forecast research scope is empty.")

    stations, requests_log = fetch_station_inventory()
    mapping, candidates = build_mapping(scope, stations)
    coverage, bands = summarize(scope, mapping)

    scope.to_csv(OUTPUT_DIR / "research_scope.csv", index=False)
    stations.to_csv(OUTPUT_DIR / "station_inventory.csv", index=False)
    requests_log.to_csv(OUTPUT_DIR / "network_request_log.csv", index=False)
    mapping.to_csv(OUTPUT_DIR / "venue_season_station_mapping.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "venue_season_station_candidates.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "coverage.csv", index=False)
    bands.to_csv(OUTPUT_DIR / "distance_bands.csv", index=False)

    print("--- station coverage ---")
    print(coverage.to_string(index=False))
    print("--- network requests ---")
    print(requests_log.groupby("status").size().to_string())
    print(f"Station inventory rows: {len(stations):,}")

    overall = coverage.loc[coverage["scope"].eq("overall")].iloc[0]
    if float(overall["coverage_pct"]) < 0.90:
        raise SystemExit(
            f"ERROR: Station mapping coverage {float(overall['coverage_pct']):.3%} is below the frozen 90% gate."
        )


if __name__ == "__main__":
    main()
