from __future__ import annotations

import argparse
import math
import time
from datetime import timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import ROOT, ensure_dir, read_df, write_df

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MODELING_PATH = "data/processed/modeling_dataset.csv"
LOCATION_PATH = "data/reference/stadium_locations.csv"
CACHE_PATH = "data/processed/wind_gust_kickoff_cache.csv"
OUTPUT_PATH = "data/processed/wind_gust_kickoff.csv"
OUTPUT_DIR = "outputs/wind_gust"

SOURCE_STACKS: dict[str, dict[str, Any]] = {
    "era5_era5_land": {
        "available_since": 1950,
        "wind_model": "era5",
        "gust_model": "era5_land",
        "same_model_components": False,
    },
    "ecmwf_ifs": {
        "available_since": 2017,
        "wind_model": "ecmwf_ifs",
        "gust_model": "ecmwf_ifs",
        "same_model_components": True,
    },
}

RESULT_COLUMNS = [
    "game_id", "season", "start_date", "kickoff_utc", "venue_id", "venue_name",
    "latitude", "longitude", "cfbd_wind_mph", "game_indoors_bool",
    "source_stack", "weather_provider", "wind_model", "gust_model",
    "same_model_components", "wind_valid_time_utc", "gust_valid_time_utc",
    "wind_age_minutes", "gust_age_minutes", "component_time_delta_minutes",
    "wind_grid_latitude", "wind_grid_longitude", "wind_grid_elevation_m",
    "gust_grid_latitude", "gust_grid_longitude", "gust_grid_elevation_m",
    "component_grid_distance_km", "wind_mph", "wind_direction_degrees",
    "gust_mph", "gust_spread_mph", "gust_factor", "fetch_status", "fetch_detail",
]


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _iso(value: Any) -> str | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else pd.Timestamp(ts).isoformat()


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    )


def prepare_games(raw: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "season", "start_date", "venue_id"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(
            "Historical modeling dataset missing required columns: "
            + ", ".join(sorted(missing))
        )

    games = raw.copy()
    for col in ["game_id", "season", "venue_id"]:
        games[col] = pd.to_numeric(games[col], errors="coerce")
    games["kickoff_utc"] = pd.to_datetime(games["start_date"], utc=True, errors="coerce")
    games["cfbd_wind_mph"] = pd.to_numeric(
        games["wind_mph"] if "wind_mph" in games.columns else np.nan,
        errors="coerce",
    )
    games["game_indoors_bool"] = (
        _bool_series(games["game_indoors"])
        if "game_indoors" in games.columns
        else np.nan
    )

    loc = venues.copy()
    for col in ["venue_id", "venue_latitude", "venue_longitude"]:
        if col not in loc.columns:
            loc[col] = np.nan
        loc[col] = pd.to_numeric(loc[col], errors="coerce")
    keep = [
        c for c in [
            "venue_id", "venue_name", "venue_latitude", "venue_longitude", "venue_timezone"
        ] if c in loc.columns
    ]
    loc = loc[keep].drop_duplicates("venue_id")

    games = games.drop(
        columns=[
            c for c in [
                "venue_name", "venue_latitude", "venue_longitude", "venue_timezone"
            ] if c in games.columns
        ],
        errors="ignore",
    ).merge(loc, on="venue_id", how="left")
    return games.rename(
        columns={"venue_latitude": "latitude", "venue_longitude": "longitude"}
    )


def payload_to_hourly(payload: dict[str, Any], variables: Iterable[str]) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame(columns=["valid_time_utc", *variables])

    out = pd.DataFrame(
        {"valid_time_utc": pd.to_datetime(times, utc=True, errors="coerce")}
    )
    for var in variables:
        values = hourly.get(var)
        out[var] = (
            pd.to_numeric(pd.Series(values), errors="coerce")
            if isinstance(values, list) and len(values) == len(out)
            else np.nan
        )
    return out.dropna(subset=["valid_time_utc"]).sort_values("valid_time_utc")


def select_snapshot(
    hourly: pd.DataFrame,
    kickoff_utc: pd.Timestamp,
    value_columns: Iterable[str],
    max_age_minutes: float = 120.0,
) -> dict[str, Any]:
    blank = {
        "valid_time_utc": pd.NaT,
        "age_minutes": np.nan,
        **{c: np.nan for c in value_columns},
    }
    kickoff = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")
    if hourly.empty or pd.isna(kickoff):
        return blank

    eligible = hourly[hourly["valid_time_utc"] <= kickoff]
    if eligible.empty:
        return blank

    row = eligible.iloc[-1]
    valid = pd.Timestamp(row["valid_time_utc"])
    age = float((kickoff - valid).total_seconds() / 60.0)
    result = {
        "valid_time_utc": valid,
        "age_minutes": age,
        **{c: row.get(c, np.nan) for c in value_columns},
    }
    if age < 0 or age > max_age_minutes:
        for col in value_columns:
            result[col] = np.nan
    return result


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> float:
    vals = [_number(v) for v in [lat1, lon1, lat2, lon2]]
    if any(pd.isna(v) for v in vals):
        return np.nan
    lat1, lon1, lat2, lon2 = vals
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2 * r * math.asin(math.sqrt(a)))


class OpenMeteoArchiveClient:
    def __init__(self, request_delay: float = 0.05) -> None:
        self.request_delay = max(0.0, float(request_delay))
        self.session = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.75,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update(
            {"User-Agent": "cfb-weather-totals gust-research (github.com/JacZerin24/cfb-weather-totals)"}
        )

    def fetch(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        model: str,
        variables: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any], int]:
        response = self.session.get(
            ARCHIVE_URL,
            params={
                "latitude": f"{latitude:.5f}",
                "longitude": f"{longitude:.5f}",
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(variables),
                "wind_speed_unit": "mph",
                "timezone": "GMT",
                "models": model,
                "cell_selection": "land",
            },
            timeout=60,
        )
        status = response.status_code
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(str(payload.get("reason") or "Open-Meteo API error"))
        if self.request_delay:
            time.sleep(self.request_delay)
        meta = {
            "latitude": _number(payload.get("latitude")),
            "longitude": _number(payload.get("longitude")),
            "elevation_m": _number(payload.get("elevation")),
        }
        return payload_to_hourly(payload, variables), meta, status


def _base_row(game: pd.Series, stack: str) -> dict[str, Any]:
    spec = SOURCE_STACKS[stack]
    row = {c: np.nan for c in RESULT_COLUMNS}
    row.update(
        {
            "game_id": game.get("game_id"),
            "season": game.get("season"),
            "start_date": game.get("start_date"),
            "kickoff_utc": _iso(game.get("kickoff_utc")),
            "venue_id": game.get("venue_id"),
            "venue_name": game.get("venue_name"),
            "latitude": game.get("latitude"),
            "longitude": game.get("longitude"),
            "cfbd_wind_mph": game.get("cfbd_wind_mph"),
            "game_indoors_bool": game.get("game_indoors_bool"),
            "source_stack": stack,
            "weather_provider": "Open-Meteo Historical Weather API",
            "wind_model": spec["wind_model"],
            "gust_model": spec["gust_model"],
            "same_model_components": spec["same_model_components"],
            "fetch_status": "not_fetched",
            "fetch_detail": None,
        }
    )
    return row


def initial_rows(games: pd.DataFrame, stack: str) -> pd.DataFrame:
    spec = SOURCE_STACKS[stack]
    rows = []
    for _, game in games.iterrows():
        row = _base_row(game, stack)
        season = game.get("season")
        if pd.isna(game.get("game_id")):
            row["fetch_status"] = "missing_game_id"
        elif pd.isna(game.get("kickoff_utc")):
            row["fetch_status"] = "missing_kickoff"
        elif pd.isna(game.get("latitude")) or pd.isna(game.get("longitude")):
            row["fetch_status"] = "missing_venue_coordinates"
        elif pd.isna(season) or int(season) < int(spec["available_since"]):
            row["fetch_status"] = "source_out_of_range"
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _apply_snapshots(
    game: pd.Series,
    stack: str,
    wind_hourly: pd.DataFrame,
    gust_hourly: pd.DataFrame,
    wind_meta: dict[str, Any],
    gust_meta: dict[str, Any],
    max_age_minutes: float,
) -> dict[str, Any]:
    row = _base_row(game, stack)
    wind = select_snapshot(
        wind_hourly,
        game["kickoff_utc"],
        ["wind_speed_10m", "wind_direction_10m"],
        max_age_minutes,
    )
    gust = select_snapshot(
        gust_hourly,
        game["kickoff_utc"],
        ["wind_gusts_10m"],
        max_age_minutes,
    )
    row.update(
        {
            "wind_valid_time_utc": _iso(wind["valid_time_utc"]),
            "gust_valid_time_utc": _iso(gust["valid_time_utc"]),
            "wind_age_minutes": wind["age_minutes"],
            "gust_age_minutes": gust["age_minutes"],
            "wind_grid_latitude": wind_meta.get("latitude"),
            "wind_grid_longitude": wind_meta.get("longitude"),
            "wind_grid_elevation_m": wind_meta.get("elevation_m"),
            "gust_grid_latitude": gust_meta.get("latitude"),
            "gust_grid_longitude": gust_meta.get("longitude"),
            "gust_grid_elevation_m": gust_meta.get("elevation_m"),
            "wind_mph": wind["wind_speed_10m"],
            "wind_direction_degrees": wind["wind_direction_10m"],
            "gust_mph": gust["wind_gusts_10m"],
        }
    )
    wt = pd.to_datetime(row["wind_valid_time_utc"], utc=True, errors="coerce")
    gt = pd.to_datetime(row["gust_valid_time_utc"], utc=True, errors="coerce")
    if pd.notna(wt) and pd.notna(gt):
        row["component_time_delta_minutes"] = float(abs((gt - wt).total_seconds()) / 60)
    row["component_grid_distance_km"] = haversine_km(
        row["wind_grid_latitude"], row["wind_grid_longitude"],
        row["gust_grid_latitude"], row["gust_grid_longitude"],
    )

    wind_value = _number(row["wind_mph"])
    gust_value = _number(row["gust_mph"])
    if pd.notna(wind_value) and pd.notna(gust_value):
        row["gust_spread_mph"] = float(gust_value - wind_value)
        row["gust_factor"] = float(gust_value / wind_value) if wind_value >= 5 else np.nan
        row["fetch_status"] = "ok"
    elif pd.notna(wind_value) or pd.notna(gust_value):
        row["fetch_status"] = "partial"
    else:
        row["fetch_status"] = "missing"
    return row


def fetch_group(
    client: OpenMeteoArchiveClient,
    games: pd.DataFrame,
    stack: str,
    max_age_minutes: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = SOURCE_STACKS[stack]
    season = int(games["season"].iloc[0])
    venue_id = games["venue_id"].iloc[0]
    latitude = float(games["latitude"].iloc[0])
    longitude = float(games["longitude"].iloc[0])
    start = (games["kickoff_utc"].min().date() - timedelta(days=1)).isoformat()
    end = (games["kickoff_utc"].max().date() + timedelta(days=1)).isoformat()

    log: list[dict[str, Any]] = []
    wind_hourly, gust_hourly = pd.DataFrame(), pd.DataFrame()
    wind_meta: dict[str, Any] = {}
    gust_meta: dict[str, Any] = {}

    def request(component: str, model: str, variables: list[str]):
        try:
            frame, meta, status = client.fetch(
                latitude, longitude, start, end, model, variables
            )
            log.append(
                {
                    "source_stack": stack, "venue_id": venue_id, "season": season,
                    "component": component, "model": model, "start_date": start,
                    "end_date": end, "status": "ok", "http_status": status,
                    "hourly_rows": len(frame), "error": None,
                }
            )
            return frame, meta
        except Exception as exc:
            log.append(
                {
                    "source_stack": stack, "venue_id": venue_id, "season": season,
                    "component": component, "model": model, "start_date": start,
                    "end_date": end, "status": "error", "http_status": np.nan,
                    "hourly_rows": 0, "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return pd.DataFrame(), {}

    if stack == "ecmwf_ifs":
        frame, meta = request(
            "combined",
            spec["wind_model"],
            ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        )
        wind_hourly = gust_hourly = frame
        wind_meta = gust_meta = meta
    else:
        wind_hourly, wind_meta = request(
            "wind", spec["wind_model"], ["wind_speed_10m", "wind_direction_10m"]
        )
        gust_hourly, gust_meta = request(
            "gust", spec["gust_model"], ["wind_gusts_10m"]
        )

    rows = [
        _apply_snapshots(
            game, stack, wind_hourly, gust_hourly, wind_meta, gust_meta,
            max_age_minutes,
        )
        for _, game in games.iterrows()
    ]
    errors = [r for r in log if r["status"] == "error"]
    if errors:
        detail = "; ".join(f"{r['component']}:{r['error']}" for r in errors)
        for row in rows:
            if row["fetch_status"] != "ok":
                row["fetch_detail"] = detail
    return rows, log


def merge_cache(base: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    if cache.empty:
        return base
    useful = cache[cache["fetch_status"].isin(["ok", "partial"])].drop_duplicates(
        ["source_stack", "game_id"], keep="last"
    )
    if useful.empty:
        return base
    keep = base.merge(
        useful[["source_stack", "game_id"]],
        on=["source_stack", "game_id"],
        how="left",
        indicator=True,
    )["_merge"].eq("left_only")
    return pd.concat([base[keep], useful], ignore_index=True)[RESULT_COLUMNS]


def coverage_table(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["complete"] = work["wind_mph"].notna() & work["gust_mph"].notna()
    work["has_coordinates"] = work[["latitude", "longitude"]].notna().all(axis=1)
    return (
        work.groupby(["source_stack", "season"], dropna=False, observed=True)
        .agg(
            games=("game_id", "nunique"),
            games_with_coordinates=("has_coordinates", "sum"),
            games_with_wind=("wind_mph", lambda s: int(s.notna().sum())),
            games_with_gust=("gust_mph", lambda s: int(s.notna().sum())),
            complete_games=("complete", "sum"),
            complete_pct=("complete", "mean"),
            unique_venues=("venue_id", "nunique"),
        )
        .reset_index()
    )


def source_comparison_table(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.dropna(subset=["wind_mph", "cfbd_wind_mph"]).copy()
    output = []
    for stack, stack_group in work.groupby("source_stack", observed=True):
        scopes = [("overall", stack_group)] + [
            (str(int(season)), group)
            for season, group in stack_group.groupby("season", observed=True)
        ]
        for scope, group in scopes:
            diff = group["wind_mph"] - group["cfbd_wind_mph"]
            output.append(
                {
                    "source_stack": stack,
                    "scope": scope,
                    "games": len(group),
                    "pearson_r_external_vs_cfbd": (
                        float(group["wind_mph"].corr(group["cfbd_wind_mph"]))
                        if len(group) >= 2 else np.nan
                    ),
                    "mean_external_minus_cfbd_mph": float(diff.mean()),
                    "median_external_minus_cfbd_mph": float(diff.median()),
                    "mean_abs_difference_mph": float(diff.abs().mean()),
                    "median_abs_difference_mph": float(diff.abs().median()),
                }
            )
    return pd.DataFrame(output)


def qc_summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for stack, group in rows.groupby("source_stack", observed=True):
        wind = pd.to_numeric(group["wind_mph"], errors="coerce")
        gust = pd.to_numeric(group["gust_mph"], errors="coerce")
        spread = pd.to_numeric(group["gust_spread_mph"], errors="coerce")
        complete = wind.notna() & gust.notna()
        grid_distance = pd.to_numeric(
            group["component_grid_distance_km"], errors="coerce"
        )
        metrics = {
            "rows": len(group),
            "complete_rows": int(complete.sum()),
            "missing_coordinate_rows": int(
                (group["fetch_status"] == "missing_venue_coordinates").sum()
            ),
            "source_out_of_range_rows": int(
                (group["fetch_status"] == "source_out_of_range").sum()
            ),
            "partial_rows": int((group["fetch_status"] == "partial").sum()),
            "missing_rows": int((group["fetch_status"] == "missing").sum()),
            "negative_wind_rows": int((wind < 0).sum()),
            "negative_gust_rows": int((gust < 0).sum()),
            "negative_gust_spread_rows": int((spread < 0).sum()),
            "negative_gust_spread_rate_complete": (
                float((spread < 0).sum() / complete.sum()) if complete.sum() else np.nan
            ),
            "gust_over_100_mph_rows": int((gust > 100).sum()),
            "gust_spread_over_50_mph_rows": int((spread > 50).sum()),
            "component_time_mismatch_rows": int(
                (
                    pd.to_numeric(group["component_time_delta_minutes"], errors="coerce")
                    .fillna(0) > 1
                ).sum()
            ),
            "component_grid_distance_over_20km_rows": int(
                (grid_distance.fillna(0) > 20).sum()
            ),
            "wind_age_over_90_min_rows": int(
                (pd.to_numeric(group["wind_age_minutes"], errors="coerce") > 90).sum()
            ),
            "gust_age_over_90_min_rows": int(
                (pd.to_numeric(group["gust_age_minutes"], errors="coerce") > 90).sum()
            ),
        }
        output.extend(
            {"source_stack": stack, "metric": metric, "value": value}
            for metric, value in metrics.items()
        )
    return pd.DataFrame(output)


def distribution_table(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for stack, group in rows.groupby("source_stack", observed=True):
        for field in ["wind_mph", "gust_mph", "gust_spread_mph", "gust_factor"]:
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            if values.empty:
                continue
            q = values.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
            output.append(
                {
                    "source_stack": stack, "field": field, "n": len(values),
                    "mean": float(values.mean()),
                    "p00": float(q.loc[0]), "p01": float(q.loc[.01]),
                    "p05": float(q.loc[.05]), "p25": float(q.loc[.25]),
                    "p50": float(q.loc[.5]), "p75": float(q.loc[.75]),
                    "p95": float(q.loc[.95]), "p99": float(q.loc[.99]),
                    "p100": float(q.loc[1]),
                }
            )
    return pd.DataFrame(output)


def extreme_values_table(rows: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    output = []
    base_cols = [
        "source_stack", "game_id", "season", "kickoff_utc", "venue_id",
        "venue_name", "wind_mph", "gust_mph", "gust_spread_mph", "cfbd_wind_mph",
    ]
    for _, group in rows.groupby("source_stack", observed=True):
        checks = [
            ("highest_gust", "gust_mph", False),
            ("highest_gust_spread", "gust_spread_mph", False),
            ("most_negative_gust_spread", "gust_spread_mph", True),
        ]
        for issue, field, ascending in checks:
            part = group.dropna(subset=[field]).sort_values(field, ascending=ascending)
            if issue == "most_negative_gust_spread":
                part = part[part[field] < 0]
            for _, row in part.head(n).iterrows():
                record = {"issue": issue}
                record.update({c: row.get(c) for c in base_cols})
                output.append(record)
    return pd.DataFrame(output)


def write_qc_outputs(rows: pd.DataFrame, request_log: pd.DataFrame) -> None:
    ensure_dir(OUTPUT_DIR)
    write_df(coverage_table(rows), f"{OUTPUT_DIR}/data_coverage.csv")
    write_df(source_comparison_table(rows), f"{OUTPUT_DIR}/source_comparison.csv")
    write_df(qc_summary_table(rows), f"{OUTPUT_DIR}/qc_summary.csv")
    write_df(distribution_table(rows), f"{OUTPUT_DIR}/distribution_summary.csv")
    write_df(extreme_values_table(rows), f"{OUTPUT_DIR}/extreme_values.csv")
    write_df(request_log, f"{OUTPUT_DIR}/request_log.csv")


def run(
    stacks: list[str],
    max_age_minutes: float = 120,
    request_delay: float = 0.05,
    max_groups: int | None = None,
    resume: bool = True,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = prepare_games(read_df(MODELING_PATH), read_df(LOCATION_PATH))
    rows = pd.concat([initial_rows(games, stack) for stack in stacks], ignore_index=True)

    if resume and (ROOT / CACHE_PATH).exists():
        rows = merge_cache(rows, read_df(CACHE_PATH))

    client = OpenMeteoArchiveClient(request_delay)
    fetched: list[dict[str, Any]] = []
    requests_log: list[dict[str, Any]] = []
    candidates = rows[rows["fetch_status"] == "not_fetched"]
    groups = list(
        candidates.groupby(["source_stack", "venue_id", "season"], observed=True)
    )
    if max_groups is not None:
        groups = groups[: max(0, int(max_groups))]

    for i, ((stack, venue_id, season), group) in enumerate(groups, start=1):
        print(
            f"[{i}/{len(groups)}] {stack} venue={venue_id} "
            f"season={season} games={len(group)}"
        )
        game_ids = set(group["game_id"].dropna())
        source_games = games[games["game_id"].isin(game_ids)]
        new_rows, new_log = fetch_group(
            client, source_games, str(stack), max_age_minutes
        )
        fetched.extend(new_rows)
        requests_log.extend(new_log)

    if fetched:
        new = pd.DataFrame(fetched)
        rows = pd.concat([rows, new], ignore_index=True).drop_duplicates(
            ["source_stack", "game_id"], keep="last"
        )

    rows = rows[RESULT_COLUMNS]
    request_df = pd.DataFrame(requests_log)
    write_df(rows, CACHE_PATH)
    write_df(rows, OUTPUT_PATH)
    write_qc_outputs(rows, request_df)

    failures = (
        int((request_df["status"] == "error").sum())
        if not request_df.empty and "status" in request_df.columns else 0
    )
    print(
        f"Wrote {len(rows):,} game-source rows; "
        f"complete={(rows['fetch_status'] == 'ok').sum():,}; "
        f"request_failures={failures:,}"
    )
    if strict and failures:
        raise RuntimeError(f"{failures} Open-Meteo requests failed.")
    return rows, request_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire kickoff-aligned historical gust data and QC only; fit no models."
    )
    parser.add_argument("--stack", action="append", choices=sorted(SOURCE_STACKS))
    parser.add_argument("--include-ifs-sensitivity", action="store_true")
    parser.add_argument("--max-age-minutes", type=float, default=120)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    stacks = args.stack or ["era5_era5_land"]
    if args.include_ifs_sensitivity and "ecmwf_ifs" not in stacks:
        stacks.append("ecmwf_ifs")
    run(
        stacks=stacks,
        max_age_minutes=args.max_age_minutes,
        request_delay=args.request_delay,
        max_groups=args.max_groups,
        resume=not args.no_resume,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
