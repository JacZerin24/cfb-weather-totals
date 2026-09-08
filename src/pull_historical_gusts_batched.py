from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .pull_historical_gusts import (
    ARCHIVE_URL,
    CACHE_PATH,
    LOCATION_PATH,
    MODELING_PATH,
    OUTPUT_PATH,
    RESULT_COLUMNS,
    SOURCE_STACKS,
    OpenMeteoArchiveClient,
    _apply_snapshots,
    _number,
    initial_rows,
    merge_cache,
    payload_to_hourly,
    prepare_games,
    write_qc_outputs,
)
from .utils import ROOT, read_df, write_df


DEFAULT_BATCH_SIZE = 20
MAX_WINDOW_DAYS = 14
IFS_VARIABLES = ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]


@dataclass(frozen=True)
class RequestBatch:
    batch_id: str
    stack: str
    window_start_utc: pd.Timestamp
    window_end_utc: pd.Timestamp
    games: pd.DataFrame

    @property
    def start_date(self) -> str:
        return (self.games["kickoff_utc"].min().date() - timedelta(days=1)).isoformat()

    @property
    def end_date(self) -> str:
        return (self.games["kickoff_utc"].max().date() + timedelta(days=1)).isoformat()

    @property
    def window_days(self) -> int:
        return (pd.Timestamp(self.end_date) - pd.Timestamp(self.start_date)).days + 1


class BatchedOpenMeteoArchiveClient(OpenMeteoArchiveClient):
    """Open-Meteo archive client that preserves request-order location identity."""

    def fetch_many(
        self,
        locations: list[tuple[float, float]],
        start_date: str,
        end_date: str,
        model: str,
        variables: list[str],
    ) -> tuple[list[tuple[pd.DataFrame, dict[str, Any]]], int]:
        if not locations:
            return [], 0

        response = self.session.get(
            ARCHIVE_URL,
            params={
                "latitude": ",".join(f"{lat:.5f}" for lat, _ in locations),
                "longitude": ",".join(f"{lon:.5f}" for _, lon in locations),
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(variables),
                "wind_speed_unit": "mph",
                "timezone": "GMT",
                "models": model,
                "cell_selection": "land",
            },
            timeout=90,
        )
        status = response.status_code
        response.raise_for_status()
        payload = response.json()

        payloads = payload if isinstance(payload, list) else [payload]
        if len(payloads) != len(locations):
            raise RuntimeError(
                "Open-Meteo multi-location response count mismatch: "
                f"requested={len(locations)} returned={len(payloads)}"
            )

        results: list[tuple[pd.DataFrame, dict[str, Any]]] = []
        for index, item in enumerate(payloads):
            if not isinstance(item, dict):
                raise RuntimeError(
                    f"Open-Meteo location {index} returned non-object payload."
                )
            if item.get("error"):
                raise RuntimeError(
                    f"Open-Meteo location {index}: "
                    + str(item.get("reason") or "API error")
                )
            meta = {
                "latitude": _number(item.get("latitude")),
                "longitude": _number(item.get("longitude")),
                "elevation_m": _number(item.get("elevation")),
                "request_location_index": index,
                "requested_latitude": float(locations[index][0]),
                "requested_longitude": float(locations[index][1]),
            }
            results.append((payload_to_hourly(item, variables), meta))

        if self.request_delay:
            time.sleep(self.request_delay)
        return results, status


def _week_start(series: pd.Series) -> pd.Series:
    kickoff = pd.to_datetime(series, utc=True, errors="coerce")
    day = kickoff.dt.floor("D")
    return day - pd.to_timedelta(day.dt.weekday, unit="D")


def build_request_batches(
    candidates: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[RequestBatch]:
    """Batch games by kickoff week, then chunk venues within each week.

    The weekly grouping keeps every API request under the two-week interval that
    Open-Meteo uses as an important call-accounting boundary while still taking
    advantage of the API's multi-location request format.
    """
    if candidates.empty:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    work = candidates.copy()
    work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["kickoff_utc", "venue_id", "latitude", "longitude"])
    work["request_week_start"] = _week_start(work["kickoff_utc"])

    batches: list[RequestBatch] = []
    for (stack, week_start), week_games in work.groupby(
        ["source_stack", "request_week_start"], observed=True, sort=True
    ):
        venue_ids = sorted(week_games["venue_id"].dropna().unique().tolist())
        for chunk_index, start in enumerate(range(0, len(venue_ids), batch_size), start=1):
            chunk_ids = venue_ids[start : start + batch_size]
            games = week_games[week_games["venue_id"].isin(chunk_ids)].copy()
            if games.empty:
                continue
            batch = RequestBatch(
                batch_id=(
                    f"{stack}-{pd.Timestamp(week_start).strftime('%Y%m%d')}-"
                    f"{chunk_index:02d}"
                ),
                stack=str(stack),
                window_start_utc=pd.Timestamp(week_start),
                window_end_utc=pd.Timestamp(week_start) + pd.Timedelta(days=6),
                games=games,
            )
            if batch.window_days > MAX_WINDOW_DAYS:
                raise RuntimeError(
                    f"Batch {batch.batch_id} spans {batch.window_days} days, "
                    f"exceeding {MAX_WINDOW_DAYS}-day safety boundary."
                )
            batches.append(batch)
    return batches


def fetch_batch(
    client: BatchedOpenMeteoArchiveClient,
    batch: RequestBatch,
    max_age_minutes: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if batch.stack != "ecmwf_ifs":
        raise RuntimeError(
            "Batched acquisition is intentionally limited to the validated ecmwf_ifs stack."
        )
    spec = SOURCE_STACKS[batch.stack]
    if not spec.get("same_model_components"):
        raise RuntimeError("Batched IFS path requires same-model wind and gust components.")

    venues = (
        batch.games[["venue_id", "latitude", "longitude"]]
        .drop_duplicates("venue_id")
        .sort_values("venue_id")
        .reset_index(drop=True)
    )
    locations = [
        (float(row.latitude), float(row.longitude))
        for row in venues.itertuples(index=False)
    ]

    try:
        payloads, status = client.fetch_many(
            locations=locations,
            start_date=batch.start_date,
            end_date=batch.end_date,
            model=str(spec["wind_model"]),
            variables=IFS_VARIABLES,
        )
        by_venue: dict[float, tuple[pd.DataFrame, dict[str, Any]]] = {}
        for venue, payload in zip(venues.itertuples(index=False), payloads, strict=True):
            by_venue[float(venue.venue_id)] = payload

        rows: list[dict[str, Any]] = []
        for _, game in batch.games.iterrows():
            frame, meta = by_venue[float(game["venue_id"])]
            rows.append(
                _apply_snapshots(
                    game,
                    batch.stack,
                    frame,
                    frame,
                    meta,
                    meta,
                    max_age_minutes,
                )
            )
        log = {
            "source_stack": batch.stack,
            "batch_id": batch.batch_id,
            "component": "combined",
            "model": spec["wind_model"],
            "start_date": batch.start_date,
            "end_date": batch.end_date,
            "window_days": batch.window_days,
            "venue_count": len(venues),
            "game_count": len(batch.games),
            "venue_ids": ",".join(str(int(v)) for v in venues["venue_id"]),
            "status": "ok",
            "http_status": status,
            "hourly_rows_total": int(sum(len(frame) for frame, _ in payloads)),
            "error": None,
        }
        return rows, log
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        rows = []
        for _, game in batch.games.iterrows():
            row = _apply_snapshots(
                game,
                batch.stack,
                pd.DataFrame(),
                pd.DataFrame(),
                {},
                {},
                max_age_minutes,
            )
            row["fetch_detail"] = detail
            rows.append(row)
        log = {
            "source_stack": batch.stack,
            "batch_id": batch.batch_id,
            "component": "combined",
            "model": spec["wind_model"],
            "start_date": batch.start_date,
            "end_date": batch.end_date,
            "window_days": batch.window_days,
            "venue_count": len(venues),
            "game_count": len(batch.games),
            "venue_ids": ",".join(str(int(v)) for v in venues["venue_id"]),
            "status": "error",
            "http_status": np.nan,
            "hourly_rows_total": 0,
            "error": detail,
        }
        return rows, log


def run(
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_age_minutes: float = 120.0,
    request_delay: float = 0.05,
    max_batches: int | None = None,
    resume: bool = True,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stack = "ecmwf_ifs"
    games = prepare_games(read_df(MODELING_PATH), read_df(LOCATION_PATH))
    rows = initial_rows(games, stack)

    if resume and (ROOT / CACHE_PATH).exists():
        rows = merge_cache(rows, read_df(CACHE_PATH))

    candidates = rows[rows["fetch_status"] == "not_fetched"].copy()
    batches = build_request_batches(candidates, batch_size=batch_size)
    if max_batches is not None:
        batches = batches[: max(0, int(max_batches))]

    client = BatchedOpenMeteoArchiveClient(request_delay=request_delay)
    fetched: list[dict[str, Any]] = []
    request_logs: list[dict[str, Any]] = []

    for i, batch in enumerate(batches, start=1):
        print(
            f"[{i}/{len(batches)}] {batch.batch_id} "
            f"venues={batch.games['venue_id'].nunique()} games={len(batch.games)} "
            f"days={batch.window_days}"
        )
        new_rows, log = fetch_batch(client, batch, max_age_minutes)
        fetched.extend(new_rows)
        request_logs.append(log)

    if fetched:
        new = pd.DataFrame(fetched)
        rows = pd.concat([rows, new], ignore_index=True).drop_duplicates(
            ["source_stack", "game_id"], keep="last"
        )

    rows = rows[RESULT_COLUMNS]
    request_df = pd.DataFrame(request_logs)
    write_df(rows, CACHE_PATH)
    write_df(rows, OUTPUT_PATH)
    write_qc_outputs(rows, request_df)

    failures = (
        int((request_df["status"] == "error").sum())
        if not request_df.empty and "status" in request_df.columns
        else 0
    )
    complete = int((rows["fetch_status"] == "ok").sum())
    print(
        f"Wrote {len(rows):,} game-source rows; complete={complete:,}; "
        f"http_requests={len(request_df):,}; request_failures={failures:,}"
    )
    if strict and failures:
        raise RuntimeError(f"{failures} batched Open-Meteo requests failed.")
    return rows, request_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire kickoff-aligned ECMWF IFS wind/gust data using weekly "
            "multi-location batches; QC only, no model fitting."
        )
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-age-minutes", type=float, default=120.0)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    run(
        batch_size=args.batch_size,
        max_age_minutes=args.max_age_minutes,
        request_delay=args.request_delay,
        max_batches=args.max_batches,
        resume=not args.no_resume,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
