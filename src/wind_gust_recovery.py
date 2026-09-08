from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from .pull_historical_gusts import (
    CACHE_PATH,
    LOCATION_PATH,
    MODELING_PATH,
    OUTPUT_PATH,
    RESULT_COLUMNS,
    initial_rows,
    merge_cache,
    prepare_games,
    write_qc_outputs,
)
from .pull_historical_gusts_batched import (
    DEFAULT_BATCH_SIZE,
    BatchedOpenMeteoArchiveClient,
    RequestBatch,
    build_request_batches,
    fetch_batch,
)
from .utils import ROOT, ensure_dir, read_df, write_df

OUTPUT_DIR = "outputs/wind_gust"
RECOVERY_PROGRESS_PATH = f"{OUTPUT_DIR}/recovery_progress.csv"
RECOVERY_REQUEST_LOG_PATH = f"{OUTPUT_DIR}/recovery_request_log.csv"
DEFAULT_MAX_LOCATION_UNITS = 400
DEFAULT_REQUEST_DELAY = 2.5


def select_recovery_batches(
    batches: list[RequestBatch],
    max_location_units: int = DEFAULT_MAX_LOCATION_UNITS,
    max_batches: int | None = None,
) -> list[RequestBatch]:
    """Select an oldest-first recovery chunk without exceeding the location-unit cap."""
    if max_location_units < 1:
        raise ValueError("max_location_units must be >= 1")

    selected: list[RequestBatch] = []
    location_units = 0
    for batch in batches:
        if max_batches is not None and len(selected) >= max(0, int(max_batches)):
            break
        units = int(batch.games["venue_id"].nunique())
        if units < 1:
            continue
        if location_units + units > max_location_units:
            break
        selected.append(batch)
        location_units += units
    return selected


def _merge_successes(rows: pd.DataFrame, fetched: list[dict[str, Any]]) -> pd.DataFrame:
    if not fetched:
        return rows[RESULT_COLUMNS]
    new = pd.DataFrame(fetched)
    if new.empty:
        return rows[RESULT_COLUMNS]
    successful = new[new["fetch_status"].isin(["ok", "partial"])].copy()
    if successful.empty:
        return rows[RESULT_COLUMNS]
    return (
        pd.concat([rows, successful], ignore_index=True)
        .drop_duplicates(["source_stack", "game_id"], keep="last")
        [RESULT_COLUMNS]
    )


def _progress_row(
    rows: pd.DataFrame,
    starting_complete: int,
    request_df: pd.DataFrame,
    selected_batches: int,
    selected_location_units: int,
    stopped_on_error: bool,
) -> dict[str, Any]:
    ending_complete = int(rows["fetch_status"].eq("ok").sum())
    eligible = int(len(rows))
    coverage = ending_complete / eligible if eligible else np.nan
    failures = (
        int(request_df["status"].eq("error").sum())
        if not request_df.empty and "status" in request_df.columns
        else 0
    )
    attempted_location_units = (
        int(pd.to_numeric(request_df.get("venue_count"), errors="coerce").fillna(0).sum())
        if not request_df.empty
        else 0
    )
    return {
        "eligible_games": eligible,
        "starting_complete_games": starting_complete,
        "ending_complete_games": ending_complete,
        "newly_recovered_games": ending_complete - starting_complete,
        "remaining_incomplete_games": eligible - ending_complete,
        "complete_pct": coverage,
        "selected_batches": selected_batches,
        "selected_location_units": selected_location_units,
        "attempted_http_requests": len(request_df),
        "attempted_location_units": attempted_location_units,
        "request_failures": failures,
        "stopped_on_first_error": bool(stopped_on_error),
        "ready_for_modeling": bool(coverage >= 0.99 and failures == 0),
    }


def run(
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_location_units: int = DEFAULT_MAX_LOCATION_UNITS,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    max_age_minutes: float = 120.0,
    max_batches: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resume a partial IFS acquisition and stop immediately on the first request error.

    The existing cache is authoritative for already-successful rows. Failed/missing rows are
    regenerated as not_fetched candidates, so successful archive calls are never repeated.
    """
    stack = "ecmwf_ifs"
    games = prepare_games(read_df(MODELING_PATH), read_df(LOCATION_PATH))
    rows = initial_rows(games, stack)

    if not (ROOT / CACHE_PATH).exists():
        raise RuntimeError(
            "Recovery requires an existing wind_gust_kickoff_cache.csv partial artifact."
        )
    rows = merge_cache(rows, read_df(CACHE_PATH))
    rows = rows.drop_duplicates(["source_stack", "game_id"], keep="last")[RESULT_COLUMNS]

    starting_complete = int(rows["fetch_status"].eq("ok").sum())
    candidates = rows[rows["fetch_status"].eq("not_fetched")].copy()
    all_batches = build_request_batches(candidates, batch_size=batch_size)
    selected = select_recovery_batches(
        all_batches,
        max_location_units=max_location_units,
        max_batches=max_batches,
    )
    selected_location_units = int(
        sum(batch.games["venue_id"].nunique() for batch in selected)
    )

    client = BatchedOpenMeteoArchiveClient(request_delay=request_delay)
    fetched_successes: list[dict[str, Any]] = []
    request_logs: list[dict[str, Any]] = []
    stopped_on_error = False

    print(
        f"Recovery start: complete={starting_complete:,}/{len(rows):,}; "
        f"remaining={len(candidates):,}; selected_batches={len(selected):,}; "
        f"selected_location_units={selected_location_units:,}"
    )

    for i, batch in enumerate(selected, start=1):
        print(
            f"[{i}/{len(selected)}] {batch.batch_id} "
            f"venues={batch.games['venue_id'].nunique()} games={len(batch.games)} "
            f"days={batch.window_days}"
        )
        new_rows, log = fetch_batch(client, batch, max_age_minutes)
        request_logs.append(log)
        if log.get("status") != "ok":
            stopped_on_error = True
            print(
                "Stopping recovery immediately after first request error; "
                "no later batches will be attempted in this run."
            )
            break
        fetched_successes.extend(new_rows)

    rows = _merge_successes(rows, fetched_successes)
    request_df = pd.DataFrame(request_logs)

    ensure_dir(OUTPUT_DIR)
    write_df(rows, CACHE_PATH)
    write_df(rows, OUTPUT_PATH)
    write_qc_outputs(rows, request_df)
    write_df(request_df, RECOVERY_REQUEST_LOG_PATH)

    progress = pd.DataFrame(
        [
            _progress_row(
                rows,
                starting_complete=starting_complete,
                request_df=request_df,
                selected_batches=len(selected),
                selected_location_units=selected_location_units,
                stopped_on_error=stopped_on_error,
            )
        ]
    )
    write_df(progress, RECOVERY_PROGRESS_PATH)
    print(progress.to_string(index=False))
    return rows, request_df, progress


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Safely resume a partial IFS wind-gust acquisition from cache, "
            "with a location-unit cap and fail-fast rate-limit behavior."
        )
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-location-units", type=int, default=DEFAULT_MAX_LOCATION_UNITS
    )
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY)
    parser.add_argument("--max-age-minutes", type=float, default=120.0)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    run(
        batch_size=args.batch_size,
        max_location_units=args.max_location_units,
        request_delay=args.request_delay,
        max_age_minutes=args.max_age_minutes,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
