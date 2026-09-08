from __future__ import annotations

import pandas as pd

from .pull_historical_gusts_batched import build_request_batches
from .wind_gust_recovery import select_recovery_batches


def _candidates() -> pd.DataFrame:
    rows = []
    game_id = 1
    for week, start in enumerate(["2024-09-07", "2024-09-14", "2024-09-21"]):
        for venue_id in range(100, 130):
            rows.append(
                {
                    "source_stack": "ecmwf_ifs",
                    "game_id": game_id,
                    "venue_id": venue_id,
                    "latitude": 30.0 + (venue_id - 100) / 10,
                    "longitude": -90.0,
                    "kickoff_utc": f"{start}T18:00:00Z",
                }
            )
            game_id += 1
    return pd.DataFrame(rows)


def test_recovery_chunk_never_exceeds_location_unit_cap() -> None:
    batches = build_request_batches(_candidates(), batch_size=20)
    selected = select_recovery_batches(batches, max_location_units=45)
    units = sum(batch.games["venue_id"].nunique() for batch in selected)
    assert units <= 45
    assert units == 40


def test_recovery_chunk_is_oldest_first() -> None:
    batches = build_request_batches(_candidates(), batch_size=20)
    selected = select_recovery_batches(batches, max_location_units=40)
    assert len(selected) == 2
    assert all("20240902" in batch.batch_id for batch in selected)


def test_max_batches_is_an_additional_guardrail() -> None:
    batches = build_request_batches(_candidates(), batch_size=20)
    selected = select_recovery_batches(
        batches, max_location_units=400, max_batches=1
    )
    assert len(selected) == 1
    assert selected[0].games["venue_id"].nunique() == 20


def main() -> None:
    tests = [
        test_recovery_chunk_never_exceeds_location_unit_cap,
        test_recovery_chunk_is_oldest_first,
        test_max_batches_is_an_additional_guardrail,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} wind-gust recovery self-tests")


if __name__ == "__main__":
    main()
