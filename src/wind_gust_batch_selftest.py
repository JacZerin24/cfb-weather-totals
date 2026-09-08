from __future__ import annotations

import pandas as pd

from .pull_historical_gusts_batched import (
    MAX_WINDOW_DAYS,
    BatchedOpenMeteoArchiveClient,
    build_request_batches,
)


def test_weekly_batches_cover_each_game_once() -> None:
    rows = pd.DataFrame(
        [
            {
                "source_stack": "ecmwf_ifs",
                "game_id": 1,
                "venue_id": 10,
                "latitude": 30.0,
                "longitude": -90.0,
                "kickoff_utc": "2022-09-03T23:00:00Z",
            },
            {
                "source_stack": "ecmwf_ifs",
                "game_id": 2,
                "venue_id": 11,
                "latitude": 35.0,
                "longitude": -85.0,
                "kickoff_utc": "2022-09-04T18:00:00Z",
            },
            {
                "source_stack": "ecmwf_ifs",
                "game_id": 3,
                "venue_id": 12,
                "latitude": 40.0,
                "longitude": -80.0,
                "kickoff_utc": "2022-09-10T19:00:00Z",
            },
        ]
    )
    batches = build_request_batches(rows, batch_size=2)
    ids = [int(game_id) for batch in batches for game_id in batch.games["game_id"]]
    assert sorted(ids) == [1, 2, 3]
    assert len(ids) == len(set(ids))
    assert all(batch.window_days <= MAX_WINDOW_DAYS for batch in batches)
    assert all(batch.games["venue_id"].nunique() <= 2 for batch in batches)


def test_multi_location_payload_order_is_preserved() -> None:
    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [
                {
                    "latitude": 30.1,
                    "longitude": -90.1,
                    "elevation": 5,
                    "hourly": {
                        "time": ["2022-09-03T18:00"],
                        "wind_speed_10m": [5.0],
                        "wind_direction_10m": [180.0],
                        "wind_gusts_10m": [12.0],
                    },
                },
                {
                    "latitude": 40.1,
                    "longitude": -80.1,
                    "elevation": 7,
                    "hourly": {
                        "time": ["2022-09-03T18:00"],
                        "wind_speed_10m": [9.0],
                        "wind_direction_10m": [270.0],
                        "wind_gusts_10m": [18.0],
                    },
                },
            ]

    class FakeSession:
        def get(self, *args, **kwargs):
            params = kwargs["params"]
            assert params["latitude"] == "30.00000,40.00000"
            assert params["longitude"] == "-90.00000,-80.00000"
            return FakeResponse()

    client = BatchedOpenMeteoArchiveClient(request_delay=0)
    client.session = FakeSession()
    results, status = client.fetch_many(
        [(30.0, -90.0), (40.0, -80.0)],
        "2022-09-02",
        "2022-09-04",
        "ecmwf_ifs",
        ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
    )
    assert status == 200
    assert len(results) == 2
    first_frame, first_meta = results[0]
    second_frame, second_meta = results[1]
    assert first_meta["request_location_index"] == 0
    assert second_meta["request_location_index"] == 1
    assert first_meta["requested_latitude"] == 30.0
    assert second_meta["requested_latitude"] == 40.0
    assert float(first_frame["wind_speed_10m"].iloc[0]) == 5.0
    assert float(second_frame["wind_speed_10m"].iloc[0]) == 9.0


def test_response_count_mismatch_fails_closed() -> None:
    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "latitude": 30.1,
                "longitude": -90.1,
                "elevation": 5,
                "hourly": {"time": ["2022-09-03T18:00"], "wind_speed_10m": [5.0]},
            }

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    client = BatchedOpenMeteoArchiveClient(request_delay=0)
    client.session = FakeSession()
    try:
        client.fetch_many(
            [(30.0, -90.0), (40.0, -80.0)],
            "2022-09-02",
            "2022-09-04",
            "ecmwf_ifs",
            ["wind_speed_10m"],
        )
    except RuntimeError as exc:
        assert "response count mismatch" in str(exc)
    else:
        raise AssertionError("Expected multi-location response-count mismatch to fail closed")


def main() -> None:
    tests = [
        test_weekly_batches_cover_each_game_once,
        test_multi_location_payload_order_is_preserved,
        test_response_count_mismatch_fails_closed,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} batched wind-gust acquisition self-tests")


if __name__ == "__main__":
    main()
