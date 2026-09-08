from __future__ import annotations

import numpy as np
import pandas as pd

from .pull_historical_gusts import (
    SOURCE_STACKS,
    _apply_snapshots,
    coverage_table,
    payload_to_hourly,
    qc_summary_table,
    select_snapshot,
    source_comparison_table,
)


def _payload() -> dict:
    return {
        "hourly": {
            "time": [
                "2024-09-07T16:00",
                "2024-09-07T17:00",
                "2024-09-07T18:00",
                "2024-09-07T19:00",
            ],
            "wind_speed_10m": [8.0, 10.0, 12.0, 15.0],
            "wind_direction_10m": [180.0, 190.0, 200.0, 210.0],
            "wind_gusts_10m": [14.0, 17.0, 21.0, 25.0],
        }
    }


def test_kickoff_alignment_and_no_future_hour() -> None:
    frame = payload_to_hourly(
        _payload(),
        ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
    )
    snap = select_snapshot(
        frame,
        pd.Timestamp("2024-09-07T18:37:00Z"),
        ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
    )
    assert snap["valid_time_utc"] == pd.Timestamp("2024-09-07T18:00:00Z")
    assert abs(snap["age_minutes"] - 37.0) < 1e-9
    assert snap["wind_speed_10m"] == 12.0
    assert snap["wind_gusts_10m"] == 21.0


def test_stale_snapshot_rejected() -> None:
    frame = payload_to_hourly(_payload(), ["wind_gusts_10m"])
    snap = select_snapshot(
        frame,
        pd.Timestamp("2024-09-07T22:00:00Z"),
        ["wind_gusts_10m"],
        max_age_minutes=120.0,
    )
    assert snap["valid_time_utc"] == pd.Timestamp("2024-09-07T19:00:00Z")
    assert snap["age_minutes"] == 180.0
    assert pd.isna(snap["wind_gusts_10m"])


def test_negative_spread_is_preserved() -> None:
    game = pd.Series(
        {
            "game_id": 1,
            "season": 2024,
            "start_date": "2024-09-07T18:30:00Z",
            "kickoff_utc": pd.Timestamp("2024-09-07T18:30:00Z"),
            "venue_id": 10,
            "venue_name": "Test",
            "latitude": 30.0,
            "longitude": -90.0,
            "cfbd_wind_mph": 19.0,
            "game_indoors_bool": False,
        }
    )
    wind = pd.DataFrame(
        {
            "valid_time_utc": [pd.Timestamp("2024-09-07T18:00:00Z")],
            "wind_speed_10m": [20.0],
            "wind_direction_10m": [180.0],
        }
    )
    gust = pd.DataFrame(
        {
            "valid_time_utc": [pd.Timestamp("2024-09-07T18:00:00Z")],
            "wind_gusts_10m": [18.0],
        }
    )
    row = _apply_snapshots(
        game,
        "era5_era5_land",
        wind,
        gust,
        {"latitude": 30.0, "longitude": -90.0, "elevation_m": 5},
        {"latitude": 30.1, "longitude": -90.1, "elevation_m": 8},
        120.0,
    )
    assert row["fetch_status"] == "ok"
    assert row["gust_spread_mph"] == -2.0
    assert row["gust_factor"] == 0.9
    assert row["component_grid_distance_km"] > 0


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_stack": "era5_era5_land", "game_id": 1, "season": 2023,
                "venue_id": 10, "latitude": 30.0, "longitude": -90.0,
                "wind_mph": 10.0, "gust_mph": 18.0, "gust_spread_mph": 8.0,
                "gust_factor": 1.8, "cfbd_wind_mph": 11.0, "fetch_status": "ok",
                "component_time_delta_minutes": 0.0, "component_grid_distance_km": 5.0,
                "wind_age_minutes": 30.0, "gust_age_minutes": 30.0,
            },
            {
                "source_stack": "era5_era5_land", "game_id": 2, "season": 2023,
                "venue_id": 11, "latitude": 35.0, "longitude": -85.0,
                "wind_mph": 20.0, "gust_mph": 18.0, "gust_spread_mph": -2.0,
                "gust_factor": 0.9, "cfbd_wind_mph": 18.0, "fetch_status": "ok",
                "component_time_delta_minutes": 0.0, "component_grid_distance_km": 5.0,
                "wind_age_minutes": 20.0, "gust_age_minutes": 20.0,
            },
            {
                "source_stack": "era5_era5_land", "game_id": 3, "season": 2024,
                "venue_id": 12, "latitude": np.nan, "longitude": np.nan,
                "wind_mph": np.nan, "gust_mph": np.nan, "gust_spread_mph": np.nan,
                "gust_factor": np.nan, "cfbd_wind_mph": 5.0,
                "fetch_status": "missing_venue_coordinates",
                "component_time_delta_minutes": np.nan,
                "component_grid_distance_km": np.nan,
                "wind_age_minutes": np.nan, "gust_age_minutes": np.nan,
            },
        ]
    )


def test_qc_tables() -> None:
    rows = _rows()
    coverage = coverage_table(rows)
    assert int(coverage["complete_games"].sum()) == 2

    qc = qc_summary_table(rows)
    neg = qc[
        (qc["source_stack"] == "era5_era5_land")
        & (qc["metric"] == "negative_gust_spread_rows")
    ]["value"].iloc[0]
    assert int(neg) == 1

    comparison = source_comparison_table(rows)
    overall = comparison[comparison["scope"] == "overall"].iloc[0]
    assert int(overall["games"]) == 2
    assert abs(float(overall["mean_abs_difference_mph"]) - 1.5) < 1e-9


def test_stack_metadata_is_explicit() -> None:
    long_stack = SOURCE_STACKS["era5_era5_land"]
    assert long_stack["wind_model"] == "era5"
    assert long_stack["gust_model"] == "era5_land"
    assert long_stack["same_model_components"] is False
    ifs = SOURCE_STACKS["ecmwf_ifs"]
    assert ifs["wind_model"] == ifs["gust_model"] == "ecmwf_ifs"
    assert ifs["same_model_components"] is True


def main() -> None:
    tests = [
        test_kickoff_alignment_and_no_future_hour,
        test_stale_snapshot_rejected,
        test_negative_spread_is_preserved,
        test_qc_tables,
        test_stack_metadata_is_explicit,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} wind-gust data/QC self-tests")


if __name__ == "__main__":
    main()
