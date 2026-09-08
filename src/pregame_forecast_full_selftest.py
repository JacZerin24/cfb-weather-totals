from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .pregame_forecast_full_acquisition import complete_game_ids, merge_cache, qc_tables


def _lead_row(game_id: int, season: int, lead: int, station: str = "KAAA") -> dict:
    kickoff = pd.Timestamp(f"{season}-09-10T19:30:00Z")
    cutoff = kickoff - pd.Timedelta(hours=lead)
    runtime = cutoff - pd.Timedelta(hours=4)
    return {
        "game_id": game_id,
        "season": season,
        "lead_hours": lead,
        "nbm_station_id": station,
        "station_distance_miles": 5.0,
        "decision_cutoff_utc": cutoff,
        "forecast_runtime_utc": runtime,
        "forecast_availability_utc": runtime + pd.Timedelta(hours=2),
        "forecast_valid_time_utc": kickoff + pd.Timedelta(minutes=30),
        "forecast_valid_offset_minutes": 30.0,
        "temperature_f": 78.0,
        "dewpoint_f": 60.0,
        "humidity": 55.0,
        "wind_mph": 8.0,
        "wind_direction_degrees": 180.0,
        "precip_probability_pct": 20.0,
        "precipitation": 0.0,
        "snowfall": 0.0,
        "hist_temperature_f": 80.0,
        "hist_dewpoint_f": 61.0,
        "hist_humidity": 54.0,
        "hist_wind_mph": 7.0,
        "hist_wind_direction_degrees": 190.0,
        "hist_precipitation": 0.0,
        "hist_snowfall": 0.0,
    }


def test_complete_game_requires_all_three_leads_same_station() -> None:
    good = pd.DataFrame([_lead_row(1, 2022, lead) for lead in [24, 12, 6]])
    assert complete_game_ids(good) == {1}

    missing = good[good["lead_hours"] != 12]
    assert complete_game_ids(missing) == set()

    mixed = good.copy()
    mixed.loc[mixed["lead_hours"] == 6, "nbm_station_id"] = "KBBB"
    assert complete_game_ids(mixed) == set()


def test_merge_cache_is_unique_by_game_and_lead() -> None:
    existing = pd.DataFrame([_lead_row(1, 2022, lead) for lead in [24, 12, 6]])
    replacement = _lead_row(1, 2022, 6)
    replacement["temperature_f"] = 81.0
    merged = merge_cache(existing, [replacement])
    assert len(merged) == 3
    assert float(merged.loc[merged["lead_hours"] == 6, "temperature_f"].iloc[0]) == 81.0


def test_qc_gate_requires_95_overall_and_90_each_season() -> None:
    scope_rows = []
    rows = []
    gid = 1
    # 20 games per season, all complete = 100% coverage.
    for season in [2021, 2022]:
        for _ in range(20):
            scope_rows.append({"game_id": gid, "season": season})
            rows.extend(_lead_row(gid, season, lead) for lead in [24, 12, 6])
            gid += 1
    scope = pd.DataFrame(scope_rows)
    data = pd.DataFrame(rows)
    coverage, validation, _ = qc_tables(scope, data, pd.DataFrame())
    assert bool(validation.iloc[0]["ready_for_modeling"])
    assert np.isclose(float(coverage.loc[coverage["scope"] == "overall", "coverage_pct"].iloc[0]), 1.0)

    # Remove three 2022 games: overall 92.5%, below 95 and 2022 85%, below 90.
    bad = data[~data["game_id"].isin([38, 39, 40])].copy()
    _, bad_validation, _ = qc_tables(scope, bad, pd.DataFrame())
    assert not bool(bad_validation.iloc[0]["ready_for_modeling"])


def test_qc_fails_future_availability_and_large_station_distance() -> None:
    scope = pd.DataFrame([{"game_id": 1, "season": 2022}])
    rows = pd.DataFrame([_lead_row(1, 2022, lead) for lead in [24, 12, 6]])
    # Tiny one-game scope cannot meet season threshold logic only if timing is clean; force explicit violations.
    rows.loc[rows["lead_hours"] == 6, "forecast_availability_utc"] = pd.Timestamp("2022-09-10T14:00:00Z")
    rows.loc[rows["lead_hours"] == 6, "decision_cutoff_utc"] = pd.Timestamp("2022-09-10T13:30:00Z")
    rows.loc[rows["lead_hours"] == 12, "station_distance_miles"] = 31.0
    _, validation, _ = qc_tables(scope, rows, pd.DataFrame())
    assert int(validation.iloc[0]["availability_violations"]) == 1
    assert int(validation.iloc[0]["station_distance_violations"]) == 1
    assert not bool(validation.iloc[0]["ready_for_modeling"])


def test_core_field_completeness_is_enforced() -> None:
    scope_rows = []
    rows = []
    for gid in range(1, 101):
        scope_rows.append({"game_id": gid, "season": 2022})
        rows.extend(_lead_row(gid, 2022, lead) for lead in [24, 12, 6])
    scope = pd.DataFrame(scope_rows)
    data = pd.DataFrame(rows)
    # Remove wind from 4/300 lead rows -> 98.67% core completeness, below 99%.
    data.loc[data.index[:4], "wind_mph"] = np.nan
    _, validation, _ = qc_tables(scope, data, pd.DataFrame())
    assert float(validation.iloc[0]["core_continuous_complete_pct"]) < 0.99
    assert not bool(validation.iloc[0]["ready_for_modeling"])


def main() -> None:
    tests = [
        test_complete_game_requires_all_three_leads_same_station,
        test_merge_cache_is_unique_by_game_and_lead,
        test_qc_gate_requires_95_overall_and_90_each_season,
        test_qc_fails_future_availability_and_large_station_distance,
        test_core_field_completeness_is_enforced,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} full-acquisition self-tests")


if __name__ == "__main__":
    main()
