from __future__ import annotations

import numpy as np
import pandas as pd

from .pregame_forecast_pilot import (
    AVAILABILITY_LAG_HOURS,
    extract_lead,
    prepare_payload,
    relative_humidity_from_temp_dewpoint,
    select_run_and_kickoff_row,
    select_six_hour_period,
)


def synthetic_payload() -> list[dict]:
    rows = []
    for runtime in ["2025-09-06T07:00:00Z", "2025-09-06T13:00:00Z", "2025-09-06T19:00:00Z"]:
        for ftime in ["2025-09-06T18:00:00Z", "2025-09-06T21:00:00Z", "2025-09-07T00:00:00Z"]:
            rows.append(
                {
                    "runtime": runtime,
                    "ftime": ftime,
                    "tmp": 80,
                    "dpt": 60,
                    "wdr": 180,
                    "wsp": 10,
                    "gst": 15,
                    "p06": 30 if ftime == "2025-09-07T00:00:00Z" else None,
                    "q06": 5 if ftime == "2025-09-07T00:00:00Z" else None,
                    "s06": 0 if ftime == "2025-09-07T00:00:00Z" else None,
                }
            )
    return rows


def test_latest_eligible_runtime_not_newer_ineligible_run() -> None:
    frame = prepare_payload(synthetic_payload())
    kickoff = pd.Timestamp("2025-09-06T20:30:00Z")
    # 6h cutoff is 14:30Z. With +2h lag, 13Z runtime is not eligible (15Z availability),
    # so 07Z must be selected even though 13Z is present in the archive response.
    selected, _, detail = select_run_and_kickoff_row(frame, kickoff, 6)
    assert selected is not None
    assert pd.Timestamp(selected["runtime"]) == pd.Timestamp("2025-09-06T07:00:00Z")
    assert pd.Timestamp(detail["max_eligible_runtime_utc"]) == pd.Timestamp("2025-09-06T12:30:00Z")


def test_selected_availability_precedes_cutoff() -> None:
    frame = prepare_payload(synthetic_payload())
    kickoff = pd.Timestamp("2025-09-06T20:30:00Z")
    extracted = extract_lead(frame, kickoff, 6)
    assert extracted is not None
    assert pd.Timestamp(extracted["forecast_availability_utc"]) <= pd.Timestamp(extracted["decision_cutoff_utc"])
    assert AVAILABILITY_LAG_HOURS == 2.0


def test_nearest_kickoff_valid_time_within_90_minutes() -> None:
    frame = prepare_payload(synthetic_payload())
    kickoff = pd.Timestamp("2025-09-06T20:30:00Z")
    extracted = extract_lead(frame, kickoff, 6)
    assert extracted is not None
    assert pd.Timestamp(extracted["forecast_valid_time_utc"]) == pd.Timestamp("2025-09-06T21:00:00Z")
    assert abs(float(extracted["forecast_valid_offset_minutes"])) == 30.0


def test_six_hour_period_uses_endpoint_covering_kickoff() -> None:
    frame = prepare_payload(synthetic_payload())
    run = frame[frame["runtime"].eq(pd.Timestamp("2025-09-06T07:00:00Z"))]
    period = select_six_hour_period(run, pd.Timestamp("2025-09-06T20:30:00Z"))
    assert period is not None
    assert pd.Timestamp(period["ftime"]) == pd.Timestamp("2025-09-07T00:00:00Z")
    extracted = extract_lead(frame, pd.Timestamp("2025-09-06T20:30:00Z"), 6)
    assert extracted is not None
    assert np.isclose(float(extracted["precipitation"]), 0.05)
    assert np.isclose(float(extracted["snowfall"]), 0.0)
    assert np.isclose(float(extracted["precip_probability_pct"]), 30.0)


def test_relative_humidity_is_physical() -> None:
    rh = relative_humidity_from_temp_dewpoint(80, 60)
    assert 0 < rh < 100
    assert relative_humidity_from_temp_dewpoint(60, 60) > 99.0


def main() -> None:
    tests = [
        test_latest_eligible_runtime_not_newer_ineligible_run,
        test_selected_availability_precedes_cutoff,
        test_nearest_kickoff_valid_time_within_90_minutes,
        test_six_hour_period_uses_endpoint_covering_kickoff,
        test_relative_humidity_is_physical,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} pregame forecast lead-time self-tests")


if __name__ == "__main__":
    main()
