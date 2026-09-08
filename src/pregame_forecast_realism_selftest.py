from __future__ import annotations

import numpy as np
import pandas as pd

from .pregame_forecast_realism_research import (
    apply_forecast_weather,
    compare_variants,
    complete_common_ids,
    recompute_bins,
)


def synthetic_forecasts() -> pd.DataFrame:
    rows = []
    for game_id, station in [(1, "KAAA"), (2, "KBBB")]:
        for lead in [24, 12, 6]:
            rows.append(
                {
                    "game_id": game_id,
                    "season": 2023,
                    "lead_hours": lead,
                    "nbm_station_id": station,
                    "temperature_f": 30.0 + game_id,
                    "dewpoint_f": 25.0 + game_id,
                    "humidity": 80.0,
                    "wind_mph": 16.0 + game_id,
                    "wind_direction_degrees": 270.0,
                    "precipitation": 0.25,
                    "snowfall": 0.0,
                    "precip_probability_pct": 60.0,
                }
            )
    return pd.DataFrame(rows)


def test_complete_common_ids() -> None:
    fcst = synthetic_forecasts()
    assert complete_common_ids(fcst) == {1, 2}
    missing = fcst[~((fcst["game_id"] == 2) & (fcst["lead_hours"] == 6))]
    assert complete_common_ids(missing) == {1}
    mismatch = fcst.copy()
    mismatch.loc[(mismatch["game_id"] == 2) & (mismatch["lead_hours"] == 6), "nbm_station_id"] = "KCCC"
    assert complete_common_ids(mismatch) == {1}


def test_primary_substitution_and_pressure_guard() -> None:
    hist = pd.DataFrame(
        {
            "game_id": [1, 2],
            "temperature_f": [75.0, 80.0],
            "dewpoint_f": [65.0, 70.0],
            "humidity": [55.0, 60.0],
            "wind_mph": [4.0, 5.0],
            "precipitation": [0.0, 0.0],
            "snowfall": [0.0, 0.0],
            "pressure": [1012.0, 1014.0],
        }
    )
    out = apply_forecast_weather(hist, synthetic_forecasts(), 24, mode="primary")
    assert len(out) == len(hist)
    assert np.allclose(out["temperature_f"], [31.0, 32.0])
    assert np.allclose(out["wind_mph"], [17.0, 18.0])
    assert np.allclose(out["precipitation"], [0.25, 0.25])
    assert out["pressure"].isna().all()
    assert out["wind_bin"].astype(str).tolist() == ["15-20", "15-20"]
    assert out["temp_bin"].astype(str).tolist() == ["<=35", "<=35"]


def test_core_substitution_keeps_period_fields_but_not_pressure() -> None:
    hist = pd.DataFrame(
        {
            "game_id": [1, 2],
            "temperature_f": [75.0, 80.0],
            "dewpoint_f": [65.0, 70.0],
            "humidity": [55.0, 60.0],
            "wind_mph": [4.0, 5.0],
            "precipitation": [0.01, 0.02],
            "snowfall": [0.1, 0.2],
            "pressure": [1012.0, 1014.0],
        }
    )
    out = apply_forecast_weather(hist, synthetic_forecasts(), 12, mode="core")
    assert np.allclose(out["precipitation"], [0.01, 0.02])
    assert np.allclose(out["snowfall"], [0.1, 0.2])
    assert out["pressure"].isna().all()


def paired_predictions(challenger_shift: float) -> pd.DataFrame:
    rows = []
    game_id = 1
    for season in [2021, 2022, 2023, 2024, 2025]:
        for _ in range(10):
            actual = 0.0
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "variant": "ref",
                    "actual_market_residual": actual,
                    "pred_market_residual": 2.0,
                }
            )
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "variant": "chal",
                    "actual_market_residual": actual,
                    "pred_market_residual": 2.0 + challenger_shift,
                }
            )
            game_id += 1
    return pd.DataFrame(rows)


def test_improvement_gate() -> None:
    # Reference absolute error=2; challenger absolute error=1.
    result = compare_variants(
        paired_predictions(-1.0),
        "ref",
        "chal",
        "synthetic_improvement",
        "improvement",
    )
    assert result["gate_pass"] is True
    assert result["mean_mae_delta"] < 0
    assert result["game_bootstrap_ci_high"] < 0
    assert result["season_cluster_ci_high"] < 0


def test_degradation_gate() -> None:
    # Reference absolute error=2; challenger absolute error=3.
    result = compare_variants(
        paired_predictions(1.0),
        "ref",
        "chal",
        "synthetic_degradation",
        "degradation",
    )
    assert result["gate_pass"] is True
    assert result["mean_mae_delta"] > 0
    assert result["game_bootstrap_ci_low"] > 0
    assert result["season_cluster_ci_low"] > 0


def test_recompute_bins_boundaries() -> None:
    frame = pd.DataFrame({"wind_mph": [5.0, 10.0, 15.0, 20.0], "temperature_f": [35.0, 50.0, 70.0, 85.0]})
    out = recompute_bins(frame)
    assert out["wind_bin"].astype(str).tolist() == ["0-5", "5-10", "10-15", "15-20"]
    assert out["temp_bin"].astype(str).tolist() == ["<=35", "35-50", "50-70", "70-85"]


def main() -> None:
    tests = [
        test_complete_common_ids,
        test_primary_substitution_and_pressure_guard,
        test_core_substitution_keeps_period_fields_but_not_pressure,
        test_improvement_gate,
        test_degradation_gate,
        test_recompute_bins_boundaries,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} pregame forecast realism model self-tests")


if __name__ == "__main__":
    main()
