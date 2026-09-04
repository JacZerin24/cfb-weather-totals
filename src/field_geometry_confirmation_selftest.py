from __future__ import annotations

import numpy as np
import pandas as pd

from .field_geometry_confirmation import (
    CONFIRM_MODELS,
    PLACEBO_COUNT,
    add_alignment_features,
    add_placebo_alongwind,
    comparison_summary,
    confirmation_status,
    placebo_orientation_maps,
    placebo_significance,
)


def test_alignment_geometry() -> None:
    df = pd.DataFrame(
        {
            "wind_field_angle_deg": [0.0, 90.0, 45.0],
        }
    )
    out = add_alignment_features(df)
    assert np.isclose(out.loc[0, "along_alignment"], 1.0)
    assert np.isclose(out.loc[0, "cross_alignment"], 0.0)
    assert np.isclose(out.loc[1, "along_alignment"], 0.0, atol=1e-12)
    assert np.isclose(out.loc[1, "cross_alignment"], 1.0)
    assert np.isclose(out.loc[2, "along_alignment"], np.sqrt(0.5))


def test_placebo_maps_and_alongwind() -> None:
    orientation = pd.DataFrame(
        {
            "venue_id": [1, 2, 3, 4],
            "field_axis_deg": [0.0, 30.0, 60.0, 90.0],
        }
    )
    maps = placebo_orientation_maps(orientation, count=5, seed=42)
    assert len(maps) == 5
    original = orientation.set_index("venue_id")["field_axis_deg"]
    for axis_map in maps:
        mapped = axis_map.set_index("venue_id")["placebo_axis_deg"]
        assert sorted(mapped.tolist()) == sorted(original.tolist())
        assert not np.array_equal(mapped.to_numpy(), original.to_numpy())

    games = pd.DataFrame(
        {
            "venue_id": [1],
            "wind_mph": [20.0],
            "wind_direction_degrees": [0.0],
        }
    )
    axis_map = pd.DataFrame({"venue_id": [1], "placebo_axis_deg": [0.0]})
    out = add_placebo_alongwind(games, axis_map)
    assert np.isclose(out.loc[0, "placebo_alongwind_mph"], 20.0)


def synthetic_predictions() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for season in range(2016, 2026):
        for game in range(80):
            actual = rng.normal(0.0, 8.0)
            baseline_error = 2.0 + (game % 4) * 0.05
            along_error = 1.0 + (game % 4) * 0.04
            row = {
                "season": season,
                "game_id": season * 1000 + game,
                "market_residual": actual,
                "baseline_pred_market_residual": actual + baseline_error,
                "along_magnitude_pred_market_residual": actual + along_error,
            }
            for model in CONFIRM_MODELS:
                if model in {"baseline", "along_magnitude"}:
                    continue
                row[f"{model}_pred_market_residual"] = actual + 1.2
            rows.append(row)
    return pd.DataFrame(rows)


def test_confirmation_logic() -> None:
    predictions = synthetic_predictions()
    comparisons = comparison_summary(predictions, bootstrap_reps=500)
    primary = comparisons[comparisons["challenger"] == "along_magnitude"].iloc[0]
    assert primary["mean_mae_delta_challenger_minus_reference"] < 0
    assert primary["evidence_status"] == "SUPPORTED_RETROSPECTIVELY"

    placebo = pd.DataFrame(
        {
            "placebo_id": list(range(PLACEBO_COUNT)),
            "mean_mae_delta_vs_baseline": np.linspace(-0.20, 0.20, PLACEBO_COUNT),
        }
    )
    placebo_test = placebo_significance(predictions, placebo)
    assert placebo_test.iloc[0]["placebo_geometry_status"] == "REAL_AXIS_BEATS_PLACEBO"
    status = confirmation_status(comparisons, placebo_test)
    assert status.iloc[0]["confirmation_status"] == "GEOMETRY_CONFIRMED_RETROSPECTIVELY"


def main() -> None:
    test_alignment_geometry()
    test_placebo_maps_and_alongwind()
    test_confirmation_logic()
    print("field_geometry_confirmation self-test passed")


if __name__ == "__main__":
    main()
