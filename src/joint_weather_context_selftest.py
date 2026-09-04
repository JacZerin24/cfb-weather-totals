from __future__ import annotations

import numpy as np
import pandas as pd

from .joint_weather_context_research import (
    MODEL_FEATURES,
    add_orientation_features,
    coverage_report,
    incremental_comparisons,
    paired_model_summary,
    regime_stability,
    walk_forward_predictions,
)
from .climate_context_research import prepare_research_data


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260904)
    venue_ids = np.arange(1, 13)
    venues = pd.DataFrame(
        {
            "venue_id": venue_ids,
            "venue_name": [f"Venue {v}" for v in venue_ids],
            "venue_latitude": np.linspace(27.0, 45.0, len(venue_ids)),
            "venue_longitude": np.linspace(-95.0, -75.0, len(venue_ids)),
            "venue_city": ["Test City"] * len(venue_ids),
            "venue_state": ["TS"] * len(venue_ids),
            "venue_timezone": ["America/Chicago"] * len(venue_ids),
        }
    )
    orientation = pd.DataFrame(
        {
            "venue_id": venue_ids,
            "field_axis_deg": np.linspace(0.0, 165.0, len(venue_ids)),
            "axis_uncertainty_deg": [2.0] * len(venue_ids),
            "roof_behavior": ["outdoor_only_observed"] * len(venue_ids),
        }
    )

    rows: list[dict] = []
    game_id = 1
    for season in range(2014, 2024):
        for i in range(90):
            venue_id = int(venue_ids[i % len(venue_ids)])
            latitude = float(
                venues.loc[venues["venue_id"] == venue_id, "venue_latitude"].iloc[0]
            )
            axis = float(
                orientation.loc[
                    orientation["venue_id"] == venue_id, "field_axis_deg"
                ].iloc[0]
            )
            month = 9 + (i % 4)
            base_temp = 82.0 - (latitude - 27.0) * 1.5 - (month - 9) * 8.0
            temperature = base_temp + rng.normal(0, 7)
            wind = max(0.5, 7.0 + (latitude - 35.0) * 0.12 + rng.normal(0, 4))
            wind_direction = float((axis + rng.uniform(0, 90)) % 360)
            angle = min(
                abs((wind_direction % 180) - (axis % 180)),
                180 - abs((wind_direction % 180) - (axis % 180)),
            )
            crosswind = wind * np.sin(np.deg2rad(angle))
            closing_total = 54.0 + rng.normal(0, 7)
            residual = (
                -0.18 * crosswind
                - 0.08 * crosswind * max(0.0, (55.0 - temperature) / 10.0)
                + 0.05 * (latitude - 35.0)
                + rng.normal(0, 10)
            )
            actual = closing_total + residual
            rows.append(
                {
                    "season": season,
                    "week": 1 + i // 12,
                    "game_id": game_id,
                    "start_date": f"{season}-{month:02d}-{1 + (i % 25):02d}T18:00:00Z",
                    "away_team": f"Away {i % 20}",
                    "home_team": f"Home {i % 20}",
                    "venue_id": venue_id,
                    "closing_total": closing_total,
                    "actual_total_points": actual,
                    "market_residual": residual,
                    "wind_mph": wind,
                    "wind_direction_degrees": wind_direction,
                    "temperature_f": temperature,
                    "humidity": 55 + rng.normal(0, 10),
                    "precipitation": max(0.0, rng.normal(0.02, 0.05)),
                    "snowfall": 0.0,
                    "dewpoint_f": temperature - 15,
                    "pressure": 1013 + rng.normal(0, 5),
                    "home_pregame_elo": 1500 + rng.normal(0, 80),
                    "away_pregame_elo": 1500 + rng.normal(0, 80),
                    "game_indoors": False,
                    "neutral_site": False,
                    "conference_game": bool(i % 2),
                    "line_provider": "synthetic",
                    "home_classification": "fbs",
                    "away_classification": "fbs",
                    "home_conference": "TEST",
                    "away_conference": "TEST",
                }
            )
            game_id += 1
    return pd.DataFrame(rows), venues, orientation


def main() -> None:
    raw, venues, orientation = synthetic_inputs()
    prepared = prepare_research_data(raw, venues)
    df = add_orientation_features(prepared, orientation)

    geometry = pd.DataFrame(
        {
            "venue_id": [1, 1],
            "wind_mph": [20.0, 20.0],
            "wind_direction_degrees": [90.0, 0.0],
            "outdoor": [True, True],
            "context_ready": [True, True],
            "fbs_vs_fbs": [True, True],
        }
    )
    geometry_orientation = pd.DataFrame(
        {
            "venue_id": [1],
            "field_axis_deg": [0.0],
            "axis_uncertainty_deg": [1.0],
            "roof_behavior": ["outdoor_only_observed"],
        }
    )
    checked = add_orientation_features(geometry, geometry_orientation)
    assert np.isclose(checked.loc[0, "crosswind_mph"], 20.0, atol=1e-6)
    assert np.isclose(checked.loc[0, "alongwind_mph"], 0.0, atol=1e-6)
    assert np.isclose(checked.loc[1, "crosswind_mph"], 0.0, atol=1e-6)
    assert np.isclose(checked.loc[1, "alongwind_mph"], 20.0, atol=1e-6)

    coverage = coverage_report(df)
    assert int(coverage.iloc[0]["joint_ready_games"]) == len(df)

    predictions, diagnostics = walk_forward_predictions(
        df, min_train_games=150, min_test_games=40
    )
    assert not predictions.empty
    assert not diagnostics.empty
    for model_name in MODEL_FEATURES:
        col = f"{model_name}_pred_market_residual"
        assert col in predictions.columns
        assert predictions[col].notna().all()

    summary = paired_model_summary(predictions, bootstrap_reps=250)
    comparisons = incremental_comparisons(predictions, bootstrap_reps=250)
    regimes = regime_stability(predictions, min_games=20)
    assert set(summary["model"]) == set(MODEL_FEATURES)
    assert not comparisons.empty
    assert not regimes.empty
    assert set(summary["evidence_status"]).issubset(
        {"REFERENCE", "NOT_PROVEN", "SUPPORTED_RETROSPECTIVELY"}
    )
    print("Joint weather-context synthetic validation passed")


if __name__ == "__main__":
    main()
