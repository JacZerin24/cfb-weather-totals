from __future__ import annotations

import numpy as np
import pandas as pd

from .climate_context_research import (
    MODEL_FEATURES,
    apply_context_features,
    build_context_reference,
    coverage_report,
    latitude_weather_interactions,
    model_by_season,
    normalize_venues,
    paired_model_summary,
    prepare_research_data,
    walk_forward_predictions,
)


def synthetic_inputs() -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(20260904)
    venue_records = [
        {"id": 101, "name": "Gulf Stadium", "latitude": 29.5, "longitude": -90.1},
        {"id": 102, "name": "South Stadium", "latitude": 32.2, "longitude": -86.3},
        {"id": 103, "name": "Mid Stadium", "latitude": 37.8, "longitude": -84.5},
        {"id": 104, "name": "North Stadium", "latitude": 43.1, "longitude": -89.4},
    ]
    latitude = {int(v["id"]): float(v["latitude"]) for v in venue_records}
    rows = []
    game_id = 1
    for season in range(2014, 2019):
        for index in range(180):
            venue_id = 101 + (index % 4)
            month = 9 + ((index // 60) % 3)
            week = 1 + (index % 12)
            local_baseline = (
                82.0 - (latitude[venue_id] - 29.5) * 1.6 - (month - 9) * 9.0
            )
            temperature = local_baseline + rng.normal(0, 8)
            wind = max(0.0, 7.0 + (latitude[venue_id] - 29.5) * 0.25 + rng.normal(0, 4))
            closing_total = 55.0 + rng.normal(0, 5)
            cold_shock = min(temperature - local_baseline, 0.0)
            residual = (
                0.16 * cold_shock - 0.20 * max(wind - 12.0, 0.0) + rng.normal(0, 8)
            )
            actual = round(closing_total + residual)
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "start_date": f"{season}-{month:02d}-{1 + (index % 25):02d}T18:00:00Z",
                    "home_team": f"Home {venue_id}",
                    "away_team": f"Away {index % 20}",
                    "venue_id": venue_id,
                    "closing_total": closing_total,
                    "actual_total_points": actual,
                    "market_residual": actual - closing_total,
                    "temperature_f": temperature,
                    "wind_mph": wind,
                    "humidity": 60.0,
                    "precipitation": 0.0,
                    "snowfall": 0.0,
                    "dewpoint_f": 55.0,
                    "pressure": 1013.0,
                    "game_indoors": index % 45 == 0,
                    "neutral_site": False,
                    "conference_game": True,
                    "line_provider": "synthetic",
                    "home_classification": "fbs",
                    "away_classification": "fbs",
                    "home_conference": "Test",
                    "away_conference": "Test",
                }
            )
            game_id += 1
    return pd.DataFrame(rows), venue_records


def main() -> None:
    raw, venue_records = synthetic_inputs()
    venues = normalize_venues(venue_records)
    assert list(venues["venue_id"]) == [101, 102, 103, 104]

    prepared = prepare_research_data(raw, venues)
    assert set(prepared["latitude_band"].dropna().astype(str)) == {
        "<30N",
        "30-35N",
        "35-40N",
        "40N+",
    }
    coverage = coverage_report(prepared)
    assert int(coverage.iloc[0]["with_coordinates"]) == len(raw)
    assert 0 < int(coverage.iloc[0]["context_ready_games"]) < len(raw)

    interactions = latitude_weather_interactions(prepared, min_games=5)
    assert {"temperature", "wind"} <= set(interactions["weather_grouping"])
    assert interactions["games"].min() >= 5

    train = prepared[prepared["season"] <= 2015]
    test = prepared[prepared["season"] == 2016]
    reference = build_context_reference(
        train, min_venue_month_games=3, min_band_month_games=3
    )
    enriched = apply_context_features(test, reference)
    ready = enriched["context_ready"]
    assert enriched.loc[ready, "temperature_anomaly_f"].notna().all()
    assert enriched.loc[ready, "wind_local_percentile"].between(0, 1).all()

    predictions, diagnostics = walk_forward_predictions(
        prepared,
        min_train_games=200,
        min_test_games=100,
    )
    assert not predictions.empty
    assert not diagnostics.empty
    for model_name in MODEL_FEATURES:
        assert f"{model_name}_pred_market_residual" in predictions.columns

    summary = paired_model_summary(predictions, bootstrap_reps=200)
    by_season = model_by_season(predictions)
    assert set(summary["model"]) == set(MODEL_FEATURES)
    assert set(by_season["model"]) == set(MODEL_FEATURES)
    baseline = summary[summary["model"] == "baseline"].iloc[0]
    assert baseline["mae_delta_vs_baseline"] == 0.0
    assert baseline["mae_delta_ci_low"] == 0.0
    assert baseline["mae_delta_ci_high"] == 0.0
    print("Climate-context research self-test passed.")


if __name__ == "__main__":
    main()
