from __future__ import annotations

import numpy as np
import pandas as pd

from .joint_weather_context_robustness import (
    ABLATION_COMPONENT,
    ABLATION_FEATURES,
    WINDOW_SPECS,
    _training_slice,
    ablation_summary,
    leave_one_test_season_out,
    robustness_indicators,
    window_summary,
)


def synthetic_predictions() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for season in range(2018, 2026):
        for game in range(80):
            actual = rng.normal(0.0, 8.0)
            base_error = 2.0 + (game % 3) * 0.15
            joint_error = 1.0 + (game % 3) * 0.10
            row = {
                "season": season,
                "game_id": season * 1000 + game,
                "market_residual": actual,
                "closing_total": 58.0,
                "actual_total_points": 55.0,
                "baseline_pred_market_residual": actual + base_error,
                "joint_core_pred_market_residual": actual + joint_error,
            }
            for index, ablation in enumerate(ABLATION_COMPONENT):
                row[f"{ablation}_pred_market_residual"] = (
                    actual + joint_error + 0.30 + index * 0.03
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_feature_ladder() -> None:
    assert "crosswind_mph" in ABLATION_FEATURES["joint_core"]
    assert "alongwind_mph" in ABLATION_FEATURES["joint_core"]
    assert "crosswind_mph" not in ABLATION_FEATURES["joint_no_crosswind"]
    assert "alongwind_mph" not in ABLATION_FEATURES["joint_no_alongwind"]
    assert "temperature_anomaly_f" not in ABLATION_FEATURES[
        "joint_no_temperature_context"
    ]
    assert "wind_local_percentile" not in ABLATION_FEATURES[
        "joint_no_local_wind_context"
    ]
    assert "venue_latitude" not in ABLATION_FEATURES["joint_no_latitude_context"]
    assert WINDOW_SPECS["expanding"] is None
    assert WINDOW_SPECS["rolling_6_seasons"] == 6
    assert WINDOW_SPECS["rolling_4_seasons"] == 4


def test_training_windows() -> None:
    df = pd.DataFrame({"season": list(range(2014, 2026))})
    expanding = _training_slice(df, 2025, None)
    rolling4 = _training_slice(df, 2025, 4)
    assert expanding["season"].min() == 2014
    assert expanding["season"].max() == 2024
    assert rolling4["season"].tolist() == [2021, 2022, 2023, 2024]


def test_summaries() -> None:
    predictions = synthetic_predictions()
    ablations = ablation_summary(predictions, bootstrap_reps=500)
    assert len(ablations) == len(ABLATION_COMPONENT)
    assert (ablations["mae_penalty_when_removed"] > 0).all()
    assert (
        ablations["component_evidence_status"] == "COMPONENT_SUPPORTED"
    ).all()

    loso = leave_one_test_season_out(predictions, bootstrap_reps=500)
    assert len(loso) == predictions["season"].nunique()
    assert (loso["mean_mae_delta_challenger_minus_reference"] < 0).all()

    windows = window_summary(
        {
            "expanding": predictions,
            "rolling_6_seasons": predictions,
            "rolling_4_seasons": predictions,
        },
        bootstrap_reps=500,
    )
    assert len(windows) == 3
    assert (windows["mean_mae_delta_challenger_minus_reference"] < 0).all()
    assert (
        windows["evidence_status"] == "SUPPORTED_RETROSPECTIVELY"
    ).all()

    indicators = robustness_indicators(windows, loso)
    row = indicators.iloc[0]
    assert bool(row["all_window_mean_deltas_negative"])
    assert bool(row["all_loso_mean_deltas_negative"])
    assert bool(row["all_windows_improve_70pct_seasons"])
    assert row["robustness_status"] == "ROBUSTNESS_STRENGTHENED"


def main() -> None:
    test_feature_ladder()
    test_training_windows()
    test_summaries()
    print("joint_weather_context_robustness self-test passed")


if __name__ == "__main__":
    main()
