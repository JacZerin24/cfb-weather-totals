from __future__ import annotations

import numpy as np
import pandas as pd

from .joint_core_orientation_reconciliation import (
    PRIMARY_CHALLENGER,
    PRIMARY_REFERENCE,
    REAL_MODEL_FEATURES,
    add_placebo_alongwind,
    comparison_row,
    placebo_significance,
    reconciliation_status,
)


def test_fixed_feature_sets() -> None:
    assert "alongwind_mph" in REAL_MODEL_FEATURES["joint_core_real"]
    assert "alongwind_mph" not in REAL_MODEL_FEATURES["joint_core_no_along"]
    assert "crosswind_mph" in REAL_MODEL_FEATURES["joint_core_real"]
    assert "crosswind_mph" in REAL_MODEL_FEATURES["joint_core_no_along"]
    assert PRIMARY_CHALLENGER == "joint_core_real"
    assert PRIMARY_REFERENCE == "joint_core_no_along"


def test_placebo_alongwind_geometry() -> None:
    games = pd.DataFrame(
        {
            "venue_id": [1, 2],
            "wind_mph": [20.0, 20.0],
            "wind_direction_degrees": [0.0, 0.0],
        }
    )
    axis_map = pd.DataFrame(
        {
            "venue_id": [1, 2],
            "placebo_axis_deg": [0.0, 90.0],
        }
    )
    out = add_placebo_alongwind(games, axis_map)
    assert np.isclose(out.loc[0, "placebo_alongwind_mph"], 20.0)
    assert np.isclose(out.loc[1, "placebo_alongwind_mph"], 0.0, atol=1e-12)


def synthetic_predictions() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(44)
    for season in range(2016, 2026):
        for game in range(100):
            actual = rng.normal(0.0, 8.0)
            reference_error = 2.0 + (game % 5) * 0.03
            real_error = 1.0 + (game % 5) * 0.02
            rows.append(
                {
                    "season": season,
                    "game_id": season * 1000 + game,
                    "market_residual": actual,
                    "joint_core_no_along_pred_market_residual": actual
                    + reference_error,
                    "joint_core_real_pred_market_residual": actual + real_error,
                }
            )
    return pd.DataFrame(rows)


def test_reconciliation_logic() -> None:
    predictions = synthetic_predictions()
    component = pd.DataFrame(
        [comparison_row(predictions, bootstrap_reps=500)]
    )
    assert component.iloc[0]["evidence_status"] == "SUPPORTED_RETROSPECTIVELY"
    assert component.iloc[0]["mean_mae_delta_challenger_minus_reference"] < 0

    strong_placebos = pd.DataFrame(
        {
            "placebo_id": list(range(39)),
            "mean_mae_delta_placebo_vs_no_along": np.linspace(-0.20, 0.20, 39),
        }
    )
    placebo_test = placebo_significance(predictions, strong_placebos)
    assert (
        placebo_test.iloc[0]["conditional_geometry_status"]
        == "REAL_ALONG_DISTINGUISHED_WITHIN_JOINT_CORE"
    )
    status = reconciliation_status(component, placebo_test)
    assert status.iloc[0]["reconciliation_status"] == "CONDITIONAL_REAL_GEOMETRY_RECONCILED"

    similar_placebos = strong_placebos.copy()
    similar_placebos.loc[:9, "mean_mae_delta_placebo_vs_no_along"] = -1.2
    placebo_test_2 = placebo_significance(predictions, similar_placebos)
    assert (
        placebo_test_2.iloc[0]["conditional_geometry_status"]
        == "RANDOMIZED_ALONG_PERFORMS_SIMILARLY"
    )
    status_2 = reconciliation_status(component, placebo_test_2)
    assert status_2.iloc[0]["reconciliation_status"] == "ORIENTATION_PROXY_NOT_DISTINGUISHED"


def main() -> None:
    test_fixed_feature_sets()
    test_placebo_alongwind_geometry()
    test_reconciliation_logic()
    print("joint_core_orientation_reconciliation self-test passed")


if __name__ == "__main__":
    main()
