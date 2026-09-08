from __future__ import annotations

import numpy as np
import pandas as pd

from .wind_gust_value_research import (
    GUST_MAGNITUDE_THRESHOLDS,
    GUST_SPREAD_THRESHOLDS,
    MODEL_FEATURES,
    add_gust_interactions,
    evidence_gate,
    prepare_paired_data,
    walk_forward_predictions,
)


def _raw_games() -> pd.DataFrame:
    rows = []
    game_id = 1
    for season in [2017, 2018, 2019]:
        for i in range(8):
            total = 50.0 + (i % 4)
            actual = total + ((i % 3) - 1) * 3.0
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": i + 1,
                    "start_date": f"{season}-09-{10+i:02d}T18:30:00Z",
                    "venue_id": 100 + (i % 2),
                    "venue_name": f"Venue {i % 2}",
                    "home_team": f"Home {game_id}",
                    "away_team": f"Away {game_id}",
                    "home_classification": "fbs",
                    "away_classification": "fbs",
                    "game_indoors": False,
                    "closing_total": total,
                    "actual_total_points": actual,
                    "market_residual": actual - total,
                    "wind_mph": 8.0 + i,
                    "wind_direction_degrees": 180.0,
                    "temperature_f": 70.0 + (i % 5),
                    "humidity": 60.0,
                    "precipitation": 0.0,
                    "snowfall": 0.0,
                    "dewpoint_f": 55.0,
                    "pressure": 1012.0,
                }
            )
            game_id += 1
    return pd.DataFrame(rows)


def _venues() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "venue_id": [100, 101],
            "venue_name": ["Venue 0", "Venue 1"],
            "venue_latitude": [30.0, 35.0],
            "venue_longitude": [-90.0, -85.0],
            "venue_timezone": ["America/Chicago", "America/New_York"],
        }
    )


def _gusts(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": raw["game_id"],
            "source_stack": "ecmwf_ifs",
            "fetch_status": "ok",
            "same_model_components": True,
            "wind_model": "ecmwf_ifs",
            "gust_model": "ecmwf_ifs",
            "weather_provider": "Open-Meteo Historical Weather API",
            "wind_mph": raw["wind_mph"].to_numpy(float) + 0.5,
            "wind_direction_degrees": 185.0,
            "gust_mph": raw["wind_mph"].to_numpy(float) + 8.0,
            "gust_spread_mph": 7.5,
            "gust_factor": 1.6,
            "wind_valid_time_utc": pd.to_datetime(raw["start_date"], utc=True)
            .dt.floor("h")
            .astype(str),
            "gust_valid_time_utc": pd.to_datetime(raw["start_date"], utc=True)
            .dt.floor("h")
            .astype(str),
            "wind_age_minutes": 30.0,
            "gust_age_minutes": 30.0,
        }
    )


def test_prepare_paired_data_keeps_baseline_wind_separate() -> None:
    raw = _raw_games()
    gusts = _gusts(raw)
    paired, coverage = prepare_paired_data(raw, gusts, _venues())
    assert len(paired) == len(raw)
    assert np.allclose(paired["wind_mph"], raw["wind_mph"])
    assert np.allclose(paired["ifs_wind_mph"], raw["wind_mph"] + 0.5)
    assert "ifs_gust_mph" in paired.columns
    assert float(coverage.loc[coverage["scope"] == "overall", "paired_pct"].iloc[0]) == 1.0


def test_prepare_paired_data_rejects_noncomplete_source_rows() -> None:
    raw = _raw_games().head(2).copy()
    gusts = _gusts(raw)
    gusts.loc[1, "fetch_status"] = "partial"
    paired, _ = prepare_paired_data(raw, gusts, _venues())
    assert list(paired["game_id"].astype(int)) == [1]


def test_negative_spread_remains_continuous() -> None:
    raw = _raw_games().head(1).copy()
    gusts = _gusts(raw)
    gusts.loc[0, "gust_mph"] = 7.0
    gusts.loc[0, "wind_mph"] = 9.0
    gusts.loc[0, "gust_spread_mph"] = -2.0
    paired, _ = prepare_paired_data(raw, gusts, _venues())
    assert paired["ifs_gust_spread_mph"].iloc[0] == -2.0


def test_interactions_match_preregistered_definitions() -> None:
    frame = pd.DataFrame(
        {
            "ifs_wind_mph": [10.0],
            "ifs_gust_mph": [20.0],
            "ifs_gust_spread_mph": [10.0],
            "closing_total": [66.0],
            "temperature_anomaly_f": [-5.0],
        }
    )
    out = add_gust_interactions(frame)
    assert out["ifs_gust_spread_x_wind"].iloc[0] == 100.0
    assert out["ifs_gust_spread_x_market_total"].iloc[0] == 10.0
    assert out["ifs_gust_x_temperature_anomaly"].iloc[0] == -10.0


def test_evidence_gate_requires_every_condition() -> None:
    status, required = evidence_gate(-0.10, -0.01, -0.02, 5, 7)
    assert status == "SUPPORTED_RETROSPECTIVELY"
    assert required == 5

    for args in [
        (0.01, -0.01, -0.02, 5, 7),
        (-0.10, 0.01, -0.02, 5, 7),
        (-0.10, -0.01, 0.02, 5, 7),
        (-0.10, -0.01, -0.02, 4, 7),
    ]:
        assert evidence_gate(*args)[0] == "NOT_PROVEN"


def test_thresholds_are_fixed_before_results() -> None:
    assert GUST_MAGNITUDE_THRESHOLDS == [20.0, 25.0, 30.0, 35.0]
    assert GUST_SPREAD_THRESHOLDS == [5.0, 10.0, 15.0]


def test_ladder_uses_same_source_sustained_control() -> None:
    assert MODEL_FEATURES["baseline"] == []
    assert MODEL_FEATURES["ifs_sustained_control"] == ["ifs_wind_mph"]
    for model in ["gust_magnitude", "gust_spread", "gust_core", "gust_interactions"]:
        assert "ifs_wind_mph" in MODEL_FEATURES[model]


def test_walk_forward_scores_identical_rows_for_every_variant() -> None:
    raw = _raw_games()
    paired, _ = prepare_paired_data(raw, _gusts(raw), _venues())
    predictions, diagnostics = walk_forward_predictions(
        paired,
        min_train_games=8,
        min_test_games=4,
    )
    assert not predictions.empty
    assert predictions["game_id"].duplicated().sum() == 0
    for model in MODEL_FEATURES:
        col = f"{model}_pred_market_residual"
        assert col in predictions.columns
        assert predictions[col].notna().all()
    counts = diagnostics.groupby("model")["test_games"].sum()
    assert counts.nunique() == 1
    assert set(counts.index) == set(MODEL_FEATURES)


def main() -> None:
    tests = [
        test_prepare_paired_data_keeps_baseline_wind_separate,
        test_prepare_paired_data_rejects_noncomplete_source_rows,
        test_negative_spread_remains_continuous,
        test_interactions_match_preregistered_definitions,
        test_evidence_gate_requires_every_condition,
        test_thresholds_are_fixed_before_results,
        test_ladder_uses_same_source_sustained_control,
        test_walk_forward_scores_identical_rows_for_every_variant,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} wind-gust model self-tests")


if __name__ == "__main__":
    main()
