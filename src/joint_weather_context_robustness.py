from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .climate_context_research import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    apply_context_features,
    bootstrap_mean_ci,
    build_context_reference,
    prepare_research_data,
)
from .joint_weather_context_research import (
    CLIMATE_FEATURES,
    add_joint_interactions,
    add_orientation_features,
    season_cluster_ci,
)
from .model_bakeoff import feature_lists, prep_features, reg_models
from .utils import ensure_dir, read_df, write_df

OUTPUT_DIR = "outputs/joint_weather_context_robustness"

JOINT_CORE_FEATURES = [
    "crosswind_mph",
    "alongwind_mph",
    *CLIMATE_FEATURES,
]

ABLATION_FEATURES = {
    "baseline": [],
    "joint_core": JOINT_CORE_FEATURES,
    "joint_no_orientation": CLIMATE_FEATURES,
    "joint_no_climate": ["crosswind_mph", "alongwind_mph"],
    "joint_no_crosswind": ["alongwind_mph", *CLIMATE_FEATURES],
    "joint_no_alongwind": ["crosswind_mph", *CLIMATE_FEATURES],
    "joint_no_temperature_context": [
        "crosswind_mph",
        "alongwind_mph",
        "venue_latitude",
        "wind_local_percentile",
        "wind_latitude_interaction",
    ],
    "joint_no_local_wind_context": [
        "crosswind_mph",
        "alongwind_mph",
        "venue_latitude",
        "temperature_anomaly_f",
        "temperature_latitude_interaction",
    ],
    "joint_no_latitude_context": [
        "crosswind_mph",
        "alongwind_mph",
        "temperature_anomaly_f",
        "wind_local_percentile",
    ],
}

ABLATION_COMPONENT = {
    "joint_no_orientation": "orientation_vector",
    "joint_no_climate": "climate_context",
    "joint_no_crosswind": "crosswind",
    "joint_no_alongwind": "along_field_wind",
    "joint_no_temperature_context": "temperature_context",
    "joint_no_local_wind_context": "local_wind_context",
    "joint_no_latitude_context": "latitude_context",
}

WINDOW_SPECS: dict[str, int | None] = {
    "expanding": None,
    "rolling_6_seasons": 6,
    "rolling_4_seasons": 4,
}


def _training_slice(
    df: pd.DataFrame,
    test_season: int,
    window_seasons: int | None,
) -> pd.DataFrame:
    train = df[df["season"] < test_season].copy()
    if window_seasons is not None:
        train = train[train["season"] >= test_season - window_seasons].copy()
    return train


def walk_forward_models(
    df: pd.DataFrame,
    model_features: dict[str, list[str]],
    window_seasons: int | None = None,
    min_train_games: int = 1_000,
    min_test_games: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_nums, cats = feature_lists(df)
    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for season in sorted(df["season"].dropna().astype(int).unique()):
        train = _training_slice(df, season, window_seasons)
        test = df[df["season"] == season].copy()
        if len(train) < min_train_games or len(test) < min_test_games:
            continue

        reference = build_context_reference(train)
        train = add_joint_interactions(apply_context_features(train, reference))
        test = add_joint_interactions(apply_context_features(test, reference))

        scored = test[test["joint_ready"]].copy()
        if scored.empty:
            continue

        train = prep_features(train, cats)
        test = prep_features(test, cats)
        scored = test.loc[scored.index].copy()

        keep = [
            c
            for c in [
                "season",
                "week",
                "game_id",
                "start_date",
                "away_team",
                "home_team",
                "venue_id",
                "venue_name",
                "venue_latitude",
                "latitude_band",
                "closing_total",
                "actual_total_points",
                "market_residual",
                "temperature_f",
                "temperature_anomaly_f",
                "wind_mph",
                "wind_direction_degrees",
                "wind_local_percentile",
                "field_axis_deg",
                "wind_field_angle_deg",
                "crosswind_mph",
                "alongwind_mph",
            ]
            if c in scored.columns
        ]
        fold = scored[keep].copy()

        for model_name, additions in model_features.items():
            nums = list(dict.fromkeys(base_nums + additions))
            model = reg_models(nums, cats)["hist_gradient_boosting"]
            model.fit(train[nums + cats], train["market_residual"])
            prediction = model.predict(scored[nums + cats])
            fold[f"{model_name}_pred_market_residual"] = prediction
            error = prediction - scored["market_residual"].to_numpy(float)
            diagnostics.append(
                {
                    "test_season": season,
                    "window_seasons": (
                        "expanding" if window_seasons is None else window_seasons
                    ),
                    "model": model_name,
                    "train_games": len(train),
                    "train_seasons": int(train["season"].nunique()),
                    "test_games": len(test),
                    "paired_joint_games": len(scored),
                    "numeric_features": len(nums),
                    "categorical_features": len(cats),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "signed_projection_bias": float(np.mean(error)),
                }
            )
        predictions.append(fold)

    if not predictions:
        return pd.DataFrame(), pd.DataFrame(diagnostics)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(diagnostics)


def _paired_delta(
    predictions: pd.DataFrame,
    challenger: str,
    reference: str,
) -> np.ndarray:
    actual = predictions["market_residual"].to_numpy(float)
    challenger_abs = np.abs(
        predictions[f"{challenger}_pred_market_residual"].to_numpy(float) - actual
    )
    reference_abs = np.abs(
        predictions[f"{reference}_pred_market_residual"].to_numpy(float) - actual
    )
    return challenger_abs - reference_abs


def _season_improvements(
    predictions: pd.DataFrame,
    challenger: str,
    reference: str,
) -> int:
    improved = 0
    for _, group in predictions.groupby("season", observed=True):
        delta = _paired_delta(group, challenger, reference)
        improved += int(float(delta.mean()) < 0)
    return improved


def _comparison_row(
    predictions: pd.DataFrame,
    challenger: str,
    reference: str,
    seed_offset: int,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    delta = _paired_delta(predictions, challenger, reference)
    seasons = int(predictions["season"].nunique())
    required = max(1, math.ceil(seasons * 0.70))
    game_low, game_high = bootstrap_mean_ci(
        delta,
        reps=bootstrap_reps,
        seed=BOOTSTRAP_SEED + seed_offset,
    )
    season_low, season_high = season_cluster_ci(
        predictions,
        challenger,
        reference,
        reps=bootstrap_reps,
        seed=BOOTSTRAP_SEED + seed_offset + 100,
    )
    improved = _season_improvements(predictions, challenger, reference)
    supported = (
        float(delta.mean()) < 0
        and game_high < 0
        and season_high < 0
        and improved >= required
    )
    return {
        "paired_games": len(predictions),
        "test_seasons": seasons,
        "mean_mae_delta_challenger_minus_reference": float(delta.mean()),
        "game_bootstrap_ci_low": game_low,
        "game_bootstrap_ci_high": game_high,
        "season_cluster_ci_low": season_low,
        "season_cluster_ci_high": season_high,
        "test_seasons_improved": improved,
        "seasons_required_for_support": required,
        "evidence_status": (
            "SUPPORTED_RETROSPECTIVELY" if supported else "NOT_PROVEN"
        ),
    }


def ablation_summary(
    predictions: pd.DataFrame,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for order, (ablation, component) in enumerate(ABLATION_COMPONENT.items()):
        row = _comparison_row(
            predictions,
            "joint_core",
            ablation,
            seed_offset=2_000 + order,
            bootstrap_reps=bootstrap_reps,
        )
        contribution = -row["mean_mae_delta_challenger_minus_reference"]
        component_game_low = -row["game_bootstrap_ci_high"]
        component_game_high = -row["game_bootstrap_ci_low"]
        component_season_low = -row["season_cluster_ci_high"]
        component_season_high = -row["season_cluster_ci_low"]
        seasons_joint_better = row["test_seasons_improved"]
        required = row["seasons_required_for_support"]

        component_supported = (
            contribution > 0
            and component_game_low > 0
            and component_season_low > 0
            and seasons_joint_better >= required
        )
        rows.append(
            {
                "removed_component": component,
                "ablation_model": ablation,
                "paired_games": row["paired_games"],
                "test_seasons": row["test_seasons"],
                "mae_penalty_when_removed": contribution,
                "component_game_ci_low": component_game_low,
                "component_game_ci_high": component_game_high,
                "component_season_ci_low": component_season_low,
                "component_season_ci_high": component_season_high,
                "test_seasons_joint_core_better": seasons_joint_better,
                "seasons_required_for_component_support": required,
                "component_evidence_status": (
                    "COMPONENT_SUPPORTED"
                    if component_supported
                    else "COMPONENT_NOT_ISOLATED"
                ),
            }
        )
    return pd.DataFrame(rows)


def leave_one_test_season_out(
    predictions: pd.DataFrame,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    seasons = sorted(predictions["season"].dropna().astype(int).unique())
    for order, omitted in enumerate(seasons):
        subset = predictions[predictions["season"].astype(int) != omitted].copy()
        row = _comparison_row(
            subset,
            "joint_core",
            "baseline",
            seed_offset=3_000 + order,
            bootstrap_reps=bootstrap_reps,
        )
        row["omitted_test_season"] = omitted
        rows.append(row)
    cols = ["omitted_test_season"] + [c for c in rows[0] if c != "omitted_test_season"]
    return pd.DataFrame(rows)[cols]


def window_summary(
    window_predictions: dict[str, pd.DataFrame],
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for order, (window_name, predictions) in enumerate(window_predictions.items()):
        if predictions.empty:
            continue
        row = _comparison_row(
            predictions,
            "joint_core",
            "baseline",
            seed_offset=4_000 + order,
            bootstrap_reps=bootstrap_reps,
        )
        row["training_window"] = window_name
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    cols = ["training_window"] + [c for c in rows[0] if c != "training_window"]
    return pd.DataFrame(rows)[cols]


def robustness_indicators(
    windows: pd.DataFrame,
    loso: pd.DataFrame,
) -> pd.DataFrame:
    if windows.empty or loso.empty:
        return pd.DataFrame(
            [
                {
                    "robustness_status": "NOT_EVALUATED",
                    "all_window_mean_deltas_negative": False,
                    "all_loso_mean_deltas_negative": False,
                    "all_windows_improve_70pct_seasons": False,
                    "windows_with_season_ci_below_zero": 0,
                    "windows_with_full_evidence_support": 0,
                    "worst_loso_mean_delta": np.nan,
                }
            ]
        )

    all_window_negative = bool(
        (windows["mean_mae_delta_challenger_minus_reference"] < 0).all()
    )
    all_loso_negative = bool(
        (loso["mean_mae_delta_challenger_minus_reference"] < 0).all()
    )
    all_windows_stable = bool(
        (
            windows["test_seasons_improved"]
            >= windows["seasons_required_for_support"]
        ).all()
    )
    season_ci_count = int((windows["season_cluster_ci_high"] < 0).sum())
    supported_windows = int(
        (windows["evidence_status"] == "SUPPORTED_RETROSPECTIVELY").sum()
    )
    worst_loso = float(
        loso["mean_mae_delta_challenger_minus_reference"].max()
    )

    strengthened = (
        all_window_negative
        and all_loso_negative
        and all_windows_stable
        and season_ci_count >= 2
    )
    return pd.DataFrame(
        [
            {
                "robustness_status": (
                    "ROBUSTNESS_STRENGTHENED"
                    if strengthened
                    else "ROBUSTNESS_MIXED"
                ),
                "all_window_mean_deltas_negative": all_window_negative,
                "all_loso_mean_deltas_negative": all_loso_negative,
                "all_windows_improve_70pct_seasons": all_windows_stable,
                "windows_with_season_ci_below_zero": season_ci_count,
                "windows_with_full_evidence_support": supported_windows,
                "worst_loso_mean_delta": worst_loso,
            }
        ]
    )


def write_summary(
    ablations: pd.DataFrame,
    windows: pd.DataFrame,
    loso: pd.DataFrame,
    indicators: pd.DataFrame,
) -> None:
    status = (
        str(indicators.iloc[0]["robustness_status"])
        if not indicators.empty
        else "NOT_EVALUATED"
    )
    lines = [
        "# Joint-Core Robustness and Ablation Research",
        "",
        "**Status: retrospective research only. No production, live-board, weekly-pick, orientation-shadow, or prospective-ledger effect.**",
        "",
        "## Purpose",
        "",
        "Stress-test the previously observed `joint_core` signal without changing the original evidence gate. This study asks which feature groups matter, whether the signal depends on any one test season, and whether it survives different historical training windows.",
        "",
        "The original joint-weather result remains `NOT_PROVEN` unless its original predeclared gate is met. This robustness study cannot retroactively move that goalpost.",
        "",
        "## Predeclared robustness checks",
        "",
        "1. Grouped and component ablations from `joint_core`.",
        "2. Leave-one-test-season-out sensitivity.",
        "3. Expanding-history, rolling-six-season, and rolling-four-season walk-forward training windows.",
        "4. The same paired MAE, game bootstrap, season-cluster uncertainty, and 70% season-stability framework used in the parent study.",
        "",
        "For component ablations, a positive `mae_penalty_when_removed` means the full `joint_core` model performed better after that component was restored. Component support requires both uncertainty intervals to remain above zero and the full model to win in at least 70% of seasons.",
        "",
        "The secondary `ROBUSTNESS_STRENGTHENED` label requires all training-window mean deltas to favor `joint_core`, every leave-one-season-out mean delta to remain favorable, every window to improve at least 70% of test seasons, and at least two training-window season-cluster intervals to be entirely below zero. It is not a production-promotion label.",
        "",
        f"Current robustness status: **{status}**",
        "",
        "## Component ablations",
        "",
        ablations.to_markdown(index=False)
        if not ablations.empty
        else "_No ablation results were produced._",
        "",
        "## Alternate temporal training windows",
        "",
        windows.to_markdown(index=False)
        if not windows.empty
        else "_No alternate-window results were produced._",
        "",
        "## Leave-one-test-season-out sensitivity",
        "",
        loso.to_markdown(index=False)
        if not loso.empty
        else "_No leave-one-out results were produced._",
        "",
        "## Robustness indicators",
        "",
        indicators.to_markdown(index=False)
        if not indicators.empty
        else "_No robustness indicators were produced._",
        "",
        "## Guardrails",
        "",
        "- These tests are retrospective and do not alter the frozen 2026 live protocol.",
        "- A component can be useful jointly without being individually identifiable because HGB features can be redundant and nonlinear.",
        "- Rolling-window sensitivity tests whether the result depends on older training history; it does not select a new production training window.",
        "- Leave-one-test-season-out sensitivity checks whether one evaluation season carries the result; it does not refit after removing that season from historical training.",
        "- Even a strengthened robustness result only supports considering a separately frozen prospective shadow challenger.",
    ]
    ensure_dir(OUTPUT_DIR).joinpath("summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    venues = read_df("data/reference/stadium_locations.csv")
    raw = read_df("data/processed/modeling_dataset.csv")
    df = prepare_research_data(raw, venues)
    df = add_orientation_features(df)

    expanding_predictions, expanding_diagnostics = walk_forward_models(
        df,
        ABLATION_FEATURES,
        window_seasons=None,
    )
    ablations = ablation_summary(expanding_predictions)
    loso = leave_one_test_season_out(expanding_predictions)

    window_predictions: dict[str, pd.DataFrame] = {
        "expanding": expanding_predictions[
            [
                c
                for c in expanding_predictions.columns
                if c
                not in [
                    col
                    for col in expanding_predictions.columns
                    if col.endswith("_pred_market_residual")
                    and not (
                        col.startswith("baseline_")
                        or col.startswith("joint_core_")
                    )
                ]
            ]
        ].copy()
    }
    window_diagnostics = [expanding_diagnostics.assign(training_window="expanding")]

    for window_name, window_seasons in WINDOW_SPECS.items():
        if window_name == "expanding":
            continue
        preds, diagnostics = walk_forward_models(
            df,
            {
                "baseline": ABLATION_FEATURES["baseline"],
                "joint_core": ABLATION_FEATURES["joint_core"],
            },
            window_seasons=window_seasons,
        )
        window_predictions[window_name] = preds
        window_diagnostics.append(diagnostics.assign(training_window=window_name))

    windows = window_summary(window_predictions)
    indicators = robustness_indicators(windows, loso)

    write_df(ablations, f"{OUTPUT_DIR}/component_ablations.csv")
    write_df(loso, f"{OUTPUT_DIR}/leave_one_test_season_out.csv")
    write_df(windows, f"{OUTPUT_DIR}/training_window_summary.csv")
    write_df(indicators, f"{OUTPUT_DIR}/robustness_indicators.csv")
    write_df(
        pd.concat(window_diagnostics, ignore_index=True),
        f"{OUTPUT_DIR}/walk_forward_diagnostics.csv",
    )
    for window_name, preds in window_predictions.items():
        write_df(
            preds,
            f"{OUTPUT_DIR}/{window_name}_predictions.csv",
        )
    write_summary(ablations, windows, loso, indicators)
    print("Wrote isolated joint-core robustness research outputs")


if __name__ == "__main__":
    main()
