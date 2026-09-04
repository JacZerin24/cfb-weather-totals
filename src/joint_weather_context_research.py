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
from .model_bakeoff import feature_lists, prep_features, reg_models
from .utils import ensure_dir, read_df, write_df

ORIENTATION_PATH = "data/reference/stadium_orientations.csv"
OUTPUT_DIR = "outputs/joint_weather_context"
MIN_REGIME_GAMES = 50

CLIMATE_FEATURES = [
    "venue_latitude",
    "temperature_anomaly_f",
    "wind_local_percentile",
    "temperature_latitude_interaction",
    "wind_latitude_interaction",
]

MODEL_FEATURES = {
    "baseline": [],
    "orientation_crosswind": ["crosswind_mph"],
    "orientation_vector": ["crosswind_mph", "alongwind_mph"],
    "climate_full": CLIMATE_FEATURES,
    "joint_core": [
        "crosswind_mph",
        "alongwind_mph",
        *CLIMATE_FEATURES,
    ],
    "joint_interactions": [
        "crosswind_mph",
        "alongwind_mph",
        *CLIMATE_FEATURES,
        "crosswind_temperature_interaction",
        "alongwind_temperature_interaction",
        "crosswind_local_wind_interaction",
        "alongwind_local_wind_interaction",
        "crosswind_latitude_interaction",
        "alongwind_latitude_interaction",
        "crosswind_market_total_interaction",
    ],
}

PAIRWISE_COMPARISONS = [
    ("orientation_crosswind", "baseline"),
    ("orientation_vector", "orientation_crosswind"),
    ("climate_full", "baseline"),
    ("joint_core", "orientation_vector"),
    ("joint_core", "climate_full"),
    ("joint_interactions", "joint_core"),
    ("joint_interactions", "baseline"),
]


def field_angle(wind_direction: np.ndarray, field_axis: np.ndarray) -> np.ndarray:
    wind_axis = np.mod(wind_direction, 180.0)
    axis = np.mod(field_axis, 180.0)
    delta = np.abs(wind_axis - axis)
    return np.minimum(delta, 180.0 - delta)


def orientation_table() -> pd.DataFrame:
    orientation = read_df(ORIENTATION_PATH).copy()
    keep = [
        c
        for c in ["venue_id", "field_axis_deg", "axis_uncertainty_deg", "roof_behavior"]
        if c in orientation.columns
    ]
    orientation = orientation[keep].copy()
    for col in ["venue_id", "field_axis_deg", "axis_uncertainty_deg"]:
        if col not in orientation.columns:
            orientation[col] = np.nan
        orientation[col] = pd.to_numeric(orientation[col], errors="coerce")
    return orientation.drop_duplicates("venue_id")


def add_orientation_features(
    df: pd.DataFrame, orientation: pd.DataFrame | None = None
) -> pd.DataFrame:
    out = df.copy()
    orientation = orientation_table() if orientation is None else orientation.copy()
    out["venue_id"] = pd.to_numeric(out.get("venue_id"), errors="coerce")
    for col in ["wind_mph", "wind_direction_degrees"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    stale = [
        c
        for c in ["field_axis_deg", "axis_uncertainty_deg", "roof_behavior"]
        if c in out.columns
    ]
    out = out.drop(columns=stale, errors="ignore").merge(
        orientation, on="venue_id", how="left"
    )

    out["wind_field_angle_deg"] = np.nan
    out["crosswind_mph"] = np.nan
    out["alongwind_mph"] = np.nan
    valid = out[
        ["wind_mph", "wind_direction_degrees", "field_axis_deg"]
    ].notna().all(axis=1)
    if valid.any():
        angle = field_angle(
            out.loc[valid, "wind_direction_degrees"].to_numpy(float),
            out.loc[valid, "field_axis_deg"].to_numpy(float),
        )
        wind = out.loc[valid, "wind_mph"].to_numpy(float)
        out.loc[valid, "wind_field_angle_deg"] = angle
        out.loc[valid, "crosswind_mph"] = wind * np.sin(np.deg2rad(angle))
        out.loc[valid, "alongwind_mph"] = wind * np.cos(np.deg2rad(angle))

    out["orientation_ready"] = (
        out["outdoor"]
        & out["field_axis_deg"].notna()
        & out["wind_direction_degrees"].notna()
        & out["wind_mph"].notna()
    )
    out["joint_ready"] = out["context_ready"] & out["orientation_ready"]
    if "fbs_vs_fbs" in out.columns:
        out["joint_ready"] &= out["fbs_vs_fbs"].astype(bool)
    return out


def add_joint_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    latitude_scaled = (pd.to_numeric(out["venue_latitude"], errors="coerce") - 35.0) / 5.0
    temp_scaled = pd.to_numeric(out["temperature_anomaly_f"], errors="coerce") / 10.0
    wind_context = pd.to_numeric(out["wind_local_percentile"], errors="coerce") - 0.5
    total_scaled = (pd.to_numeric(out["closing_total"], errors="coerce") - 56.0) / 10.0

    out["crosswind_temperature_interaction"] = out["crosswind_mph"] * temp_scaled
    out["alongwind_temperature_interaction"] = out["alongwind_mph"] * temp_scaled
    out["crosswind_local_wind_interaction"] = out["crosswind_mph"] * wind_context
    out["alongwind_local_wind_interaction"] = out["alongwind_mph"] * wind_context
    out["crosswind_latitude_interaction"] = out["crosswind_mph"] * latitude_scaled
    out["alongwind_latitude_interaction"] = out["alongwind_mph"] * latitude_scaled
    out["crosswind_market_total_interaction"] = out["crosswind_mph"] * total_scaled
    return out


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    def summarize(label: str, group: pd.DataFrame) -> dict[str, Any]:
        return {
            "scope": label,
            "games": len(group),
            "outdoor_games": int(group["outdoor"].sum()),
            "fbs_vs_fbs_games": int(group["fbs_vs_fbs"].sum()),
            "context_ready_games": int(group["context_ready"].sum()),
            "orientation_ready_games": int(group["orientation_ready"].sum()),
            "joint_ready_games": int(group["joint_ready"].sum()),
            "joint_ready_pct_of_fbs": (
                float(group["joint_ready"].sum() / group["fbs_vs_fbs"].sum())
                if group["fbs_vs_fbs"].sum()
                else np.nan
            ),
            "unique_joint_ready_venues": int(
                group.loc[group["joint_ready"], "venue_id"].nunique(dropna=True)
            ),
        }

    rows = [summarize("overall", df)]
    for season, group in df.groupby("season", observed=True):
        rows.append(summarize(str(int(season)), group))
    return pd.DataFrame(rows)


def walk_forward_predictions(
    df: pd.DataFrame,
    min_train_games: int = 1_000,
    min_test_games: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_nums, cats = feature_lists(df)
    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for season in sorted(df["season"].dropna().astype(int).unique()):
        train = df[df["season"] < season].copy()
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
                "axis_uncertainty_deg",
                "wind_field_angle_deg",
                "crosswind_mph",
                "alongwind_mph",
            ]
            if c in scored.columns
        ]
        fold = scored[keep].copy()

        for model_name, additions in MODEL_FEATURES.items():
            nums = list(dict.fromkeys(base_nums + additions))
            model = reg_models(nums, cats)["hist_gradient_boosting"]
            model.fit(train[nums + cats], train["market_residual"])
            prediction = model.predict(scored[nums + cats])
            fold[f"{model_name}_pred_market_residual"] = prediction
            error = prediction - scored["market_residual"].to_numpy(float)
            diagnostics.append(
                {
                    "test_season": season,
                    "model": model_name,
                    "train_games": len(train),
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


def season_cluster_ci(
    predictions: pd.DataFrame,
    challenger: str,
    reference: str,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED + 100,
) -> tuple[float, float]:
    if predictions.empty:
        return np.nan, np.nan
    season_deltas: list[float] = []
    for _, group in predictions.groupby("season", observed=True):
        actual = group["market_residual"].to_numpy(float)
        challenger_abs = np.abs(
            group[f"{challenger}_pred_market_residual"].to_numpy(float) - actual
        )
        reference_abs = np.abs(
            group[f"{reference}_pred_market_residual"].to_numpy(float) - actual
        )
        season_deltas.append(float(np.mean(challenger_abs - reference_abs)))
    return bootstrap_mean_ci(
        np.asarray(season_deltas, dtype=float), reps=reps, seed=seed
    )


def _under_qualifier_metrics(
    predictions: pd.DataFrame, prediction_col: str
) -> dict[str, Any]:
    qualifiers = predictions[
        (predictions[prediction_col] <= -3.5)
        & (predictions["closing_total"] >= 56.0)
    ].copy()
    wins = int((qualifiers["actual_total_points"] < qualifiers["closing_total"]).sum())
    losses = int(
        (qualifiers["actual_total_points"] > qualifiers["closing_total"]).sum()
    )
    pushes = int(
        (qualifiers["actual_total_points"] == qualifiers["closing_total"]).sum()
    )
    graded = wins + losses
    units = wins * (100 / 110) - losses
    return {
        "qualifiers": len(qualifiers),
        "qualifier_wins": wins,
        "qualifier_losses": losses,
        "qualifier_pushes": pushes,
        "qualifier_hit_rate": wins / graded if graded else np.nan,
        "qualifier_roi_per_1u": units / len(qualifiers) if len(qualifiers) else np.nan,
    }


def paired_model_summary(
    predictions: pd.DataFrame,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    actual = predictions["market_residual"].to_numpy(float)
    baseline_pred = predictions["baseline_pred_market_residual"].to_numpy(float)
    baseline_abs = np.abs(baseline_pred - actual)
    seasons = int(predictions["season"].nunique())
    season_requirement = max(1, math.ceil(seasons * 0.70))
    rows: list[dict[str, Any]] = []

    for order, model_name in enumerate(MODEL_FEATURES):
        col = f"{model_name}_pred_market_residual"
        predicted = predictions[col].to_numpy(float)
        error = predicted - actual
        abs_error = np.abs(error)
        delta = abs_error - baseline_abs

        if model_name == "baseline":
            game_low = game_high = season_low = season_high = 0.0
            improved = 0
            evidence = "REFERENCE"
        else:
            game_low, game_high = bootstrap_mean_ci(
                delta, reps=bootstrap_reps, seed=BOOTSTRAP_SEED + order
            )
            season_low, season_high = season_cluster_ci(
                predictions,
                model_name,
                "baseline",
                reps=bootstrap_reps,
                seed=BOOTSTRAP_SEED + 100 + order,
            )
            improved = 0
            for _, group in predictions.groupby("season", observed=True):
                y = group["market_residual"].to_numpy(float)
                base_mae = np.abs(
                    group["baseline_pred_market_residual"].to_numpy(float) - y
                ).mean()
                model_mae = np.abs(
                    group[col].to_numpy(float) - y
                ).mean()
                improved += int(model_mae < base_mae)
            evidence = (
                "SUPPORTED_RETROSPECTIVELY"
                if float(delta.mean()) < 0
                and game_high < 0
                and season_high < 0
                and improved >= season_requirement
                else "NOT_PROVEN"
            )

        row = {
            "model": model_name,
            "paired_games": len(predictions),
            "test_seasons": seasons,
            "mae": float(abs_error.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "signed_projection_bias": float(error.mean()),
            "mae_delta_vs_baseline": float(delta.mean()),
            "game_bootstrap_ci_low": game_low,
            "game_bootstrap_ci_high": game_high,
            "season_cluster_ci_low": season_low,
            "season_cluster_ci_high": season_high,
            "test_seasons_improved_vs_baseline": improved,
            "seasons_required_for_support": season_requirement,
            "evidence_status": evidence,
        }
        row.update(_under_qualifier_metrics(predictions, col))
        rows.append(row)
    return pd.DataFrame(rows)


def incremental_comparisons(
    predictions: pd.DataFrame,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    seasons = int(predictions["season"].nunique())
    required = max(1, math.ceil(seasons * 0.70))
    for order, (challenger, reference) in enumerate(PAIRWISE_COMPARISONS):
        actual = predictions["market_residual"].to_numpy(float)
        challenger_abs = np.abs(
            predictions[f"{challenger}_pred_market_residual"].to_numpy(float) - actual
        )
        reference_abs = np.abs(
            predictions[f"{reference}_pred_market_residual"].to_numpy(float) - actual
        )
        delta = challenger_abs - reference_abs
        game_low, game_high = bootstrap_mean_ci(
            delta, reps=bootstrap_reps, seed=BOOTSTRAP_SEED + 500 + order
        )
        season_low, season_high = season_cluster_ci(
            predictions,
            challenger,
            reference,
            reps=bootstrap_reps,
            seed=BOOTSTRAP_SEED + 600 + order,
        )
        improved = 0
        for _, group in predictions.groupby("season", observed=True):
            y = group["market_residual"].to_numpy(float)
            ch = np.abs(
                group[f"{challenger}_pred_market_residual"].to_numpy(float) - y
            ).mean()
            ref = np.abs(
                group[f"{reference}_pred_market_residual"].to_numpy(float) - y
            ).mean()
            improved += int(ch < ref)
        supported = (
            float(delta.mean()) < 0
            and game_high < 0
            and season_high < 0
            and improved >= required
        )
        rows.append(
            {
                "challenger": challenger,
                "reference": reference,
                "paired_games": len(predictions),
                "mae_delta_challenger_minus_reference": float(delta.mean()),
                "game_bootstrap_ci_low": game_low,
                "game_bootstrap_ci_high": game_high,
                "season_cluster_ci_low": season_low,
                "season_cluster_ci_high": season_high,
                "test_seasons_improved": improved,
                "test_seasons": seasons,
                "seasons_required_for_support": required,
                "incremental_evidence_status": (
                    "SUPPORTED_RETROSPECTIVELY" if supported else "NOT_PROVEN"
                ),
            }
        )
    return pd.DataFrame(rows)


def model_by_season(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for season, group in predictions.groupby("season", observed=True):
        actual = group["market_residual"].to_numpy(float)
        baseline_abs = np.abs(
            group["baseline_pred_market_residual"].to_numpy(float) - actual
        )
        for model_name in MODEL_FEATURES:
            pred = group[f"{model_name}_pred_market_residual"].to_numpy(float)
            error = pred - actual
            abs_error = np.abs(error)
            rows.append(
                {
                    "test_season": int(season),
                    "model": model_name,
                    "paired_games": len(group),
                    "mae": float(abs_error.mean()),
                    "mae_delta_vs_baseline": float((abs_error - baseline_abs).mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "signed_projection_bias": float(error.mean()),
                }
            )
    return pd.DataFrame(rows)


def regime_stability(
    predictions: pd.DataFrame,
    challenger: str = "joint_interactions",
    min_games: int = MIN_REGIME_GAMES,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df["crosswind_bin"] = pd.cut(
        df["crosswind_mph"],
        [-np.inf, 5, 10, 15, 20, np.inf],
        labels=["0-5", "5-10", "10-15", "15-20", "20+"],
    )
    df["temperature_anomaly_bin"] = pd.cut(
        df["temperature_anomaly_f"],
        [-np.inf, -10, 0, 10, np.inf],
        labels=["<=-10F", "-10-0F", "0-10F", "10F+"],
    )
    df["local_wind_percentile_bin"] = pd.cut(
        df["wind_local_percentile"],
        [-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["<=25th", "25-50th", "50-75th", "75th+"],
    )
    df["closing_total_bin"] = pd.cut(
        df["closing_total"],
        [-np.inf, 48, 56, 64, np.inf],
        labels=["<48", "48-56", "56-64", "64+"],
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for grouping in [
        "latitude_band",
        "crosswind_bin",
        "temperature_anomaly_bin",
        "local_wind_percentile_bin",
        "closing_total_bin",
    ]:
        for value, group in df.dropna(subset=[grouping]).groupby(grouping, observed=True):
            if len(group) < min_games:
                continue
            actual = group["market_residual"].to_numpy(float)
            baseline_abs = np.abs(
                group["baseline_pred_market_residual"].to_numpy(float) - actual
            )
            challenger_abs = np.abs(
                group[f"{challenger}_pred_market_residual"].to_numpy(float) - actual
            )
            delta = challenger_abs - baseline_abs
            low, high = bootstrap_mean_ci(
                delta,
                reps=min(5000, BOOTSTRAP_REPS),
                seed=BOOTSTRAP_SEED + len(rows) + 1000,
            )
            rows.append(
                {
                    "challenger": challenger,
                    "grouping": grouping,
                    "value": str(value),
                    "games": len(group),
                    "seasons": int(group["season"].nunique()),
                    "baseline_mae": float(baseline_abs.mean()),
                    "challenger_mae": float(challenger_abs.mean()),
                    "mae_delta_vs_baseline": float(delta.mean()),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    coverage: pd.DataFrame,
    model_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    by_season: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:
    overall = coverage.iloc[0] if not coverage.empty else pd.Series(dtype=object)
    joint_row = (
        model_summary[model_summary["model"] == "joint_interactions"].iloc[0]
        if not model_summary.empty
        and (model_summary["model"] == "joint_interactions").any()
        else pd.Series(dtype=object)
    )
    evidence = str(joint_row.get("evidence_status", "NOT_EVALUATED"))
    lines = [
        "# Joint Weather-Context Research",
        "",
        "**Status: retrospective research only. No production, live-board, weekly-pick, orientation-shadow, or prospective-ledger effect.**",
        "",
        "## Research question",
        "",
        "Does combining the existing GENERAL HGB weather/matchup model with field-relative wind, local climate context, latitude, and physically motivated interaction terms improve prediction of `market_residual = actual_total_points - closing_total` out of sample?",
        "",
        "## Isolation boundary",
        "",
        "- This script reads the historical modeling artifact and reference tables only.",
        "- It writes only to `outputs/joint_weather_context/`.",
        "- It does not call the live weekly runner, change production model features, modify thresholds, write the official 2026 prospective ledger, or change the existing orientation shadow.",
        "- Retrospective evidence can only justify considering a separately versioned future challenger. Nothing here promotes itself.",
        "",
        "## Paired-data coverage",
        "",
        f"Joint orientation-and-climate-ready FBS games: **{int(overall.get('joint_ready_games', 0)):,}**",
        "",
        coverage.to_markdown(index=False)
        if not coverage.empty
        else "_No coverage rows were produced._",
        "",
        "## Predeclared model ladder",
        "",
        "1. `baseline`: existing GENERAL HGB features.",
        "2. `orientation_crosswind`: baseline + field-relative crosswind.",
        "3. `orientation_vector`: crosswind + along-field wind.",
        "4. `climate_full`: latitude + leak-safe local temperature/wind context and latitude interactions.",
        "5. `joint_core`: orientation vector + climate context together.",
        "6. `joint_interactions`: joint core + a limited set of physically motivated orientation × climate/market interactions.",
        "",
        "All models are evaluated on the exact same joint-ready games in each test season. Every test season is predicted using only prior seasons, and local weather baselines are fitted only from those prior seasons.",
        "",
        "## Evidence gate",
        "",
        "A challenger is labeled `SUPPORTED_RETROSPECTIVELY` only when all of the following hold against its reference:",
        "",
        "- mean paired MAE delta is negative;",
        "- the 95% game-level paired bootstrap interval is entirely below zero;",
        "- the 95% season-cluster interval is entirely below zero;",
        "- it improves in at least 70% of evaluated test seasons.",
        "",
        "Qualifier hit rate and ROI are secondary context and cannot satisfy this gate. Even a supported retrospective result still requires a new frozen prospective shadow before any production discussion.",
        "",
        f"Current full-joint retrospective evidence status: **{evidence}**",
        "",
        "## Model comparison vs baseline",
        "",
        model_summary.to_markdown(index=False)
        if not model_summary.empty
        else "_No walk-forward model results were produced._",
        "",
        "## Incremental comparisons",
        "",
        comparisons.to_markdown(index=False)
        if not comparisons.empty
        else "_No incremental comparisons were produced._",
        "",
        "## Results by test season",
        "",
        by_season.to_markdown(index=False)
        if not by_season.empty
        else "_No season-level results were produced._",
        "",
        "## Regime stability for the full joint challenger",
        "",
        regimes.to_markdown(index=False)
        if not regimes.empty
        else "_No regime cells met the minimum sample._",
        "",
        "## Interpretation guardrails",
        "",
        "- HistGradientBoosting can learn nonlinear interactions without manually enumerating every combination. The explicit interaction list is intentionally small to reduce overfit risk.",
        "- Latitude is treated as context, not a causal mechanism.",
        "- `temperature_anomaly_f` and `wind_local_percentile` are historical football-game weather context, not official NOAA climate normals.",
        "- Field orientation is an undirected 0-180 degree axis; crosswind and along-field wind are magnitudes.",
        "- A narrow betting-strategy result is not enough. Prediction error, uncertainty, year-to-year stability, and regime stability are primary.",
        "- No retrospective result may alter the frozen 2026 live protocol or existing prospective records.",
    ]
    out = ensure_dir(OUTPUT_DIR) / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    venues = read_df("data/reference/stadium_locations.csv")
    raw = read_df("data/processed/modeling_dataset.csv")
    df = prepare_research_data(raw, venues)
    df = add_orientation_features(df)

    coverage = coverage_report(df)
    predictions, diagnostics = walk_forward_predictions(df)
    model_summary = paired_model_summary(predictions)
    comparisons = incremental_comparisons(predictions)
    by_season = model_by_season(predictions)
    regimes = regime_stability(predictions)

    write_df(coverage, f"{OUTPUT_DIR}/coverage.csv")
    write_df(predictions, f"{OUTPUT_DIR}/walk_forward_predictions.csv")
    write_df(diagnostics, f"{OUTPUT_DIR}/walk_forward_diagnostics.csv")
    write_df(model_summary, f"{OUTPUT_DIR}/model_summary.csv")
    write_df(comparisons, f"{OUTPUT_DIR}/incremental_comparisons.csv")
    write_df(by_season, f"{OUTPUT_DIR}/model_by_season.csv")
    write_df(regimes, f"{OUTPUT_DIR}/regime_stability.csv")
    write_summary(coverage, model_summary, comparisons, by_season, regimes)
    print("Wrote isolated joint weather-context research outputs")


if __name__ == "__main__":
    main()
