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
    add_orientation_features,
    field_angle,
    orientation_table,
    season_cluster_ci,
)
from .model_bakeoff import feature_lists, prep_features, reg_models
from .utils import ensure_dir, read_df, write_df

OUTPUT_DIR = "outputs/field_geometry_confirmation"
PLACEBO_COUNT = 39
PLACEBO_SEED = 20260904

CONFIRM_MODELS: dict[str, list[str]] = {
    "baseline": [],
    "along_magnitude": ["alongwind_mph"],
    "cross_magnitude": ["crosswind_mph"],
    "vector_magnitude": ["crosswind_mph", "alongwind_mph"],
    "along_alignment": ["along_alignment"],
    "alignment_pair": ["along_alignment", "cross_alignment"],
    "joint_core": ["crosswind_mph", "alongwind_mph", *CLIMATE_FEATURES],
}

PRIMARY_COMPARISONS = [
    ("along_magnitude", "baseline"),
    ("cross_magnitude", "baseline"),
    ("vector_magnitude", "baseline"),
    ("along_alignment", "baseline"),
    ("alignment_pair", "baseline"),
    ("joint_core", "baseline"),
]


def add_alignment_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    angle = pd.to_numeric(out.get("wind_field_angle_deg"), errors="coerce")
    out["along_alignment"] = np.cos(np.deg2rad(angle))
    out["cross_alignment"] = np.sin(np.deg2rad(angle))
    return out


def _deranged_indices(size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 1:
        return np.arange(size)
    base = np.arange(size)
    for _ in range(1000):
        candidate = rng.permutation(size)
        if not np.any(candidate == base):
            return candidate
    return np.roll(base, 1)


def placebo_orientation_maps(
    orientation: pd.DataFrame,
    count: int = PLACEBO_COUNT,
    seed: int = PLACEBO_SEED,
) -> list[pd.DataFrame]:
    valid = orientation.dropna(subset=["venue_id", "field_axis_deg"])[
        ["venue_id", "field_axis_deg"]
    ].drop_duplicates("venue_id")
    valid = valid.sort_values("venue_id").reset_index(drop=True)
    axes = valid["field_axis_deg"].to_numpy(float)
    rng = np.random.default_rng(seed)
    maps: list[pd.DataFrame] = []
    for placebo_id in range(count):
        perm = _deranged_indices(len(valid), rng)
        maps.append(
            pd.DataFrame(
                {
                    "venue_id": valid["venue_id"].to_numpy(),
                    "placebo_axis_deg": axes[perm],
                    "placebo_id": placebo_id,
                }
            )
        )
    return maps


def add_placebo_alongwind(
    df: pd.DataFrame,
    axis_map: pd.DataFrame,
    output_col: str = "placebo_alongwind_mph",
) -> pd.DataFrame:
    out = df.copy()
    mapping = axis_map.drop_duplicates("venue_id").set_index("venue_id")[
        "placebo_axis_deg"
    ]
    out["placebo_axis_deg"] = pd.to_numeric(
        out["venue_id"].map(mapping), errors="coerce"
    )
    out[output_col] = np.nan
    valid = out[
        ["wind_mph", "wind_direction_degrees", "placebo_axis_deg"]
    ].notna().all(axis=1)
    if valid.any():
        angle = field_angle(
            out.loc[valid, "wind_direction_degrees"].to_numpy(float),
            out.loc[valid, "placebo_axis_deg"].to_numpy(float),
        )
        out.loc[valid, output_col] = (
            out.loc[valid, "wind_mph"].to_numpy(float)
            * np.cos(np.deg2rad(angle))
        )
    return out


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


def comparison_row(
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
    improved = 0
    for _, group in predictions.groupby("season", observed=True):
        improved += int(float(_paired_delta(group, challenger, reference).mean()) < 0)
    supported = (
        float(delta.mean()) < 0
        and game_high < 0
        and season_high < 0
        and improved >= required
    )
    return {
        "challenger": challenger,
        "reference": reference,
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


def walk_forward_confirmation(
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
        train = add_alignment_features(apply_context_features(train, reference))
        test = add_alignment_features(apply_context_features(test, reference))
        scored = test[test["joint_ready"]].copy()
        if scored.empty:
            continue

        prior_venues = set(pd.to_numeric(train["venue_id"], errors="coerce").dropna())
        scored["venue_seen_in_prior_train"] = scored["venue_id"].isin(prior_venues)

        train = prep_features(train, cats)
        test = prep_features(test, cats)
        scored = test.loc[scored.index].copy()
        scored["venue_seen_in_prior_train"] = test.loc[
            scored.index, "venue_id"
        ].isin(prior_venues)

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
                "along_alignment",
                "cross_alignment",
                "venue_seen_in_prior_train",
            ]
            if c in scored.columns
        ]
        fold = scored[keep].copy()

        for model_name, additions in CONFIRM_MODELS.items():
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


def comparison_summary(
    predictions: pd.DataFrame,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for order, (challenger, reference) in enumerate(PRIMARY_COMPARISONS):
        rows.append(
            comparison_row(
                predictions,
                challenger,
                reference,
                seed_offset=5_000 + order,
                bootstrap_reps=bootstrap_reps,
            )
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
        for model_name in CONFIRM_MODELS:
            pred = group[f"{model_name}_pred_market_residual"].to_numpy(float)
            error = pred - actual
            rows.append(
                {
                    "test_season": int(season),
                    "model": model_name,
                    "paired_games": len(group),
                    "mae": float(np.mean(np.abs(error))),
                    "mae_delta_vs_baseline": float(
                        np.mean(np.abs(error) - baseline_abs)
                    ),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "signed_projection_bias": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def placebo_walk_forward(
    df: pd.DataFrame,
    orientation: pd.DataFrame,
    real_predictions: pd.DataFrame,
    placebo_count: int = PLACEBO_COUNT,
    seed: int = PLACEBO_SEED,
    min_train_games: int = 1_000,
    min_test_games: int = 100,
) -> pd.DataFrame:
    if real_predictions.empty:
        return pd.DataFrame()
    base_nums, cats = feature_lists(df)
    maps = placebo_orientation_maps(orientation, count=placebo_count, seed=seed)
    rows: list[dict[str, Any]] = []

    baseline_lookup = real_predictions.set_index(["season", "game_id"])[
        "baseline_pred_market_residual"
    ]

    for placebo_id, axis_map in enumerate(maps):
        fold_predictions: list[pd.DataFrame] = []
        for season in sorted(df["season"].dropna().astype(int).unique()):
            train = df[df["season"] < season].copy()
            test = df[df["season"] == season].copy()
            if len(train) < min_train_games or len(test) < min_test_games:
                continue

            train = add_placebo_alongwind(train, axis_map)
            test = add_placebo_alongwind(test, axis_map)
            scored = test[test["joint_ready"]].copy()
            if scored.empty:
                continue

            train = prep_features(train, cats)
            test = prep_features(test, cats)
            scored = test.loc[scored.index].copy()
            nums = list(dict.fromkeys(base_nums + ["placebo_alongwind_mph"]))
            model = reg_models(nums, cats)["hist_gradient_boosting"]
            model.fit(train[nums + cats], train["market_residual"])
            prediction = model.predict(scored[nums + cats])

            fold = scored[["season", "game_id", "market_residual"]].copy()
            fold["placebo_pred_market_residual"] = prediction
            keys = pd.MultiIndex.from_frame(fold[["season", "game_id"]])
            fold["baseline_pred_market_residual"] = baseline_lookup.reindex(keys).to_numpy()
            if fold["baseline_pred_market_residual"].isna().any():
                raise RuntimeError("Placebo cohort did not align with real baseline cohort")
            fold_predictions.append(fold)

        if not fold_predictions:
            continue
        combined = pd.concat(fold_predictions, ignore_index=True)
        actual = combined["market_residual"].to_numpy(float)
        placebo_abs = np.abs(
            combined["placebo_pred_market_residual"].to_numpy(float) - actual
        )
        baseline_abs = np.abs(
            combined["baseline_pred_market_residual"].to_numpy(float) - actual
        )
        delta = placebo_abs - baseline_abs
        improved = 0
        for _, group in combined.groupby("season", observed=True):
            y = group["market_residual"].to_numpy(float)
            p = np.abs(group["placebo_pred_market_residual"].to_numpy(float) - y)
            b = np.abs(group["baseline_pred_market_residual"].to_numpy(float) - y)
            improved += int(float(np.mean(p - b)) < 0)
        rows.append(
            {
                "placebo_id": placebo_id,
                "paired_games": len(combined),
                "test_seasons": int(combined["season"].nunique()),
                "mean_mae_delta_vs_baseline": float(delta.mean()),
                "test_seasons_improved": improved,
            }
        )
    return pd.DataFrame(rows)


def placebo_significance(
    real_predictions: pd.DataFrame,
    placebo_summary: pd.DataFrame,
) -> pd.DataFrame:
    if real_predictions.empty or placebo_summary.empty:
        return pd.DataFrame()
    real_delta = float(_paired_delta(real_predictions, "along_magnitude", "baseline").mean())
    placebo = placebo_summary["mean_mae_delta_vs_baseline"].to_numpy(float)
    as_good_or_better = int(np.sum(placebo <= real_delta))
    randomization_p = (as_good_or_better + 1) / (len(placebo) + 1)
    q05 = float(np.quantile(placebo, 0.05))
    return pd.DataFrame(
        [
            {
                "real_along_mean_mae_delta_vs_baseline": real_delta,
                "placebo_count": len(placebo),
                "placebo_mean_delta": float(np.mean(placebo)),
                "placebo_median_delta": float(np.median(placebo)),
                "placebo_05_quantile_delta": q05,
                "placebos_as_good_or_better_than_real": as_good_or_better,
                "randomization_p_value": randomization_p,
                "real_better_than_95pct_placebos": bool(real_delta < q05),
                "placebo_geometry_status": (
                    "REAL_AXIS_BEATS_PLACEBO"
                    if randomization_p <= 0.05 and real_delta < q05
                    else "REAL_AXIS_NOT_DISTINGUISHED"
                ),
            }
        ]
    )


def confirmation_status(
    comparisons: pd.DataFrame,
    placebo_test: pd.DataFrame,
) -> pd.DataFrame:
    if comparisons.empty or placebo_test.empty:
        return pd.DataFrame([{"confirmation_status": "NOT_EVALUATED"}])
    primary = comparisons[comparisons["challenger"] == "along_magnitude"].iloc[0]
    placebo = placebo_test.iloc[0]
    confirmed = (
        primary["evidence_status"] == "SUPPORTED_RETROSPECTIVELY"
        and placebo["placebo_geometry_status"] == "REAL_AXIS_BEATS_PLACEBO"
    )
    return pd.DataFrame(
        [
            {
                "confirmation_status": (
                    "GEOMETRY_CONFIRMED_RETROSPECTIVELY"
                    if confirmed
                    else "GEOMETRY_NOT_CONFIRMED"
                ),
                "along_model_evidence_status": primary["evidence_status"],
                "placebo_geometry_status": placebo["placebo_geometry_status"],
                "randomization_p_value": placebo["randomization_p_value"],
            }
        ]
    )


def season_context_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for season, group in predictions.groupby("season", observed=True):
        actual = group["market_residual"].to_numpy(float)
        baseline_abs = np.abs(
            group["baseline_pred_market_residual"].to_numpy(float) - actual
        )
        along_abs = np.abs(
            group["along_magnitude_pred_market_residual"].to_numpy(float) - actual
        )
        joint_abs = np.abs(
            group["joint_core_pred_market_residual"].to_numpy(float) - actual
        )
        rows.append(
            {
                "test_season": int(season),
                "games": len(group),
                "unique_venues": int(group["venue_id"].nunique(dropna=True)),
                "pct_venues_seen_in_prior_train": float(
                    group["venue_seen_in_prior_train"].mean()
                ),
                "mean_wind_mph": float(group["wind_mph"].mean()),
                "mean_alongwind_mph": float(group["alongwind_mph"].mean()),
                "mean_crosswind_mph": float(group["crosswind_mph"].mean()),
                "mean_field_angle_deg": float(group["wind_field_angle_deg"].mean()),
                "mean_along_alignment": float(group["along_alignment"].mean()),
                "mean_temperature_f": float(group["temperature_f"].mean()),
                "mean_temperature_anomaly_f": float(group["temperature_anomaly_f"].mean()),
                "mean_local_wind_percentile": float(group["wind_local_percentile"].mean()),
                "mean_closing_total": float(group["closing_total"].mean()),
                "baseline_mae": float(baseline_abs.mean()),
                "along_mae": float(along_abs.mean()),
                "along_mae_delta_vs_baseline": float((along_abs - baseline_abs).mean()),
                "joint_core_mae": float(joint_abs.mean()),
                "joint_core_mae_delta_vs_baseline": float((joint_abs - baseline_abs).mean()),
            }
        )
    return pd.DataFrame(rows)


def season_2022_standardized_shift(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or 2022 not in set(predictions["season"].astype(int)):
        return pd.DataFrame()
    focus = predictions[predictions["season"].astype(int) == 2022]
    other = predictions[predictions["season"].astype(int) != 2022]
    features = [
        "wind_mph",
        "alongwind_mph",
        "crosswind_mph",
        "wind_field_angle_deg",
        "along_alignment",
        "temperature_f",
        "temperature_anomaly_f",
        "wind_local_percentile",
        "closing_total",
    ]
    rows = []
    for feature in features:
        a = pd.to_numeric(focus[feature], errors="coerce").dropna()
        b = pd.to_numeric(other[feature], errors="coerce").dropna()
        pooled = math.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2.0)
        smd = (float(a.mean()) - float(b.mean())) / pooled if pooled > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "season_2022_mean": float(a.mean()),
                "other_seasons_mean": float(b.mean()),
                "standardized_mean_difference_2022_minus_others": smd,
                "absolute_standardized_shift": abs(smd) if pd.notna(smd) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "absolute_standardized_shift", ascending=False
    )


def regime_2022_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df["field_angle_bin"] = pd.cut(
        df["wind_field_angle_deg"],
        [-np.inf, 22.5, 45.0, 67.5, np.inf],
        labels=["0-22.5_parallel", "22.5-45", "45-67.5", "67.5-90_cross"],
        right=False,
    )
    df["wind_speed_bin"] = pd.cut(
        df["wind_mph"],
        [-np.inf, 5, 10, 15, np.inf],
        labels=["0-5", "5-10", "10-15", "15+"],
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for grouping in ["field_angle_bin", "wind_speed_bin"]:
        for value in df[grouping].dropna().unique():
            for period, mask in [
                ("2022", df["season"].astype(int) == 2022),
                ("other_test_seasons", df["season"].astype(int) != 2022),
            ]:
                group = df[mask & (df[grouping] == value)]
                if group.empty:
                    continue
                actual = group["market_residual"].to_numpy(float)
                baseline_abs = np.abs(
                    group["baseline_pred_market_residual"].to_numpy(float) - actual
                )
                along_abs = np.abs(
                    group["along_magnitude_pred_market_residual"].to_numpy(float) - actual
                )
                rows.append(
                    {
                        "grouping": grouping,
                        "value": str(value),
                        "period": period,
                        "games": len(group),
                        "baseline_mae": float(baseline_abs.mean()),
                        "along_mae": float(along_abs.mean()),
                        "along_mae_delta_vs_baseline": float(
                            (along_abs - baseline_abs).mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_summary(
    comparisons: pd.DataFrame,
    placebo_test: pd.DataFrame,
    status: pd.DataFrame,
    by_season: pd.DataFrame,
    season_context: pd.DataFrame,
    shift_2022: pd.DataFrame,
    regimes_2022: pd.DataFrame,
) -> None:
    status_text = (
        str(status.iloc[0]["confirmation_status"])
        if not status.empty
        else "NOT_EVALUATED"
    )
    lines = [
        "# Confirmatory Field-Geometry Research",
        "",
        "**Status: retrospective research only. No production, live-board, weekly-pick, orientation-shadow, or prospective-ledger effect.**",
        "",
        "## Purpose",
        "",
        "Confirm or falsify the previously isolated along-field-wind signal without expanding the production feature set or changing the parent evidence gate.",
        "",
        "## Predeclared primary confirmation",
        "",
        "The real-axis `along_magnitude` challenger must satisfy the same four-part retrospective evidence gate used previously and must also beat a 39-permutation venue-axis placebo test at randomization p <= 0.05. Field axes are permuted across venues while preserving the real axis distribution and orientation coverage.",
        "",
        "Alignment-only models are mechanism diagnostics: because baseline already contains wind speed, adding cos(field-relative angle) tests whether actual relative geometry adds information beyond wind magnitude alone.",
        "",
        f"Current confirmation status: **{status_text}**",
        "",
        "## Real-axis model comparisons",
        "",
        comparisons.to_markdown(index=False) if not comparisons.empty else "_No model comparisons produced._",
        "",
        "## Venue-axis permutation placebo",
        "",
        placebo_test.to_markdown(index=False) if not placebo_test.empty else "_No placebo result produced._",
        "",
        "## Results by test season",
        "",
        by_season.to_markdown(index=False) if not by_season.empty else "_No season results produced._",
        "",
        "## Season context diagnostics",
        "",
        season_context.to_markdown(index=False) if not season_context.empty else "_No season context produced._",
        "",
        "## 2022 standardized context shift",
        "",
        "These are descriptive standardized mean differences, not a post-hoc promotion test.",
        "",
        shift_2022.to_markdown(index=False) if not shift_2022.empty else "_No 2022 shift diagnostics produced._",
        "",
        "## 2022 geometry-regime diagnostics",
        "",
        regimes_2022.to_markdown(index=False) if not regimes_2022.empty else "_No 2022 regime diagnostics produced._",
        "",
        "## Guardrails",
        "",
        "- The primary hypothesis is fixed before this run: real along-field geometry must beat baseline and randomized venue axes.",
        "- Crosswind, vector, and alignment models are secondary mechanism checks and cannot substitute for failure of the primary confirmation.",
        "- The 2022 diagnostics are explanatory only. No season may be removed to manufacture significance.",
        "- Even retrospective confirmation would only justify considering a separately frozen prospective shadow challenger. It would not alter the live 2026 model.",
    ]
    ensure_dir(OUTPUT_DIR).joinpath("summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    venues = read_df("data/reference/stadium_locations.csv")
    raw = read_df("data/processed/modeling_dataset.csv")
    df = prepare_research_data(raw, venues)
    df = add_alignment_features(add_orientation_features(df))

    predictions, diagnostics = walk_forward_confirmation(df)
    comparisons = comparison_summary(predictions)
    by_season = model_by_season(predictions)

    orientation = orientation_table()
    placebos = placebo_walk_forward(df, orientation, predictions)
    placebo_test = placebo_significance(predictions, placebos)
    status = confirmation_status(comparisons, placebo_test)

    season_context = season_context_summary(predictions)
    shift_2022 = season_2022_standardized_shift(predictions)
    regimes_2022 = regime_2022_comparison(predictions)

    write_df(predictions, f"{OUTPUT_DIR}/walk_forward_predictions.csv")
    write_df(diagnostics, f"{OUTPUT_DIR}/walk_forward_diagnostics.csv")
    write_df(comparisons, f"{OUTPUT_DIR}/model_comparisons.csv")
    write_df(by_season, f"{OUTPUT_DIR}/model_by_season.csv")
    write_df(placebos, f"{OUTPUT_DIR}/axis_placebo_summary.csv")
    write_df(placebo_test, f"{OUTPUT_DIR}/axis_placebo_test.csv")
    write_df(status, f"{OUTPUT_DIR}/confirmation_status.csv")
    write_df(season_context, f"{OUTPUT_DIR}/season_context_summary.csv")
    write_df(shift_2022, f"{OUTPUT_DIR}/season_2022_standardized_shift.csv")
    write_df(regimes_2022, f"{OUTPUT_DIR}/season_2022_regime_comparison.csv")
    write_summary(
        comparisons,
        placebo_test,
        status,
        by_season,
        season_context,
        shift_2022,
        regimes_2022,
    )
    print("Wrote isolated confirmatory field-geometry research outputs")


if __name__ == "__main__":
    main()
