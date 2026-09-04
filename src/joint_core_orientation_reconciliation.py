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
from .field_geometry_confirmation import (
    PLACEBO_COUNT,
    PLACEBO_SEED,
    placebo_orientation_maps,
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

OUTPUT_DIR = "outputs/joint_core_orientation_reconciliation"

REAL_MODEL_FEATURES: dict[str, list[str]] = {
    "climate_full": [*CLIMATE_FEATURES],
    "joint_core_no_along": ["crosswind_mph", *CLIMATE_FEATURES],
    "joint_core_real": ["crosswind_mph", "alongwind_mph", *CLIMATE_FEATURES],
}

PRIMARY_CHALLENGER = "joint_core_real"
PRIMARY_REFERENCE = "joint_core_no_along"


def add_placebo_alongwind(
    df: pd.DataFrame,
    axis_map: pd.DataFrame,
    output_col: str = "placebo_alongwind_mph",
) -> pd.DataFrame:
    """Replace only the venue-axis identity used for along-field wind.

    Real crosswind and all climate/context features remain untouched. This is
    intentional: the primary test isolates the exact along-field component
    that was supported in the earlier joint-core ablation.
    """
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
    challenger: str = PRIMARY_CHALLENGER,
    reference: str = PRIMARY_REFERENCE,
    seed_offset: int = 0,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    delta = _paired_delta(predictions, challenger, reference)
    seasons = int(predictions["season"].nunique())
    required = max(1, math.ceil(seasons * 0.70))
    game_low, game_high = bootstrap_mean_ci(
        delta,
        reps=bootstrap_reps,
        seed=BOOTSTRAP_SEED + 7_000 + seed_offset,
    )
    season_low, season_high = season_cluster_ci(
        predictions,
        challenger,
        reference,
        reps=bootstrap_reps,
        seed=BOOTSTRAP_SEED + 8_000 + seed_offset,
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
        "mae_penalty_when_real_along_removed": float(-delta.mean()),
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


def walk_forward_real_models(
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
        train = apply_context_features(train, reference)
        test = apply_context_features(test, reference)
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

        for model_name, additions in REAL_MODEL_FEATURES.items():
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


def placebo_walk_forward(
    df: pd.DataFrame,
    orientation: pd.DataFrame,
    real_predictions: pd.DataFrame,
    placebo_count: int = PLACEBO_COUNT,
    seed: int = PLACEBO_SEED,
    min_train_games: int = 1_000,
    min_test_games: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the conditional along-wind placebo distribution.

    Each model contains the same baseline features, climate features, and REAL
    crosswind as joint_core_no_along. Only the added along-wind feature uses a
    permuted venue-axis assignment.
    """
    if real_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    base_nums, cats = feature_lists(df)
    maps = placebo_orientation_maps(orientation, count=placebo_count, seed=seed)
    summary_rows: list[dict[str, Any]] = []
    season_rows: list[dict[str, Any]] = []

    reference_lookup = real_predictions.set_index(["season", "game_id"])[
        f"{PRIMARY_REFERENCE}_pred_market_residual"
    ]

    for placebo_id, axis_map in enumerate(maps):
        fold_predictions: list[pd.DataFrame] = []
        for season in sorted(df["season"].dropna().astype(int).unique()):
            train = df[df["season"] < season].copy()
            test = df[df["season"] == season].copy()
            if len(train) < min_train_games or len(test) < min_test_games:
                continue

            reference = build_context_reference(train)
            train = apply_context_features(train, reference)
            test = apply_context_features(test, reference)
            train = add_placebo_alongwind(train, axis_map)
            test = add_placebo_alongwind(test, axis_map)
            scored = test[test["joint_ready"]].copy()
            if scored.empty:
                continue

            train = prep_features(train, cats)
            test = prep_features(test, cats)
            scored = test.loc[scored.index].copy()

            additions = ["crosswind_mph", "placebo_alongwind_mph", *CLIMATE_FEATURES]
            nums = list(dict.fromkeys(base_nums + additions))
            model = reg_models(nums, cats)["hist_gradient_boosting"]
            model.fit(train[nums + cats], train["market_residual"])
            prediction = model.predict(scored[nums + cats])

            fold = scored[["season", "game_id", "market_residual"]].copy()
            fold["placebo_pred_market_residual"] = prediction
            keys = pd.MultiIndex.from_frame(fold[["season", "game_id"]])
            fold["reference_pred_market_residual"] = reference_lookup.reindex(
                keys
            ).to_numpy()
            if fold["reference_pred_market_residual"].isna().any():
                raise RuntimeError(
                    "Conditional placebo cohort did not align with the real no-along cohort"
                )
            fold_predictions.append(fold)

        if not fold_predictions:
            continue

        combined = pd.concat(fold_predictions, ignore_index=True)
        actual = combined["market_residual"].to_numpy(float)
        placebo_abs = np.abs(
            combined["placebo_pred_market_residual"].to_numpy(float) - actual
        )
        reference_abs = np.abs(
            combined["reference_pred_market_residual"].to_numpy(float) - actual
        )
        delta = placebo_abs - reference_abs

        improved = 0
        for season, group in combined.groupby("season", observed=True):
            y = group["market_residual"].to_numpy(float)
            p = np.abs(group["placebo_pred_market_residual"].to_numpy(float) - y)
            r = np.abs(group["reference_pred_market_residual"].to_numpy(float) - y)
            season_delta = float(np.mean(p - r))
            improved += int(season_delta < 0)
            season_rows.append(
                {
                    "placebo_id": placebo_id,
                    "test_season": int(season),
                    "paired_games": len(group),
                    "mean_mae_delta_placebo_vs_no_along": season_delta,
                }
            )

        summary_rows.append(
            {
                "placebo_id": placebo_id,
                "paired_games": len(combined),
                "test_seasons": int(combined["season"].nunique()),
                "mean_mae_delta_placebo_vs_no_along": float(delta.mean()),
                "mae_penalty_when_placebo_along_removed": float(-delta.mean()),
                "test_seasons_improved_vs_no_along": improved,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(season_rows)


def placebo_significance(
    real_predictions: pd.DataFrame,
    placebo_summary: pd.DataFrame,
) -> pd.DataFrame:
    if real_predictions.empty or placebo_summary.empty:
        return pd.DataFrame()

    real_delta = float(
        _paired_delta(real_predictions, PRIMARY_CHALLENGER, PRIMARY_REFERENCE).mean()
    )
    placebo = placebo_summary[
        "mean_mae_delta_placebo_vs_no_along"
    ].to_numpy(float)
    as_good_or_better = int(np.sum(placebo <= real_delta))
    randomization_p = (as_good_or_better + 1) / (len(placebo) + 1)
    q05 = float(np.quantile(placebo, 0.05))

    distinguished = randomization_p <= 0.05 and real_delta < q05
    return pd.DataFrame(
        [
            {
                "real_joint_core_delta_vs_no_along": real_delta,
                "real_mae_penalty_when_along_removed": -real_delta,
                "placebo_count": len(placebo),
                "placebo_mean_delta_vs_no_along": float(np.mean(placebo)),
                "placebo_median_delta_vs_no_along": float(np.median(placebo)),
                "placebo_05_quantile_delta_vs_no_along": q05,
                "placebos_as_good_or_better_than_real": as_good_or_better,
                "randomization_p_value": randomization_p,
                "real_better_than_95pct_placebos": bool(real_delta < q05),
                "conditional_geometry_status": (
                    "REAL_ALONG_DISTINGUISHED_WITHIN_JOINT_CORE"
                    if distinguished
                    else "RANDOMIZED_ALONG_PERFORMS_SIMILARLY"
                ),
            }
        ]
    )


def season_reconciliation(
    real_predictions: pd.DataFrame,
    placebo_by_season: pd.DataFrame,
) -> pd.DataFrame:
    if real_predictions.empty or placebo_by_season.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for season, group in real_predictions.groupby("season", observed=True):
        real_delta = float(
            _paired_delta(group, PRIMARY_CHALLENGER, PRIMARY_REFERENCE).mean()
        )
        placebo = placebo_by_season[
            placebo_by_season["test_season"].astype(int) == int(season)
        ]["mean_mae_delta_placebo_vs_no_along"].to_numpy(float)
        if len(placebo) == 0:
            continue
        as_good = int(np.sum(placebo <= real_delta))
        rows.append(
            {
                "test_season": int(season),
                "paired_games": len(group),
                "real_delta_vs_no_along": real_delta,
                "placebo_mean_delta": float(np.mean(placebo)),
                "placebo_median_delta": float(np.median(placebo)),
                "placebo_05_quantile_delta": float(np.quantile(placebo, 0.05)),
                "placebos_as_good_or_better_than_real": as_good,
                "placebo_count": len(placebo),
                "real_percentile_among_placebos_lower_is_better": float(
                    (as_good + 1) / (len(placebo) + 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def reconciliation_status(
    real_component: pd.DataFrame,
    placebo_test: pd.DataFrame,
) -> pd.DataFrame:
    if real_component.empty or placebo_test.empty:
        return pd.DataFrame([{"reconciliation_status": "NOT_EVALUATED"}])

    component = real_component.iloc[0]
    placebo = placebo_test.iloc[0]
    component_supported = component["evidence_status"] == "SUPPORTED_RETROSPECTIVELY"
    geometry_distinguished = (
        placebo["conditional_geometry_status"]
        == "REAL_ALONG_DISTINGUISHED_WITHIN_JOINT_CORE"
    )

    if component_supported and geometry_distinguished:
        status = "CONDITIONAL_REAL_GEOMETRY_RECONCILED"
        interpretation = (
            "Real along-field orientation is uniquely informative inside the fixed joint-core context."
        )
    elif component_supported and not geometry_distinguished:
        status = "ORIENTATION_PROXY_NOT_DISTINGUISHED"
        interpretation = (
            "Along-field magnitude helps conditionally, but the real stadium-axis identity is not uniquely informative versus randomized axes."
        )
    else:
        status = "CONDITIONAL_ALONG_SIGNAL_NOT_REPRODUCED"
        interpretation = (
            "The previously isolated along-field contribution did not reproduce under the fixed reconciliation gate."
        )

    return pd.DataFrame(
        [
            {
                "reconciliation_status": status,
                "real_component_evidence_status": component["evidence_status"],
                "conditional_geometry_status": placebo["conditional_geometry_status"],
                "randomization_p_value": placebo["randomization_p_value"],
                "interpretation": interpretation,
            }
        ]
    )


def write_summary(
    real_component: pd.DataFrame,
    placebo_test: pd.DataFrame,
    status: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    by_season: pd.DataFrame,
) -> None:
    status_text = (
        str(status.iloc[0]["reconciliation_status"])
        if not status.empty
        else "NOT_EVALUATED"
    )
    lines = [
        "# Final Joint-Core Orientation Reconciliation",
        "",
        "**Status: retrospective research only. No production, live-board, weekly-pick, shadow, or prospective-ledger effect.**",
        "",
        "## Fixed question",
        "",
        "Does the real venue-axis identity explain the previously supported along-field-wind contribution inside `joint_core`, or can randomized venue axes provide the same conditional benefit?",
        "",
        "## Predeclared primary test",
        "",
        "The reference model is `joint_core_no_along`: the frozen GENERAL baseline plus all climate-context features plus the REAL crosswind feature. The real challenger adds REAL `alongwind_mph`. Each of 39 placebo challengers is otherwise identical but replaces only `alongwind_mph` with an along-wind magnitude calculated from a deranged venue-to-axis assignment.",
        "",
        "This deliberately holds raw wind, climate context, real crosswind, model hyperparameters, training folds, and the scored game cohort fixed. It isolates the exact along-field component that was supported in the earlier ablation.",
        "",
        "Reconciliation as real conditional geometry requires BOTH: (1) the real challenger passes the same paired-MAE/game-bootstrap/season-cluster/70%-season evidence gate, and (2) the real venue axes beat the 39-placebo distribution at randomization p <= 0.05 and below its 5th percentile.",
        "",
        f"Current reconciliation status: **{status_text}**",
        "",
        "## Real conditional component test",
        "",
        real_component.to_markdown(index=False)
        if not real_component.empty
        else "_No real-component result produced._",
        "",
        "## Conditional venue-axis placebo test",
        "",
        placebo_test.to_markdown(index=False)
        if not placebo_test.empty
        else "_No placebo significance result produced._",
        "",
        "## Real vs placebo by season",
        "",
        by_season.to_markdown(index=False)
        if not by_season.empty
        else "_No season reconciliation produced._",
        "",
        "## Placebo distribution",
        "",
        placebo_summary.to_markdown(index=False)
        if not placebo_summary.empty
        else "_No placebo distribution produced._",
        "",
        "## Decision rule after this study",
        "",
        "- `CONDITIONAL_REAL_GEOMETRY_RECONCILED`: real venue axes are uniquely informative inside joint_core. This still does not change production; it could only justify a separately frozen prospective shadow.",
        "- `ORIENTATION_PROXY_NOT_DISTINGUISHED`: the real axes are not special even though an along-wind-like feature may help conditionally. Retire field orientation from further retrospective tuning and do not create an orientation shadow.",
        "- `CONDITIONAL_ALONG_SIGNAL_NOT_REPRODUCED`: the prior along-field ablation itself did not reproduce. Retire that signal.",
        "",
        "## Guardrails",
        "",
        "- No new features, thresholds, training windows, or post-hoc season exclusions are tested here.",
        "- The 39 permutations and seed are fixed before observing this run's outcome.",
        "- The prior failed standalone geometry confirmation is not overwritten by this test.",
        "- Production remains frozen regardless of the retrospective result.",
    ]
    ensure_dir(OUTPUT_DIR).joinpath("summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    venues = read_df("data/reference/stadium_locations.csv")
    raw = read_df("data/processed/modeling_dataset.csv")
    df = prepare_research_data(raw, venues)
    df = add_orientation_features(df)

    real_predictions, diagnostics = walk_forward_real_models(df)
    real_component = pd.DataFrame([comparison_row(real_predictions)])

    orientation = orientation_table()
    placebo_summary, placebo_by_season = placebo_walk_forward(
        df,
        orientation,
        real_predictions,
    )
    placebo_test = placebo_significance(real_predictions, placebo_summary)
    by_season = season_reconciliation(real_predictions, placebo_by_season)
    status = reconciliation_status(real_component, placebo_test)

    write_df(real_predictions, f"{OUTPUT_DIR}/walk_forward_predictions.csv")
    write_df(diagnostics, f"{OUTPUT_DIR}/walk_forward_diagnostics.csv")
    write_df(real_component, f"{OUTPUT_DIR}/real_conditional_component_test.csv")
    write_df(placebo_summary, f"{OUTPUT_DIR}/conditional_placebo_summary.csv")
    write_df(placebo_by_season, f"{OUTPUT_DIR}/conditional_placebo_by_season.csv")
    write_df(placebo_test, f"{OUTPUT_DIR}/conditional_placebo_test.csv")
    write_df(by_season, f"{OUTPUT_DIR}/season_reconciliation.csv")
    write_df(status, f"{OUTPUT_DIR}/reconciliation_status.csv")
    write_summary(real_component, placebo_test, status, placebo_summary, by_season)
    print("Wrote isolated final joint-core orientation reconciliation outputs")


if __name__ == "__main__":
    main()
