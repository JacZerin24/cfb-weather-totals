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

MODELING_PATH = "data/processed/modeling_dataset.csv"
GUST_PATH = "data/processed/wind_gust_kickoff.csv"
LOCATION_PATH = "data/reference/stadium_locations.csv"
OUTPUT_DIR = "outputs/wind_gust_value"

MIN_TRAIN_GAMES = 1_000
MIN_TEST_GAMES = 100

GUST_MAGNITUDE_THRESHOLDS = [20.0, 25.0, 30.0, 35.0]
GUST_SPREAD_THRESHOLDS = [5.0, 10.0, 15.0]

MODEL_FEATURES: dict[str, list[str]] = {
    "baseline": [],
    "ifs_sustained_control": ["ifs_wind_mph"],
    "gust_magnitude": ["ifs_wind_mph", "ifs_gust_mph"],
    "gust_spread": ["ifs_wind_mph", "ifs_gust_spread_mph"],
    "gust_core": ["ifs_wind_mph", "ifs_gust_mph", "ifs_gust_spread_mph"],
    "gust_interactions": [
        "ifs_wind_mph",
        "ifs_gust_mph",
        "ifs_gust_spread_mph",
        "ifs_gust_spread_x_wind",
        "ifs_gust_spread_x_market_total",
        "ifs_gust_x_temperature_anomaly",
    ],
}

PAIRWISE_COMPARISONS = [
    ("ifs_sustained_control", "baseline"),
    ("gust_magnitude", "ifs_sustained_control"),
    ("gust_spread", "ifs_sustained_control"),
    ("gust_core", "ifs_sustained_control"),
    ("gust_interactions", "gust_core"),
]

PRIMARY_GUST_COMPARISONS = {
    ("gust_magnitude", "ifs_sustained_control"),
    ("gust_spread", "ifs_sustained_control"),
    ("gust_core", "ifs_sustained_control"),
}


def _bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "false": False,
                "0": False,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def prepare_paired_data(
    raw: pd.DataFrame,
    gusts: pd.DataFrame,
    venues: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact common sample for all gust ladder variants."""
    base = prepare_research_data(raw, venues)
    base["game_id"] = pd.to_numeric(base.get("game_id"), errors="coerce")
    base["season"] = pd.to_numeric(base.get("season"), errors="coerce")

    scope = base[
        base["season"].between(2017, 2025, inclusive="both")
        & base["outdoor"].astype(bool)
        & base["fbs_vs_fbs"].astype(bool)
    ].copy()

    if scope["game_id"].duplicated().any():
        raise RuntimeError("Duplicate game_id rows in GENERAL-HGB gust research scope.")

    external = gusts.copy()
    external["game_id"] = pd.to_numeric(external.get("game_id"), errors="coerce")
    external = external[
        external.get("source_stack", pd.Series("", index=external.index))
        .astype(str)
        .eq("ecmwf_ifs")
        & external.get("fetch_status", pd.Series("", index=external.index))
        .astype(str)
        .eq("ok")
    ].copy()

    if "same_model_components" in external.columns:
        same_model = _bool_series(external["same_model_components"])
        external = external[same_model].copy()

    if external["game_id"].duplicated().any():
        dupes = int(external["game_id"].duplicated().sum())
        raise RuntimeError(f"Duplicate IFS gust rows detected: {dupes} duplicate game_id values.")

    required_external = ["wind_mph", "gust_mph", "gust_spread_mph"]
    missing_cols = [c for c in required_external if c not in external.columns]
    if missing_cols:
        raise RuntimeError(
            "Full gust artifact is missing required fields: " + ", ".join(missing_cols)
        )

    external = external.dropna(subset=required_external).copy()
    rename = {
        "wind_mph": "ifs_wind_mph",
        "wind_direction_degrees": "ifs_wind_direction_degrees",
        "gust_mph": "ifs_gust_mph",
        "gust_spread_mph": "ifs_gust_spread_mph",
        "gust_factor": "ifs_gust_factor",
        "wind_valid_time_utc": "ifs_wind_valid_time_utc",
        "gust_valid_time_utc": "ifs_gust_valid_time_utc",
        "wind_age_minutes": "ifs_wind_age_minutes",
        "gust_age_minutes": "ifs_gust_age_minutes",
        "wind_model": "ifs_wind_model",
        "gust_model": "ifs_gust_model",
        "weather_provider": "ifs_weather_provider",
    }
    keep = ["game_id"] + [c for c in rename if c in external.columns]
    external = external[keep].rename(columns=rename)

    paired = scope.merge(external, on="game_id", how="inner", validate="one_to_one")
    paired = paired.sort_values(["season", "game_id"]).reset_index(drop=True)

    coverage_rows: list[dict[str, Any]] = []
    for label, group in [("overall", scope)] + [
        (str(int(season)), group)
        for season, group in scope.groupby("season", observed=True)
    ]:
        paired_games = (
            len(paired)
            if label == "overall"
            else int((paired["season"] == int(label)).sum())
        )
        coverage_rows.append(
            {
                "scope": label,
                "general_hgb_scope_games": len(group),
                "paired_complete_ifs_games": paired_games,
                "paired_pct": paired_games / len(group) if len(group) else np.nan,
                "unique_scope_venues": int(group["venue_id"].nunique(dropna=True)),
                "unique_paired_venues": (
                    int(paired["venue_id"].nunique(dropna=True))
                    if label == "overall"
                    else int(
                        paired.loc[
                            paired["season"] == int(label), "venue_id"
                        ].nunique(dropna=True)
                    )
                ),
            }
        )
    return paired, pd.DataFrame(coverage_rows)


def add_gust_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    wind = pd.to_numeric(out["ifs_wind_mph"], errors="coerce")
    gust = pd.to_numeric(out["ifs_gust_mph"], errors="coerce")
    spread = pd.to_numeric(out["ifs_gust_spread_mph"], errors="coerce")
    total_scaled = (pd.to_numeric(out["closing_total"], errors="coerce") - 56.0) / 10.0
    temp_anomaly_scaled = (
        pd.to_numeric(out.get("temperature_anomaly_f"), errors="coerce") / 10.0
    )
    out["ifs_gust_spread_x_wind"] = spread * wind
    out["ifs_gust_spread_x_market_total"] = spread * total_scaled
    out["ifs_gust_x_temperature_anomaly"] = gust * temp_anomaly_scaled
    return out


def walk_forward_predictions(
    df: pd.DataFrame,
    min_train_games: int = MIN_TRAIN_GAMES,
    min_test_games: int = MIN_TEST_GAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every ladder variant on identical training rows and score identical test rows."""
    base_nums, cats = feature_lists(df)
    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for season in sorted(df["season"].dropna().astype(int).unique()):
        train = df[df["season"] < season].copy()
        test = df[df["season"] == season].copy()
        if len(train) < min_train_games or len(test) < min_test_games:
            continue

        reference = build_context_reference(train)
        train = add_gust_interactions(apply_context_features(train, reference))
        test = add_gust_interactions(apply_context_features(test, reference))

        train = prep_features(train, cats)
        test = prep_features(test, cats)

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
                "wind_mph",
                "temperature_f",
                "temperature_anomaly_f",
                "ifs_wind_mph",
                "ifs_wind_direction_degrees",
                "ifs_gust_mph",
                "ifs_gust_spread_mph",
                "ifs_gust_factor",
                "ifs_wind_valid_time_utc",
                "ifs_gust_valid_time_utc",
            ]
            if c in test.columns
        ]
        fold = test[keep].copy()

        for model_name, additions in MODEL_FEATURES.items():
            nums = list(dict.fromkeys(base_nums + additions))
            missing = [c for c in nums + cats if c not in train.columns or c not in test.columns]
            if missing:
                raise RuntimeError(
                    f"{model_name} missing required modeling fields: {', '.join(sorted(set(missing)))}"
                )

            model = reg_models(nums, cats)["hist_gradient_boosting"]
            model.fit(train[nums + cats], train["market_residual"])
            pred = model.predict(test[nums + cats])
            fold[f"{model_name}_pred_market_residual"] = pred
            error = pred - test["market_residual"].to_numpy(float)
            diagnostics.append(
                {
                    "test_season": season,
                    "model": model_name,
                    "train_games": len(train),
                    "test_games": len(test),
                    "paired_train_games": len(train),
                    "paired_test_games": len(test),
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
    seed: int = BOOTSTRAP_SEED + 900,
) -> tuple[float, float]:
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
    if not season_deltas:
        return np.nan, np.nan
    return bootstrap_mean_ci(
        np.asarray(season_deltas, dtype=float), reps=reps, seed=seed
    )


def evidence_gate(
    mean_delta: float,
    game_ci_high: float,
    season_ci_high: float,
    improved_seasons: int,
    test_seasons: int,
) -> tuple[str, int]:
    required = max(1, math.ceil(test_seasons * 0.70))
    supported = (
        mean_delta < 0
        and game_ci_high < 0
        and season_ci_high < 0
        and improved_seasons >= required
    )
    return (
        "SUPPORTED_RETROSPECTIVELY" if supported else "NOT_PROVEN",
        required,
    )


def comparison_stats(
    predictions: pd.DataFrame,
    challenger: str,
    reference: str,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    order: int = 0,
) -> dict[str, Any]:
    actual = predictions["market_residual"].to_numpy(float)
    challenger_abs = np.abs(
        predictions[f"{challenger}_pred_market_residual"].to_numpy(float) - actual
    )
    reference_abs = np.abs(
        predictions[f"{reference}_pred_market_residual"].to_numpy(float) - actual
    )
    delta = challenger_abs - reference_abs
    game_low, game_high = bootstrap_mean_ci(
        delta,
        reps=bootstrap_reps,
        seed=BOOTSTRAP_SEED + 1_000 + order,
    )
    season_low, season_high = season_cluster_ci(
        predictions,
        challenger,
        reference,
        reps=bootstrap_reps,
        seed=BOOTSTRAP_SEED + 1_100 + order,
    )
    improved = 0
    seasons = int(predictions["season"].nunique())
    for _, group in predictions.groupby("season", observed=True):
        y = group["market_residual"].to_numpy(float)
        challenger_mae = np.abs(
            group[f"{challenger}_pred_market_residual"].to_numpy(float) - y
        ).mean()
        reference_mae = np.abs(
            group[f"{reference}_pred_market_residual"].to_numpy(float) - y
        ).mean()
        improved += int(challenger_mae < reference_mae)

    status, required = evidence_gate(
        float(delta.mean()), game_high, season_high, improved, seasons
    )
    return {
        "challenger": challenger,
        "reference": reference,
        "paired_games": len(predictions),
        "test_seasons": seasons,
        "mae_delta_challenger_minus_reference": float(delta.mean()),
        "game_bootstrap_ci_low": game_low,
        "game_bootstrap_ci_high": game_high,
        "season_cluster_ci_low": season_low,
        "season_cluster_ci_high": season_high,
        "test_seasons_improved": improved,
        "seasons_required_for_support": required,
        "incremental_evidence_status": status,
        "primary_gust_comparison": (challenger, reference) in PRIMARY_GUST_COMPARISONS,
    }


def incremental_comparisons(
    predictions: pd.DataFrame,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            comparison_stats(
                predictions,
                challenger,
                reference,
                bootstrap_reps=bootstrap_reps,
                order=order,
            )
            for order, (challenger, reference) in enumerate(PAIRWISE_COMPARISONS)
        ]
    )


def model_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    actual = predictions["market_residual"].to_numpy(float)
    baseline_abs = np.abs(
        predictions["baseline_pred_market_residual"].to_numpy(float) - actual
    )
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_FEATURES:
        pred = predictions[f"{model_name}_pred_market_residual"].to_numpy(float)
        error = pred - actual
        abs_error = np.abs(error)
        rows.append(
            {
                "model": model_name,
                "paired_games": len(predictions),
                "test_seasons": int(predictions["season"].nunique()),
                "mae": float(abs_error.mean()),
                "mae_delta_vs_baseline": float((abs_error - baseline_abs).mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "signed_projection_bias": float(error.mean()),
            }
        )
    return pd.DataFrame(rows)


def model_by_season(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for season, group in predictions.groupby("season", observed=True):
        actual = group["market_residual"].to_numpy(float)
        base_abs = np.abs(
            group["baseline_pred_market_residual"].to_numpy(float) - actual
        )
        control_abs = np.abs(
            group["ifs_sustained_control_pred_market_residual"].to_numpy(float)
            - actual
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
                    "mae_delta_vs_baseline": float((abs_error - base_abs).mean()),
                    "mae_delta_vs_ifs_sustained_control": float(
                        (abs_error - control_abs).mean()
                    ),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "signed_projection_bias": float(error.mean()),
                }
            )
    return pd.DataFrame(rows)


def threshold_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    specs = [
        ("gust_magnitude", "ifs_gust_mph", threshold)
        for threshold in GUST_MAGNITUDE_THRESHOLDS
    ] + [
        ("gust_spread", "ifs_gust_spread_mph", threshold)
        for threshold in GUST_SPREAD_THRESHOLDS
    ]
    for family, field, threshold in specs:
        group = predictions[
            pd.to_numeric(predictions[field], errors="coerce") >= threshold
        ].copy()
        if group.empty:
            rows.append(
                {
                    "threshold_family": family,
                    "field": field,
                    "threshold_mph": threshold,
                    "games": 0,
                }
            )
            continue
        actual = group["market_residual"].to_numpy(float)
        baseline_error = np.abs(
            group["baseline_pred_market_residual"].to_numpy(float) - actual
        )
        control_error = np.abs(
            group["ifs_sustained_control_pred_market_residual"].to_numpy(float)
            - actual
        )
        core_error = np.abs(
            group["gust_core_pred_market_residual"].to_numpy(float) - actual
        )
        under = group["actual_total_points"] < group["closing_total"]
        push = group["actual_total_points"] == group["closing_total"]
        graded = (~push).sum()
        rows.append(
            {
                "threshold_family": family,
                "field": field,
                "threshold_mph": threshold,
                "games": len(group),
                "test_seasons": int(group["season"].nunique()),
                "avg_market_residual": float(group["market_residual"].mean()),
                "under_rate_ex_push": (
                    float(under[~push].mean()) if int(graded) else np.nan
                ),
                "baseline_mae": float(baseline_error.mean()),
                "ifs_sustained_control_mae": float(control_error.mean()),
                "gust_core_mae": float(core_error.mean()),
                "gust_core_mae_delta_vs_control": float(
                    (core_error - control_error).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    out = ensure_dir(OUTPUT_DIR) / "summary.md"
    primary = comparisons[comparisons["primary_gust_comparison"].astype(bool)].copy()
    lines = [
        "# Wind Gust Incremental-Value Research",
        "",
        "**Retrospective research only. No production, live-board, weekly-pick, shadow, or prospective-ledger effect.**",
        "",
        "## Design",
        "",
        "- Population: 2017-2025 outdoor FBS-vs-FBS games with a closing total/final score and complete same-model ECMWF IFS kickoff wind+gust data.",
        "- Target: market residual.",
        "- Model: the repository's existing GENERAL HistGradientBoostingRegressor configuration.",
        "- Evaluation: chronological walk-forward; every ladder variant uses identical train/test game rows.",
        "- External-source control: IFS sustained wind is added before any gust feature receives credit.",
        "- Temperature-anomaly interaction context is built from training seasons only.",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(index=False) if not coverage.empty else "_No coverage output._",
        "",
        "## Model summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No model output._",
        "",
        "## Incremental comparisons",
        "",
        comparisons.to_markdown(index=False)
        if not comparisons.empty
        else "_No comparison output._",
        "",
        "## Primary gust evidence",
        "",
        primary.to_markdown(index=False) if not primary.empty else "_No primary gust comparisons._",
        "",
        "## Interpretation",
        "",
    ]
    if primary.empty:
        lines.append("- No primary gust evidence could be evaluated.")
    else:
        supported = primary[
            primary["incremental_evidence_status"].eq("SUPPORTED_RETROSPECTIVELY")
        ]
        if supported.empty:
            lines.append(
                "- **Gust incremental value is NOT_PROVEN** by the preregistered MAE evidence gate."
            )
        else:
            names = ", ".join(supported["challenger"].astype(str))
            lines.append(
                f"- **Retrospective gust value is supported for:** {names}."
            )
            lines.append(
                "- This remains reanalysis/historical evidence, not evidence that a bettor could have obtained the same value pregame."
            )

    lines.extend(
        [
            "- Threshold diagnostics are descriptive and cannot override the paired MAE gate.",
            "- Betting hit rate/ROI is intentionally not a promotion criterion.",
            "- A positive retrospective result must still advance to fixed-lead archived forecasts before any prospective shadow is considered.",
            "",
            "## Fold diagnostics",
            "",
            diagnostics.to_markdown(index=False)
            if not diagnostics.empty
            else "_No fold diagnostics._",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def run(
    bootstrap_reps: int = BOOTSTRAP_REPS,
    min_train_games: int = MIN_TRAIN_GAMES,
    min_test_games: int = MIN_TEST_GAMES,
) -> dict[str, pd.DataFrame]:
    raw = read_df(MODELING_PATH)
    gusts = read_df(GUST_PATH)
    venues = read_df(LOCATION_PATH)
    paired, coverage = prepare_paired_data(raw, gusts, venues)

    if paired.empty:
        raise RuntimeError("No complete paired IFS gust rows are available for modeling.")

    predictions, diagnostics = walk_forward_predictions(
        paired,
        min_train_games=min_train_games,
        min_test_games=min_test_games,
    )
    if predictions.empty:
        raise RuntimeError(
            "No walk-forward folds met the minimum train/test game requirements."
        )

    summary = model_summary(predictions)
    comparisons = incremental_comparisons(
        predictions, bootstrap_reps=bootstrap_reps
    )
    by_season = model_by_season(predictions)
    thresholds = threshold_diagnostics(predictions)

    ensure_dir(OUTPUT_DIR)
    outputs = {
        "coverage": coverage,
        "walk_forward_predictions": predictions,
        "walk_forward_diagnostics": diagnostics,
        "model_summary": summary,
        "incremental_comparisons": comparisons,
        "model_by_season": by_season,
        "threshold_diagnostics": thresholds,
    }
    for name, frame in outputs.items():
        write_df(frame, f"{OUTPUT_DIR}/{name}.csv")
    write_summary(coverage, summary, comparisons, diagnostics)
    print(
        f"Wrote wind-gust value research: paired_rows={len(paired):,}; "
        f"scored_rows={len(predictions):,}; test_seasons={predictions['season'].nunique():,}"
    )
    return outputs


def main() -> None:
    run()


if __name__ == "__main__":
    main()
