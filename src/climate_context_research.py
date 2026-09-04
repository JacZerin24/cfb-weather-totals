from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .cfbd_client import CFBDClient
from .deep_research import prep
from .model_bakeoff import feature_lists, prep_features, reg_models
from .utils import ensure_dir, read_df, write_df

LOCATION_PATH = "data/reference/stadium_locations.csv"
OUTPUT_DIR = "outputs/climate_context"
LATITUDE_EDGES = [-np.inf, 30.0, 35.0, 40.0, np.inf]
LATITUDE_LABELS = ["<30N", "30-35N", "35-40N", "40N+"]
MIN_INTERACTION_GAMES = 25
MIN_VENUE_MONTH_GAMES = 8
MIN_BAND_MONTH_GAMES = 25
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260904


def normalize_venues(records: list[dict[str, Any]]) -> pd.DataFrame:
    venues = pd.json_normalize(records)
    if venues.empty:
        return pd.DataFrame(
            columns=[
                "venue_id",
                "venue_name",
                "venue_latitude",
                "venue_longitude",
                "venue_city",
                "venue_state",
                "venue_timezone",
            ]
        )
    venues = venues.rename(
        columns={
            "id": "venue_id",
            "name": "venue_name",
            "latitude": "venue_latitude",
            "longitude": "venue_longitude",
            "city": "venue_city",
            "state": "venue_state",
            "timezone": "venue_timezone",
        }
    )
    keep = [
        c
        for c in [
            "venue_id",
            "venue_name",
            "venue_latitude",
            "venue_longitude",
            "venue_city",
            "venue_state",
            "venue_timezone",
        ]
        if c in venues.columns
    ]
    venues = venues[keep].copy()
    if "venue_id" not in venues.columns:
        return pd.DataFrame(
            columns=[
                "venue_id",
                "venue_name",
                "venue_latitude",
                "venue_longitude",
                "venue_city",
                "venue_state",
                "venue_timezone",
            ]
        )
    for col in ["venue_id", "venue_latitude", "venue_longitude"]:
        if col not in venues.columns:
            venues[col] = np.nan
        venues[col] = pd.to_numeric(venues[col], errors="coerce")
    return (
        venues.dropna(subset=["venue_id"])
        .drop_duplicates("venue_id")
        .reset_index(drop=True)
    )


def fetch_venue_locations(client: CFBDClient | None = None) -> pd.DataFrame:
    client = client or CFBDClient()
    venues = normalize_venues(client.get("/venues"))
    if venues.empty:
        raise RuntimeError(
            "CFBD returned no venue records for climate-context research."
        )
    write_df(venues, LOCATION_PATH)
    return venues


def add_latitude_band(latitude: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(latitude, errors="coerce"),
        bins=LATITUDE_EDGES,
        labels=LATITUDE_LABELS,
        right=False,
    )


def prepare_research_data(raw: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    df = prep(raw)
    df["venue_id"] = pd.to_numeric(df.get("venue_id"), errors="coerce")

    location_cols = [
        c
        for c in [
            "venue_id",
            "venue_name",
            "venue_latitude",
            "venue_longitude",
            "venue_city",
            "venue_state",
            "venue_timezone",
        ]
        if c in venues.columns
    ]
    locations = venues[location_cols].drop_duplicates("venue_id").copy()
    stale = [c for c in locations.columns if c != "venue_id" and c in df.columns]
    df = df.drop(columns=stale, errors="ignore").merge(
        locations, on="venue_id", how="left"
    )

    df["venue_latitude"] = pd.to_numeric(df.get("venue_latitude"), errors="coerce")
    df["venue_longitude"] = pd.to_numeric(df.get("venue_longitude"), errors="coerce")
    df["latitude_band"] = add_latitude_band(df["venue_latitude"])
    kickoff = pd.to_datetime(df.get("start_date"), utc=True, errors="coerce")
    df["calendar_month"] = kickoff.dt.month.astype("Int64")

    df["temp_bin"] = pd.cut(
        df["temperature_f"],
        bins=[-np.inf, 35, 50, 70, 85, np.inf],
        labels=["<=35", "35-50", "50-70", "70-85", "85+"],
        right=True,
    )
    df["wind_bin"] = pd.cut(
        df["wind_mph"],
        bins=[-np.inf, 5, 10, 15, 20, np.inf],
        labels=["0-5", "5-10", "10-15", "15-20", "20+"],
        right=True,
    )
    df["coordinate_matched"] = (
        df[["venue_latitude", "venue_longitude"]].notna().all(axis=1)
    )
    df["context_ready"] = (
        df["outdoor"]
        & df["coordinate_matched"]
        & df["calendar_month"].notna()
        & df["temperature_f"].notna()
        & df["wind_mph"].notna()
    )
    return df


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    def row(label: str, group: pd.DataFrame) -> dict[str, Any]:
        games = len(group)
        ready = int(group["context_ready"].sum())
        fbs = (
            group["fbs_vs_fbs"].astype(bool)
            if "fbs_vs_fbs" in group.columns
            else pd.Series(False, index=group.index)
        )
        fbs_games = int(fbs.sum())
        fbs_ready = int((group["context_ready"] & fbs).sum())
        return {
            "scope": label,
            "games": games,
            "unique_venues": int(group["venue_id"].nunique(dropna=True)),
            "with_venue_id": int(group["venue_id"].notna().sum()),
            "with_coordinates": int(group["coordinate_matched"].sum()),
            "outdoor_games": int(group["outdoor"].sum()),
            "with_temperature": int(group["temperature_f"].notna().sum()),
            "with_wind": int(group["wind_mph"].notna().sum()),
            "context_ready_games": ready,
            "context_ready_pct": ready / games if games else np.nan,
            "fbs_vs_fbs_games": fbs_games,
            "fbs_context_ready_games": fbs_ready,
            "fbs_context_ready_pct": fbs_ready / fbs_games if fbs_games else np.nan,
        }

    rows = [row("overall", df)]
    for season, group in df.groupby("season", observed=True):
        rows.append(row(str(int(season)), group))
    return pd.DataFrame(rows)


def latitude_weather_interactions(
    df: pd.DataFrame,
    min_games: int = MIN_INTERACTION_GAMES,
) -> pd.DataFrame:
    eligible_mask = df["context_ready"].copy()
    if "fbs_vs_fbs" in df.columns:
        eligible_mask &= df["fbs_vs_fbs"].astype(bool)
    eligible = df[eligible_mask].copy()
    eligible["went_under_context"] = (
        eligible["actual_total_points"] < eligible["closing_total"]
    )
    rows: list[dict[str, Any]] = []
    for grouping, bin_col in [("temperature", "temp_bin"), ("wind", "wind_bin")]:
        source = eligible.dropna(subset=["latitude_band", bin_col])
        for (weather_bin, latitude_band), group in source.groupby(
            [bin_col, "latitude_band"], observed=True
        ):
            if len(group) < min_games:
                continue
            residual = group["market_residual"].dropna()
            standard_error = (
                residual.std(ddof=1) / math.sqrt(len(residual))
                if len(residual) > 1
                else np.nan
            )
            rows.append(
                {
                    "weather_grouping": grouping,
                    "weather_bin": str(weather_bin),
                    "latitude_band": str(latitude_band),
                    "games": len(group),
                    "unique_venues": int(group["venue_id"].nunique()),
                    "avg_latitude": float(group["venue_latitude"].mean()),
                    "avg_weather_value": float(
                        group["temperature_f"].mean()
                        if grouping == "temperature"
                        else group["wind_mph"].mean()
                    ),
                    "avg_closing_total": float(group["closing_total"].mean()),
                    "avg_actual_total": float(group["actual_total_points"].mean()),
                    "avg_market_residual": float(residual.mean()),
                    "residual_ci_low": float(residual.mean() - 1.96 * standard_error),
                    "residual_ci_high": float(residual.mean() + 1.96 * standard_error),
                    "under_rate": float(group["went_under_context"].mean()),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    weights = result["games"]
    result["_weighted_residual"] = result["avg_market_residual"] * weights
    result["_weighted_under"] = result["under_rate"] * weights
    reference = (
        result.groupby(["weather_grouping", "weather_bin"], observed=True)
        .agg(
            weather_bin_games=("games", "sum"),
            weighted_residual=("_weighted_residual", "sum"),
            weighted_under=("_weighted_under", "sum"),
        )
        .reset_index()
    )
    reference["weather_bin_avg_residual"] = (
        reference["weighted_residual"] / reference["weather_bin_games"]
    )
    reference["weather_bin_under_rate"] = (
        reference["weighted_under"] / reference["weather_bin_games"]
    )
    result = result.merge(
        reference[
            [
                "weather_grouping",
                "weather_bin",
                "weather_bin_games",
                "weather_bin_avg_residual",
                "weather_bin_under_rate",
            ]
        ],
        on=["weather_grouping", "weather_bin"],
        how="left",
    )
    result["residual_vs_same_weather_bin"] = (
        result["avg_market_residual"] - result["weather_bin_avg_residual"]
    )
    result["under_rate_vs_same_weather_bin"] = (
        result["under_rate"] - result["weather_bin_under_rate"]
    )
    return (
        result.drop(columns=["_weighted_residual", "_weighted_under"])
        .sort_values(["weather_grouping", "weather_bin", "latitude_band"])
        .reset_index(drop=True)
    )


def _mean_lookup(
    df: pd.DataFrame,
    value_col: str,
    key_cols: list[str],
    min_games: int,
) -> dict[Any, float]:
    source = df.dropna(subset=key_cols + [value_col])
    if source.empty:
        return {}
    grouped = source.groupby(key_cols, observed=True)[value_col].agg(["count", "mean"])
    grouped = grouped[grouped["count"] >= min_games]
    lookup: dict[Any, float] = {}
    for key, value in grouped["mean"].items():
        lookup[key] = float(value)
    return lookup


def _distribution_lookup(
    df: pd.DataFrame,
    value_col: str,
    key_cols: list[str],
    min_games: int,
) -> dict[Any, np.ndarray]:
    source = df.dropna(subset=key_cols + [value_col])
    lookup: dict[Any, np.ndarray] = {}
    for key, group in source.groupby(key_cols, observed=True):
        if len(key_cols) == 1 and isinstance(key, tuple):
            key = key[0]
        values = np.sort(group[value_col].to_numpy(float))
        if len(values) >= min_games:
            lookup[key] = values
    return lookup


def build_context_reference(
    train: pd.DataFrame,
    min_venue_month_games: int = MIN_VENUE_MONTH_GAMES,
    min_band_month_games: int = MIN_BAND_MONTH_GAMES,
) -> dict[str, Any]:
    outdoor = train[train["outdoor"]].copy()
    temp = outdoor["temperature_f"].dropna()
    wind = outdoor["wind_mph"].dropna()
    return {
        "temperature_venue_month": _mean_lookup(
            outdoor,
            "temperature_f",
            ["venue_id", "calendar_month"],
            min_venue_month_games,
        ),
        "temperature_band_month": _mean_lookup(
            outdoor,
            "temperature_f",
            ["latitude_band", "calendar_month"],
            min_band_month_games,
        ),
        "temperature_month": _mean_lookup(
            outdoor, "temperature_f", ["calendar_month"], 1
        ),
        "temperature_global": float(temp.mean()) if not temp.empty else np.nan,
        "wind_venue_month": _distribution_lookup(
            outdoor, "wind_mph", ["venue_id", "calendar_month"], min_venue_month_games
        ),
        "wind_band_month": _distribution_lookup(
            outdoor,
            "wind_mph",
            ["latitude_band", "calendar_month"],
            min_band_month_games,
        ),
        "wind_month": _distribution_lookup(outdoor, "wind_mph", ["calendar_month"], 1),
        "wind_global": np.sort(wind.to_numpy(float)),
    }


def _single_key(value: object) -> object:
    if pd.isna(value):
        return None
    return int(value)


def _pair_key(first: object, second: object) -> tuple[object, object] | None:
    if pd.isna(first) or pd.isna(second):
        return None
    left: object = str(first) if isinstance(first, str) else float(first)
    return left, int(second)


def _percentile(value: object, distribution: np.ndarray | None) -> float:
    if pd.isna(value) or distribution is None or len(distribution) == 0:
        return np.nan
    rank = np.searchsorted(distribution, float(value), side="right") - 0.5
    return float(np.clip(rank / len(distribution), 0.0, 1.0))


def apply_context_features(df: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    temperature_normals: list[float] = []
    wind_percentiles: list[float] = []
    for venue, band, month, temperature, wind in zip(
        out["venue_id"],
        out["latitude_band"].astype(object),
        out["calendar_month"],
        out["temperature_f"],
        out["wind_mph"],
    ):
        venue_month = _pair_key(venue, month)
        band_month = _pair_key(str(band) if pd.notna(band) else np.nan, month)
        month_key = _single_key(month)

        normal = reference["temperature_venue_month"].get(venue_month)
        if normal is None:
            normal = reference["temperature_band_month"].get(band_month)
        if normal is None:
            normal = reference["temperature_month"].get(month_key)
        if normal is None:
            normal = reference["temperature_global"]
        temperature_normals.append(float(normal) if pd.notna(normal) else np.nan)

        distribution = reference["wind_venue_month"].get(venue_month)
        if distribution is None:
            distribution = reference["wind_band_month"].get(band_month)
        if distribution is None:
            distribution = reference["wind_month"].get(month_key)
        if distribution is None or len(distribution) == 0:
            distribution = reference["wind_global"]
        wind_percentiles.append(_percentile(wind, distribution))

    out["temperature_context_normal_f"] = temperature_normals
    out["temperature_anomaly_f"] = (
        out["temperature_f"] - out["temperature_context_normal_f"]
    )
    out["wind_local_percentile"] = wind_percentiles
    latitude_scaled = (out["venue_latitude"] - 35.0) / 5.0
    out["temperature_latitude_interaction"] = (
        out["temperature_anomaly_f"] * latitude_scaled
    )
    out["wind_latitude_interaction"] = (
        out["wind_local_percentile"] - 0.5
    ) * latitude_scaled

    new_features = [
        "venue_latitude",
        "temperature_context_normal_f",
        "temperature_anomaly_f",
        "wind_local_percentile",
        "temperature_latitude_interaction",
        "wind_latitude_interaction",
    ]
    indoor = ~out["outdoor"]
    out.loc[indoor, new_features] = np.nan
    return out


MODEL_FEATURES = {
    "baseline": [],
    "latitude_only": ["venue_latitude"],
    "local_weather_context": ["temperature_anomaly_f", "wind_local_percentile"],
    "full_latitude_context": [
        "venue_latitude",
        "temperature_anomaly_f",
        "wind_local_percentile",
        "temperature_latitude_interaction",
        "wind_latitude_interaction",
    ],
}


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
        train = apply_context_features(train, reference)
        test = apply_context_features(test, reference)

        eligible = test["context_ready"].copy()
        if "fbs_vs_fbs" in test.columns:
            eligible &= test["fbs_vs_fbs"].astype(bool)
        train = prep_features(train, cats)
        test = prep_features(test, cats)
        scored = test[eligible].copy()
        if scored.empty:
            continue

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
                "venue_longitude",
                "latitude_band",
                "closing_total",
                "actual_total_points",
                "market_residual",
                "temperature_f",
                "wind_mph",
                "temperature_context_normal_f",
                "temperature_anomaly_f",
                "wind_local_percentile",
                "temperature_latitude_interaction",
                "wind_latitude_interaction",
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
                    "paired_context_games": len(scored),
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


def bootstrap_mean_ci(
    values: np.ndarray,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5 or reps <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    chunk_size = 500
    for start in range(0, reps, chunk_size):
        count = min(chunk_size, reps - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _under_qualifier_metrics(
    predictions: pd.DataFrame, prediction_col: str
) -> dict[str, Any]:
    qualifiers = predictions[
        (predictions[prediction_col] <= -3.5) & (predictions["closing_total"] >= 56.0)
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
    baseline_abs_error = np.abs(baseline_pred - actual)
    rows: list[dict[str, Any]] = []
    for order, model_name in enumerate(MODEL_FEATURES):
        prediction_col = f"{model_name}_pred_market_residual"
        predicted = predictions[prediction_col].to_numpy(float)
        error = predicted - actual
        abs_error = np.abs(error)
        delta = abs_error - baseline_abs_error
        if model_name == "baseline":
            ci_low, ci_high = 0.0, 0.0
        else:
            ci_low, ci_high = bootstrap_mean_ci(
                delta, reps=bootstrap_reps, seed=BOOTSTRAP_SEED + order
            )
        row = {
            "model": model_name,
            "paired_games": len(predictions),
            "mae": float(abs_error.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "signed_projection_bias": float(error.mean()),
            "mae_delta_vs_baseline": float(delta.mean()),
            "mae_delta_ci_low": ci_low,
            "mae_delta_ci_high": ci_high,
            "test_seasons_improved_vs_baseline": 0,
            "test_seasons_evaluated": int(predictions["season"].nunique()),
        }
        if model_name != "baseline":
            improved = 0
            for _, group in predictions.groupby("season", observed=True):
                group_actual = group["market_residual"].to_numpy(float)
                group_base = np.abs(
                    group["baseline_pred_market_residual"].to_numpy(float)
                    - group_actual
                ).mean()
                group_challenger = np.abs(
                    group[prediction_col].to_numpy(float) - group_actual
                ).mean()
                improved += int(group_challenger < group_base)
            row["test_seasons_improved_vs_baseline"] = improved
        row.update(_under_qualifier_metrics(predictions, prediction_col))
        rows.append(row)
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
            predicted = group[f"{model_name}_pred_market_residual"].to_numpy(float)
            error = predicted - actual
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


def write_summary(
    coverage: pd.DataFrame,
    interactions: pd.DataFrame,
    model_summary: pd.DataFrame,
    by_season: pd.DataFrame,
) -> None:
    overall = coverage.iloc[0] if not coverage.empty else pd.Series(dtype=object)
    lines = [
        "# Climate-Context and Latitude Research",
        "",
        "**Status: retrospective research only. No production or prospective-ledger effect.**",
        "",
        "## Research question",
        "",
        "Do temperature and wind have different relationships with college-football totals at different latitudes, and does locally unusual weather add information beyond the raw forecast values?",
        "",
        "The outcome is `market_residual = actual_total_points - closing_total`, so the analysis asks what the market may not have fully priced rather than whether weather simply changes raw scoring.",
        "",
        "## Coordinate and weather coverage",
        "",
        f"Games with closing totals: **{int(overall.get('games', 0)):,}**",
        f"Games matched to venue coordinates: **{int(overall.get('with_coordinates', 0)):,}**",
        f"Outdoor FBS-vs-FBS games ready for paired temperature/wind context testing: **{int(overall.get('fbs_context_ready_games', 0)):,}**",
        "",
        coverage.to_markdown(index=False)
        if not coverage.empty
        else "_No coverage rows were produced._",
        "",
        "## Descriptive latitude-by-weather cells",
        "",
        "Cells require at least 25 games. These are in-sample descriptions and are not promotion evidence.",
        "",
        interactions.to_markdown(index=False)
        if not interactions.empty
        else "_No interaction cells met the minimum sample._",
        "",
        "## Paired walk-forward model comparison",
        "",
        "Every test season is predicted using only earlier seasons. Venue-month temperature baselines and wind distributions are also learned only from the training seasons. Negative MAE deltas favor the challenger.",
        "",
        model_summary.to_markdown(index=False)
        if not model_summary.empty
        else "_No walk-forward results were produced._",
        "",
        "## Results by test season",
        "",
        by_season.to_markdown(index=False)
        if not by_season.empty
        else "_No season-level results were produced._",
        "",
        "## Interpretation guardrails",
        "",
        "- Latitude is a geographic proxy, not a physical cause. Elevation, coastality, stadium design, season timing, and team acclimatization may contribute.",
        "- The local temperature reference is a venue-month historical game-weather baseline, not an official NOAA climate normal.",
        "- The wind feature is a percentile of prior outdoor game winds at the venue-month level, with latitude-band/month and month fallbacks when samples are sparse.",
        "- The descriptive interaction table is exploratory. The paired chronological model comparison is the primary retrospective evidence.",
        "- Qualifier hit rate and ROI are secondary and cannot override paired prediction error and season-to-season stability.",
        "- This script does not select or freeze a 2026 challenger. A prospective shadow version requires a separate documented decision after these results are reviewed.",
    ]
    out = ensure_dir(OUTPUT_DIR) / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    venues = fetch_venue_locations()
    raw = read_df("data/processed/modeling_dataset.csv")
    df = prepare_research_data(raw, venues)
    coverage = coverage_report(df)
    interactions = latitude_weather_interactions(df)
    predictions, diagnostics = walk_forward_predictions(df)
    summary = paired_model_summary(predictions)
    by_season = model_by_season(predictions)

    write_df(coverage, f"{OUTPUT_DIR}/coordinate_coverage.csv")
    write_df(interactions, f"{OUTPUT_DIR}/latitude_weather_interactions.csv")
    write_df(predictions, f"{OUTPUT_DIR}/walk_forward_predictions.csv")
    write_df(diagnostics, f"{OUTPUT_DIR}/walk_forward_diagnostics.csv")
    write_df(summary, f"{OUTPUT_DIR}/model_summary.csv")
    write_df(by_season, f"{OUTPUT_DIR}/model_by_season.csv")
    write_summary(coverage, interactions, summary, by_season)
    print("Wrote retrospective climate-context research outputs")


if __name__ == "__main__":
    main()
