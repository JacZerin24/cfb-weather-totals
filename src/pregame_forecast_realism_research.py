from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .deep_research import prep
from .model_bakeoff import feature_lists, prep_features, reg_models

DATA_PATH = Path("data/processed/modeling_dataset.csv")
FORECAST_PATH = Path("data/processed/pregame_forecast_full.csv")
VALIDATION_PATH = Path("outputs/pregame_forecast_realism/full_acquisition/validation.csv")
OUTPUT_DIR = Path("outputs/pregame_forecast_realism/model_results")

LEADS = [24, 12, 6]
PRIMARY_START_SEASON = 2019
PRIMARY_END_SEASON = 2025
MIN_FORECAST_TRAIN_GAMES = 1000
N_BOOT = 5000
SEED = 42

CORE_WEATHER = ["temperature_f", "dewpoint_f", "humidity", "wind_mph"]
PERIOD_WEATHER = ["precipitation", "snowfall"]
PRIMARY_WEATHER = CORE_WEATHER + PERIOD_WEATHER


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def recompute_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "wind_mph" in out.columns:
        out["wind_mph"] = pd.to_numeric(out["wind_mph"], errors="coerce")
        out["wind_bin"] = pd.cut(
            out["wind_mph"],
            bins=[-1, 5, 10, 15, 20, 200],
            labels=["0-5", "5-10", "10-15", "15-20", "20+"],
        )
    if "temperature_f" in out.columns:
        out["temperature_f"] = pd.to_numeric(out["temperature_f"], errors="coerce")
        out["temp_bin"] = pd.cut(
            out["temperature_f"],
            bins=[-100, 35, 50, 70, 85, 200],
            labels=["<=35", "35-50", "50-70", "70-85", "85+"],
        )
    return out


def prepare_historical(raw: pd.DataFrame) -> pd.DataFrame:
    hist = prep(raw)
    hist["game_id"] = pd.to_numeric(hist["game_id"], errors="coerce")
    hist["season"] = pd.to_numeric(hist["season"], errors="coerce")
    hist = hist.dropna(subset=["game_id", "season", "market_residual"]).copy()
    hist["game_id"] = hist["game_id"].astype(int)
    hist["season"] = hist["season"].astype(int)
    hist = recompute_bins(hist)
    return hist


def prepare_forecasts(raw: pd.DataFrame) -> pd.DataFrame:
    fcst = raw.copy()
    fcst["game_id"] = pd.to_numeric(fcst["game_id"], errors="coerce")
    fcst["season"] = pd.to_numeric(fcst["season"], errors="coerce")
    fcst["lead_hours"] = pd.to_numeric(fcst["lead_hours"], errors="coerce")
    fcst = fcst.dropna(subset=["game_id", "season", "lead_hours"]).copy()
    fcst["game_id"] = fcst["game_id"].astype(int)
    fcst["season"] = fcst["season"].astype(int)
    fcst["lead_hours"] = fcst["lead_hours"].astype(int)
    fcst = fcst[fcst["lead_hours"].isin(LEADS)].copy()
    return fcst.drop_duplicates(["game_id", "lead_hours"], keep="last")


def complete_common_ids(fcst: pd.DataFrame) -> set[int]:
    good: set[int] = set()
    for game_id, group in fcst.groupby("game_id", observed=True):
        if set(group["lead_hours"].astype(int)) != set(LEADS) or len(group) != len(LEADS):
            continue
        if "nbm_station_id" in group.columns and group["nbm_station_id"].astype(str).nunique() != 1:
            continue
        if not group[CORE_WEATHER].notna().all(axis=1).all():
            continue
        good.add(int(game_id))
    return good


def lead_frame(fcst: pd.DataFrame, lead: int) -> pd.DataFrame:
    cols = ["game_id", "lead_hours"] + [c for c in PRIMARY_WEATHER if c in fcst.columns]
    if "wind_direction_degrees" in fcst.columns:
        cols.append("wind_direction_degrees")
    if "precip_probability_pct" in fcst.columns:
        cols.append("precip_probability_pct")
    out = fcst.loc[fcst["lead_hours"].eq(lead), cols].copy()
    rename = {c: f"fcst_{c}" for c in cols if c not in {"game_id", "lead_hours"}}
    return out.rename(columns=rename).drop(columns=["lead_hours"])


def apply_forecast_weather(
    historical_rows: pd.DataFrame,
    forecasts: pd.DataFrame,
    lead: int,
    mode: str = "primary",
) -> pd.DataFrame:
    if mode not in {"primary", "core"}:
        raise ValueError(f"Unknown substitution mode: {mode}")
    f = lead_frame(forecasts, lead)
    out = historical_rows.merge(f, on="game_id", how="inner", validate="one_to_one")
    replace = PRIMARY_WEATHER if mode == "primary" else CORE_WEATHER
    for col in replace:
        fc = f"fcst_{col}"
        if fc in out.columns:
            out[col] = pd.to_numeric(out[fc], errors="coerce")
    # Live NWS scoring does not provide pressure; realized historical pressure is prohibited.
    if "pressure" in out.columns:
        out["pressure"] = np.nan
    return recompute_bins(out)


def fit_hgb(train: pd.DataFrame, nums: list[str], cats: list[str]):
    model = reg_models(nums, cats)["hist_gradient_boosting"]
    train_ready = prep_features(train, cats)
    model.fit(train_ready[nums + cats], train_ready["market_residual"])
    return model


def predict_hgb(model, frame: pd.DataFrame, nums: list[str], cats: list[str]) -> np.ndarray:
    ready = prep_features(frame, cats)
    return model.predict(ready[nums + cats])


def prediction_rows(frame: pd.DataFrame, pred: np.ndarray, variant: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": frame["game_id"].astype(int).to_numpy(),
            "season": frame["season"].astype(int).to_numpy(),
            "variant": variant,
            "actual_market_residual": pd.to_numeric(frame["market_residual"], errors="coerce").to_numpy(),
            "pred_market_residual": np.asarray(pred, dtype=float),
            "closing_total": pd.to_numeric(frame["closing_total"], errors="coerce").to_numpy(),
            "actual_total_points": pd.to_numeric(frame["actual_total_points"], errors="coerce").to_numpy(),
        }
    )


def run_current_production_realism(hist: pd.DataFrame, fcst: pd.DataFrame, common_ids: set[int]) -> pd.DataFrame:
    nums, cats = feature_lists(hist)
    parts: list[pd.DataFrame] = []
    for season in range(PRIMARY_START_SEASON, PRIMARY_END_SEASON + 1):
        train = hist[hist["season"] < season].copy()
        test = hist[hist["season"].eq(season) & hist["game_id"].isin(common_ids)].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        model = fit_hgb(train, nums, cats)
        ref_pred = predict_hgb(model, test, nums, cats)
        parts.append(prediction_rows(test, ref_pred, "retrospective_control"))
        for lead in LEADS:
            primary = apply_forecast_weather(test, fcst, lead, mode="primary")
            if len(primary) != len(test):
                raise RuntimeError(f"Primary {lead}h substitution changed paired test size in {season}.")
            pred = predict_hgb(model, primary, nums, cats)
            parts.append(prediction_rows(primary, pred, f"production_forecast_{lead}h"))

            core = apply_forecast_weather(test, fcst, lead, mode="core")
            if len(core) != len(test):
                raise RuntimeError(f"Core {lead}h substitution changed paired test size in {season}.")
            core_pred = predict_hgb(model, core, nums, cats)
            parts.append(prediction_rows(core, core_pred, f"production_core_forecast_{lead}h"))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def run_forecast_trained(hist: pd.DataFrame, fcst: pd.DataFrame, common_ids: set[int]) -> pd.DataFrame:
    nums, cats = feature_lists(hist)
    research_hist = hist[
        hist["game_id"].isin(common_ids)
        & hist["season"].between(PRIMARY_START_SEASON, PRIMARY_END_SEASON)
    ].copy()
    parts: list[pd.DataFrame] = []
    for lead in LEADS:
        for season in range(PRIMARY_START_SEASON + 1, PRIMARY_END_SEASON + 1):
            train_hist = research_hist[research_hist["season"] < season].copy()
            test_hist = research_hist[research_hist["season"].eq(season)].copy()
            if len(train_hist) < MIN_FORECAST_TRAIN_GAMES or len(test_hist) < 100:
                continue
            train_fcst = apply_forecast_weather(train_hist, fcst, lead, mode="primary")
            test_fcst = apply_forecast_weather(test_hist, fcst, lead, mode="primary")
            if len(train_fcst) != len(train_hist) or len(test_fcst) != len(test_hist):
                raise RuntimeError(f"Forecast-trained {lead}h common sample changed in {season}.")

            matched_model = fit_hgb(train_hist, nums, cats)
            matched_pred = predict_hgb(matched_model, test_fcst, nums, cats)
            parts.append(
                prediction_rows(
                    test_fcst,
                    matched_pred,
                    f"matched_retro_train_forecast_score_{lead}h",
                )
            )

            fcst_model = fit_hgb(train_fcst, nums, cats)
            fcst_pred = predict_hgb(fcst_model, test_fcst, nums, cats)
            parts.append(prediction_rows(test_fcst, fcst_pred, f"forecast_trained_{lead}h"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def metric_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in predictions.groupby("variant", observed=True):
        err = group["pred_market_residual"] - group["actual_market_residual"]
        rows.append(
            {
                "variant": variant,
                "games": len(group),
                "test_seasons": int(group["season"].nunique()),
                "first_test_season": int(group["season"].min()),
                "last_test_season": int(group["season"].max()),
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(err)))),
                "prediction_bias": float(err.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def season_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, season), group in predictions.groupby(["variant", "season"], observed=True):
        err = group["pred_market_residual"] - group["actual_market_residual"]
        rows.append(
            {
                "variant": variant,
                "season": int(season),
                "games": len(group),
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(err)))),
                "prediction_bias": float(err.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "season"]).reset_index(drop=True)


def game_bootstrap_ci(delta: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(delta)
    for i in range(n_boot):
        means[i] = float(delta[rng.integers(0, n, n)].mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def season_cluster_ci(paired: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED + 1) -> tuple[float, float]:
    grouped = paired.groupby("season", observed=True)["delta_abs_error"].agg(["sum", "count"])
    if grouped.empty:
        return np.nan, np.nan
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    k = len(grouped)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, k, k)
        means[i] = float(sums[idx].sum() / counts[idx].sum())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compare_variants(
    predictions: pd.DataFrame,
    reference: str,
    challenger: str,
    comparison: str,
    gate_kind: str,
    minimum_material_delta: float = 0.03,
) -> dict[str, Any]:
    ref = predictions[predictions["variant"].eq(reference)][
        ["game_id", "season", "actual_market_residual", "pred_market_residual"]
    ].rename(columns={"pred_market_residual": "ref_pred"})
    chal = predictions[predictions["variant"].eq(challenger)][
        ["game_id", "season", "actual_market_residual", "pred_market_residual"]
    ].rename(columns={"pred_market_residual": "chal_pred", "actual_market_residual": "chal_actual"})
    paired = ref.merge(chal, on=["game_id", "season"], how="inner", validate="one_to_one")
    if paired.empty:
        return {
            "comparison": comparison,
            "reference": reference,
            "challenger": challenger,
            "gate_kind": gate_kind,
            "paired_games": 0,
            "gate_pass": False,
        }
    if not np.allclose(
        paired["actual_market_residual"].to_numpy(dtype=float),
        paired["chal_actual"].to_numpy(dtype=float),
        equal_nan=False,
    ):
        raise RuntimeError(f"Outcome mismatch in paired comparison {comparison}.")
    paired["ref_abs_error"] = (paired["ref_pred"] - paired["actual_market_residual"]).abs()
    paired["chal_abs_error"] = (paired["chal_pred"] - paired["actual_market_residual"]).abs()
    paired["delta_abs_error"] = paired["chal_abs_error"] - paired["ref_abs_error"]
    mean_delta = float(paired["delta_abs_error"].mean())
    game_low, game_high = game_bootstrap_ci(paired["delta_abs_error"].to_numpy())
    cluster_low, cluster_high = season_cluster_ci(paired)
    per_season = paired.groupby("season", observed=True)["delta_abs_error"].mean()
    improved = int((per_season < 0).sum())
    worsened = int((per_season > 0).sum())
    seasons = int(len(per_season))
    improve_fraction = improved / seasons if seasons else np.nan
    worse_fraction = worsened / seasons if seasons else np.nan

    if gate_kind == "improvement":
        gate_pass = bool(
            mean_delta < 0
            and game_high < 0
            and cluster_high < 0
            and improve_fraction >= 0.70
            and mean_delta <= -minimum_material_delta
        )
    elif gate_kind == "degradation":
        gate_pass = bool(
            mean_delta > 0
            and game_low > 0
            and cluster_low > 0
            and worse_fraction >= 0.70
            and mean_delta >= minimum_material_delta
        )
    elif gate_kind == "diagnostic":
        gate_pass = False
    else:
        raise ValueError(f"Unknown gate kind: {gate_kind}")

    return {
        "comparison": comparison,
        "reference": reference,
        "challenger": challenger,
        "gate_kind": gate_kind,
        "paired_games": len(paired),
        "test_seasons": seasons,
        "mean_mae_delta": mean_delta,
        "game_bootstrap_ci_low": game_low,
        "game_bootstrap_ci_high": game_high,
        "season_cluster_ci_low": cluster_low,
        "season_cluster_ci_high": cluster_high,
        "seasons_improved": improved,
        "seasons_worsened": worsened,
        "season_improvement_fraction": improve_fraction,
        "season_worsening_fraction": worse_fraction,
        "material_delta_floor": minimum_material_delta,
        "gate_pass": gate_pass,
    }


def build_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lead in LEADS:
        rows.append(
            compare_variants(
                predictions,
                "retrospective_control",
                f"production_forecast_{lead}h",
                f"production_realism_{lead}h_vs_retrospective",
                "degradation",
            )
        )
        rows.append(
            compare_variants(
                predictions,
                "retrospective_control",
                f"production_core_forecast_{lead}h",
                f"core_sensitivity_{lead}h_vs_retrospective",
                "diagnostic",
            )
        )

    for earlier, later in [(24, 12), (12, 6), (24, 6)]:
        rows.append(
            compare_variants(
                predictions,
                f"production_forecast_{earlier}h",
                f"production_forecast_{later}h",
                f"update_{later}h_vs_{earlier}h",
                "improvement",
            )
        )

    for lead in LEADS:
        rows.append(
            compare_variants(
                predictions,
                f"production_forecast_{lead}h",
                f"forecast_trained_{lead}h",
                f"forecast_trained_{lead}h_vs_production_style",
                "improvement",
            )
        )
        rows.append(
            compare_variants(
                predictions,
                f"matched_retro_train_forecast_score_{lead}h",
                f"forecast_trained_{lead}h",
                f"forecast_trained_{lead}h_vs_matched_population_control",
                "diagnostic",
            )
        )
    return pd.DataFrame(rows)


def weather_effect_persistence(hist: pd.DataFrame, fcst: pd.DataFrame, common_ids: set[int]) -> pd.DataFrame:
    base = hist[
        hist["game_id"].isin(common_ids)
        & hist["season"].between(PRIMARY_START_SEASON, PRIMARY_END_SEASON)
    ].copy()
    representations: list[tuple[str, pd.DataFrame]] = [("retrospective", base)]
    for lead in LEADS:
        representations.append((f"{lead}h", apply_forecast_weather(base, fcst, lead, mode="primary")))

    rows: list[dict[str, Any]] = []
    for representation, frame in representations:
        wind = pd.to_numeric(frame.get("wind_mph"), errors="coerce")
        temp = pd.to_numeric(frame.get("temperature_f"), errors="coerce")
        humidity = pd.to_numeric(frame.get("humidity"), errors="coerce")
        precip = pd.to_numeric(frame.get("precipitation"), errors="coerce")
        total = pd.to_numeric(frame.get("closing_total"), errors="coerce")
        screens = {
            "wind_15_plus": wind >= 15,
            "wind_20_plus": wind >= 20,
            "cold_35_or_lower": temp <= 35,
            "cold_40_wind_12": (temp <= 40) & (wind >= 12),
            "total_60_wind_10": (total >= 60) & (wind >= 10),
            "total_58_wind_12": (total >= 58) & (wind >= 12),
            "total_60_rh_80": (total >= 60) & (humidity >= 80),
            "precip_any_diagnostic": precip > 0,
        }
        for screen, mask in screens.items():
            group = frame[mask.fillna(False)].copy()
            if len(group) < 25:
                continue
            rows.append(
                {
                    "representation": representation,
                    "screen": screen,
                    "games": len(group),
                    "avg_market_residual": float(pd.to_numeric(group["market_residual"], errors="coerce").mean()),
                    "under_rate": float(
                        (
                            pd.to_numeric(group["actual_total_points"], errors="coerce")
                            < pd.to_numeric(group["closing_total"], errors="coerce")
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["screen", "representation"]).reset_index(drop=True)


def validate_source_artifact() -> None:
    if not VALIDATION_PATH.exists():
        raise RuntimeError("Full-acquisition validation file is missing.")
    validation = pd.read_csv(VALIDATION_PATH)
    if validation.empty or not as_bool(validation.iloc[0].get("ready_for_modeling", False)):
        raise RuntimeError("Full archived NBM acquisition is not marked ready_for_modeling.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validate_source_artifact()
    if not DATA_PATH.exists() or not FORECAST_PATH.exists():
        raise RuntimeError("Required modeling or archived forecast data is missing.")

    hist = prepare_historical(pd.read_csv(DATA_PATH, low_memory=False))
    fcst = prepare_forecasts(pd.read_csv(FORECAST_PATH, low_memory=False))
    common_ids = complete_common_ids(fcst)
    if len(common_ids) < 4500:
        raise RuntimeError(f"Unexpectedly small complete forecast common sample: {len(common_ids):,}")

    current = run_current_production_realism(hist, fcst, common_ids)
    trained = run_forecast_trained(hist, fcst, common_ids)
    predictions = pd.concat([current, trained], ignore_index=True)
    if predictions.empty:
        raise RuntimeError("No model predictions were produced.")

    summary = metric_summary(predictions)
    seasons = season_summary(predictions)
    comparisons = build_comparisons(predictions)
    persistence = weather_effect_persistence(hist, fcst, common_ids)

    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "variant_summary.csv", index=False)
    seasons.to_csv(OUTPUT_DIR / "season_summary.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR / "paired_comparisons.csv", index=False)
    persistence.to_csv(OUTPUT_DIR / "weather_effect_persistence.csv", index=False)
    pd.DataFrame(
        [
            {
                "complete_common_games": len(common_ids),
                "primary_start_season": PRIMARY_START_SEASON,
                "primary_end_season": PRIMARY_END_SEASON,
                "forecast_train_min_games": MIN_FORECAST_TRAIN_GAMES,
                "bootstrap_resamples": N_BOOT,
                "production_files_modified": False,
            }
        ]
    ).to_csv(OUTPUT_DIR / "run_metadata.csv", index=False)

    print("--- variant summary ---")
    print(summary.to_string(index=False))
    print("--- paired comparisons ---")
    print(comparisons.to_string(index=False))
    print("--- weather-effect persistence ---")
    print(persistence.to_string(index=False))


if __name__ == "__main__":
    main()
