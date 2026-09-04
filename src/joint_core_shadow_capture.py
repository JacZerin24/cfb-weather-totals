from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from .climate_context_research import (
    add_latitude_band,
    apply_context_features,
    build_context_reference,
    prepare_research_data,
)
from .joint_weather_context_research import CLIMATE_FEATURES, add_orientation_features
from .model_bakeoff import feature_lists, prep_features, reg_models
from .orientation_shadow_capture import (
    as_bool,
    challenger_status,
    fetch_shadow_direction,
    live_game_context,
)
from .predict_week import add_live_categories, merge_prior_team_features
from .utils import ROOT, ensure_dir, load_yaml, read_df

CHALLENGER_CONFIG_PATH = ROOT / "config/joint_core_challenger_2026.yml"
EVAL_PROTOCOL_PATH = ROOT / "config/joint_core_evaluation_protocol_2026.yml"
SHADOW_DIR = ROOT / "outputs/joint_core_shadow/2026"
LOCATION_PATH = ROOT / "data/reference/stadium_locations.csv"
CHALLENGER_VERSION = "joint-core-weather-context-hgb-v0.1"
SEASON = 2026


def sha256_path(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_specs() -> tuple[dict, dict]:
    challenger = load_yaml("config/joint_core_challenger_2026.yml")
    protocol = load_yaml("config/joint_core_evaluation_protocol_2026.yml")
    if str(challenger.get("version")) != CHALLENGER_VERSION:
        raise RuntimeError("Joint-core challenger version does not match capture code.")
    if str(challenger.get("status")) != "research_only_shadow":
        raise RuntimeError("Joint-core challenger must remain research-only shadow.")
    if int(challenger.get("season", 0)) != SEASON or int(protocol.get("season", 0)) != SEASON:
        raise RuntimeError("Joint-core shadow must remain scoped to the 2026 season.")
    if str(protocol.get("challenger_version")) != CHALLENGER_VERSION:
        raise RuntimeError("Joint-core evaluation protocol version does not match capture code.")
    if pd.Timestamp(challenger["frozen_at_utc"]) != pd.Timestamp(protocol["frozen_at_utc"]):
        raise RuntimeError("Challenger and protocol freeze times differ.")
    return challenger, protocol


def historical_training_bundle() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    venues = read_df(str(LOCATION_PATH.relative_to(ROOT))).copy()
    raw = read_df("data/processed/modeling_dataset.csv")
    historical = prepare_research_data(raw, venues)
    historical["season"] = pd.to_numeric(historical.get("season"), errors="coerce")
    historical = historical[historical["season"] < SEASON].copy()
    if historical.empty:
        raise RuntimeError("No pre-2026 historical training data are available for joint-core shadow.")

    historical = add_orientation_features(historical)
    context_reference = build_context_reference(historical)
    historical = apply_context_features(historical, context_reference)
    return historical, context_reference, venues


def add_live_context_features(
    board: pd.DataFrame,
    context_reference: dict,
) -> pd.DataFrame:
    live = add_live_categories(merge_prior_team_features(board.copy()))
    live["outdoor"] = ~live["game_indoors_bool"].map(as_bool)
    live["venue_id"] = pd.to_numeric(live.get("venue_id"), errors="coerce")
    live["venue_latitude"] = pd.to_numeric(live.get("venue_latitude"), errors="coerce")
    live["venue_longitude"] = pd.to_numeric(live.get("venue_longitude"), errors="coerce")
    live["latitude_band"] = add_latitude_band(live["venue_latitude"])
    kickoff = pd.to_datetime(live.get("start_date"), utc=True, errors="coerce")
    live["calendar_month"] = kickoff.dt.month.astype("Int64")
    live["coordinate_matched"] = live[["venue_latitude", "venue_longitude"]].notna().all(axis=1)
    live["context_ready"] = (
        live["outdoor"]
        & live["coordinate_matched"]
        & live["calendar_month"].notna()
        & live["temperature_f"].notna()
        & live["wind_mph"].notna()
    )
    live = apply_context_features(live, context_reference)
    live = add_orientation_features(live)
    return live


def prepare_live_board(board: pd.DataFrame, context_reference: dict) -> pd.DataFrame:
    board = board.copy()
    board["game_id"] = pd.to_numeric(board.get("game_id"), errors="coerce")
    context = live_game_context(SEASON)
    board = board.drop(
        columns=[c for c in ["venue_id", "neutral_site", "conference_game"] if c in board.columns],
        errors="ignore",
    )
    board = board.merge(context, on="game_id", how="left")
    direction = fetch_shadow_direction(board)
    board = board.merge(direction, on="game_id", how="left")
    board["wind_direction_degrees"] = pd.to_numeric(
        board.get("shadow_wind_direction_degrees"), errors="coerce"
    )
    return add_live_context_features(board, context_reference)


def score_joint_core(board: pd.DataFrame) -> pd.DataFrame:
    historical, context_reference, _ = historical_training_bundle()
    base_nums, cats = feature_lists(historical)
    nums = list(dict.fromkeys(base_nums + ["crosswind_mph", "alongwind_mph", *CLIMATE_FEATURES]))

    historical = prep_features(historical, cats)
    model = reg_models(nums, cats)["hist_gradient_boosting"]
    model.fit(historical[nums + cats], historical["market_residual"])

    live = prepare_live_board(board, context_reference)
    for col in nums:
        if col not in live.columns:
            live[col] = np.nan
        live[col] = pd.to_numeric(live[col], errors="coerce")
    for col in cats:
        if col not in live.columns:
            live[col] = "missing"
        live[col] = live[col].astype(str).fillna("missing")
    live = prep_features(live, cats)

    fbs = live.get("division_track", pd.Series("", index=live.index)).astype(str).str.upper().eq("FBS")
    fbs_vs_fbs = live.get("fbs_vs_fbs", pd.Series(False, index=live.index)).map(as_bool)
    ready = live.get("joint_ready", pd.Series(False, index=live.index)).map(as_bool)
    mask = fbs & fbs_vs_fbs & ready & live["closing_total"].notna()

    live["challenger_pred_market_residual"] = np.nan
    if mask.any():
        live.loc[mask, "challenger_pred_market_residual"] = model.predict(live.loc[mask, nums + cats])
    live["challenger_projected_total"] = live["closing_total"] + live["challenger_pred_market_residual"]
    live["challenger_abs_edge"] = live["challenger_pred_market_residual"].abs()
    live["challenger_status"] = [
        challenger_status(
            float(pred) if pd.notna(pred) else np.nan,
            float(total) if pd.notna(total) else np.nan,
            weather,
            tbd,
        )
        for pred, total, weather, tbd in zip(
            live["challenger_pred_market_residual"],
            live["closing_total"],
            live.get("nws_status", pd.Series("", index=live.index)),
            live.get("start_time_tbd", pd.Series(False, index=live.index)),
        )
    ]
    return live


def official_snapshot_eligible(event_name: str, event_schedule: str, captured: pd.Timestamp, protocol: dict) -> bool:
    policy = protocol["official_entry_policy"]
    freeze = pd.Timestamp(protocol["frozen_at_utc"])
    if freeze.tzinfo is None:
        freeze = freeze.tz_localize("UTC")
    else:
        freeze = freeze.tz_convert("UTC")
    return bool(
        captured >= freeze
        and event_name == str(policy["eligible_event_name"])
        and event_schedule in {str(v) for v in policy.get("eligible_crons", [])}
    )


def capture() -> tuple[pd.DataFrame, str]:
    challenger, protocol = load_frozen_specs()
    board = read_df("outputs/weekly_board.csv").copy()
    if board.empty:
        raise RuntimeError("weekly_board.csv is empty.")
    season_values = pd.to_numeric(board.get("season"), errors="coerce").dropna()
    if season_values.empty or int(season_values.max()) != SEASON:
        raise RuntimeError("Joint-core shadow capture expects the 2026 weekly board.")

    scored = score_joint_core(board)
    captured = pd.Timestamp.now(tz="UTC")
    event_name = os.getenv("JOINT_CORE_SHADOW_EVENT_NAME", os.getenv("GITHUB_EVENT_NAME", "")).strip()
    event_schedule = os.getenv("JOINT_CORE_SHADOW_EVENT_SCHEDULE", "").strip()
    official_eligible = official_snapshot_eligible(event_name, event_schedule, captured, protocol)

    scored["challenger_version"] = CHALLENGER_VERSION
    scored["baseline_status"] = scored.get("status", "")
    scored["baseline_pred_market_residual"] = pd.to_numeric(scored.get("pred_market_residual"), errors="coerce")
    scored["challenger_minus_baseline_edge"] = (
        scored["challenger_pred_market_residual"] - scored["baseline_pred_market_residual"]
    )
    scored["status_changed"] = scored["challenger_status"].astype(str) != scored["baseline_status"].astype(str)
    scored["captured_at_utc"] = captured.isoformat()
    scored["github_event_name"] = event_name
    scored["github_event_schedule"] = event_schedule
    scored["github_run_id"] = os.getenv("GITHUB_RUN_ID", "")
    scored["github_run_attempt"] = os.getenv("GITHUB_RUN_ATTEMPT", "")
    scored["github_sha"] = os.getenv("GITHUB_SHA", "")
    scored["evaluation_protocol_version"] = str(protocol["protocol_version"])
    scored["evaluation_protocol_sha256"] = sha256_path(EVAL_PROTOCOL_PATH)
    scored["challenger_config_sha256"] = sha256_path(CHALLENGER_CONFIG_PATH)
    scored["official_evaluation_eligible"] = bool(official_eligible)
    scored["entry_lead_minutes"] = (
        pd.to_datetime(scored.get("start_date"), utc=True, errors="coerce") - captured
    ).dt.total_seconds() / 60.0
    scored["post_freeze_capture"] = captured >= pd.Timestamp(protocol["frozen_at_utc"])
    scored["joint_core_ready"] = (
        scored.get("division_track", pd.Series("", index=scored.index)).astype(str).str.upper().eq("FBS")
        & scored.get("fbs_vs_fbs", pd.Series(False, index=scored.index)).map(as_bool)
        & scored.get("outdoor", pd.Series(False, index=scored.index)).map(as_bool)
        & scored.get("context_ready", pd.Series(False, index=scored.index)).map(as_bool)
        & scored.get("orientation_ready", pd.Series(False, index=scored.index)).map(as_bool)
        & scored["baseline_pred_market_residual"].notna()
        & scored["challenger_pred_market_residual"].notna()
    )
    scored["research_only"] = True

    keep = [
        c
        for c in [
            "captured_at_utc",
            "github_event_name",
            "github_event_schedule",
            "github_run_id",
            "github_run_attempt",
            "github_sha",
            "evaluation_protocol_version",
            "evaluation_protocol_sha256",
            "challenger_config_sha256",
            "official_evaluation_eligible",
            "post_freeze_capture",
            "entry_lead_minutes",
            "challenger_version",
            "research_only",
            "joint_core_ready",
            "context_ready",
            "orientation_ready",
            "outdoor",
            "fbs_vs_fbs",
            "season",
            "week",
            "game_id",
            "start_date",
            "away_team",
            "home_team",
            "venue_id",
            "venue_name",
            "division_track",
            "closing_total",
            "line_provider",
            "baseline_status",
            "baseline_pred_market_residual",
            "model_projected_total",
            "challenger_status",
            "challenger_pred_market_residual",
            "challenger_projected_total",
            "challenger_abs_edge",
            "challenger_minus_baseline_edge",
            "status_changed",
            "temperature_f",
            "temperature_context_normal_f",
            "temperature_anomaly_f",
            "venue_latitude",
            "wind_mph",
            "wind_local_percentile",
            "temperature_latitude_interaction",
            "wind_latitude_interaction",
            "shadow_wind_direction_degrees",
            "field_axis_deg",
            "axis_uncertainty_deg",
            "wind_field_angle_deg",
            "crosswind_mph",
            "alongwind_mph",
            "nws_status",
            "shadow_weather_status",
            "weather_summary",
        ]
        if c in scored.columns
    ]
    output = scored[keep].copy()

    ensure_dir(SHADOW_DIR)
    payload = output.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    stamp = captured.strftime("%Y%m%dT%H%M%SZ")
    run = os.getenv("GITHUB_RUN_ID", "local")
    attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
    snapshot = SHADOW_DIR / f"shadow_{stamp}_run{run}_a{attempt}_{digest[:12]}.csv"
    with open(snapshot, "xb") as handle:
        handle.write(payload)
    output.to_csv(SHADOW_DIR / "latest.csv", index=False)

    manifest = SHADOW_DIR / "manifest.csv"
    row = pd.DataFrame(
        [
            {
                "captured_at_utc": captured.isoformat(),
                "snapshot_file": snapshot.name,
                "snapshot_sha256": digest,
                "evaluation_protocol_version": str(protocol["protocol_version"]),
                "evaluation_protocol_sha256": sha256_path(EVAL_PROTOCOL_PATH),
                "challenger_config_sha256": sha256_path(CHALLENGER_CONFIG_PATH),
                "challenger_version": CHALLENGER_VERSION,
                "github_event_name": event_name,
                "github_event_schedule": event_schedule,
                "official_evaluation_eligible": bool(official_eligible),
                "games": len(output),
                "fbs_games": int(output.get("division_track", pd.Series("", index=output.index)).astype(str).str.upper().eq("FBS").sum()),
                "joint_core_ready_games": int(output.get("joint_core_ready", pd.Series(False, index=output.index)).map(as_bool).sum()),
                "status_disagreements": int(output.get("status_changed", pd.Series(False, index=output.index)).map(as_bool).sum()),
                "github_run_id": run,
                "github_run_attempt": attempt,
                "github_sha": os.getenv("GITHUB_SHA", ""),
            }
        ]
    )
    if manifest.exists():
        old = pd.read_csv(manifest)
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(manifest, index=False)
    print(
        f"Wrote research-only joint-core shadow snapshot: {snapshot}; "
        f"official-evaluation-eligible={official_eligible}"
    )
    return output, str(snapshot)


def main() -> None:
    capture()


if __name__ == "__main__":
    main()
