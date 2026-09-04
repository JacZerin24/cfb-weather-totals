from __future__ import annotations

import pandas as pd

from .joint_core_shadow_capture import (
    CHALLENGER_VERSION,
    load_frozen_specs,
    official_snapshot_eligible,
)
from .joint_core_shadow_grade import prospective_signal, select_official_shadow_entries

EXPECTED_ADDED = [
    "crosswind_mph",
    "alongwind_mph",
    "venue_latitude",
    "temperature_anomaly_f",
    "wind_local_percentile",
    "temperature_latitude_interaction",
    "wind_latitude_interaction",
]


def test_frozen_specs() -> None:
    challenger, protocol = load_frozen_specs()
    assert challenger["version"] == CHALLENGER_VERSION
    assert protocol["challenger_version"] == CHALLENGER_VERSION
    assert challenger["challenger"]["added_numeric_features"] == EXPECTED_ADDED
    assert challenger["challenger"]["training_cutoff"] == "season < 2026"
    assert challenger["separation"]["may_change_weekly_board_status"] is False
    assert challenger["separation"]["may_change_prospective_ledger"] is False
    assert challenger["separation"]["may_change_weekly_picks"] is False


def test_capture_eligibility() -> None:
    _, protocol = load_frozen_specs()
    after = pd.Timestamp("2026-09-05T07:17:00Z")
    before = pd.Timestamp("2026-09-04T13:59:59Z")
    assert official_snapshot_eligible("schedule", "17 7 * * 6", after, protocol)
    assert not official_snapshot_eligible("schedule", "17 7 * * 6", before, protocol)
    assert not official_snapshot_eligible("push", "", after, protocol)
    assert not official_snapshot_eligible("schedule", "17 10 * * 1", after, protocol)


def test_official_selection() -> None:
    protocol_version = load_frozen_specs()[1]["protocol_version"]
    base = {
        "start_date": "2026-09-05T19:00:00Z",
        "game_id": 123,
        "division_track": "FBS",
        "fbs_vs_fbs": True,
        "outdoor": True,
        "joint_core_ready": True,
        "challenger_version": CHALLENGER_VERSION,
        "evaluation_protocol_version": protocol_version,
        "github_event_name": "schedule",
        "github_event_schedule": "17 7 * * 6",
        "official_evaluation_eligible": True,
        "baseline_pred_market_residual": -4.0,
        "challenger_pred_market_residual": -5.0,
    }
    rows = [
        {
            **base,
            "captured_at_utc": "2026-09-05T07:17:00Z",
            "github_run_id": "100",
            "github_run_attempt": 1,
        },
        {
            **base,
            "captured_at_utc": "2026-09-05T11:17:00Z",
            "github_run_id": "101",
            "github_run_attempt": 1,
        },
        {
            **base,
            "captured_at_utc": "2026-09-05T11:18:00Z",
            "github_run_id": "101",
            "github_run_attempt": 2,
        },
        {
            **base,
            "captured_at_utc": "2026-09-04T13:50:00Z",
            "github_run_id": "99",
            "github_run_attempt": 1,
        },
    ]
    selected = select_official_shadow_entries(pd.DataFrame(rows))
    assert len(selected) == 1
    assert str(selected.iloc[0]["github_run_id"]) == "101"
    assert int(selected.iloc[0]["github_run_attempt"]) == 1
    assert selected.iloc[0]["captured_at_utc"] == pd.Timestamp("2026-09-05T11:17:00Z")


def test_signal_labels() -> None:
    assert prospective_signal(4, -1.0, -0.1) == "INSUFFICIENT_SAMPLE"
    assert prospective_signal(20, -0.2, -0.01) == "PROSPECTIVE_MAE_SUPPORTED"
    assert prospective_signal(20, -0.2, 0.10) == "PROSPECTIVE_MAE_FAVORABLE_UNCERTAIN"
    assert prospective_signal(20, 0.01, 0.20) == "PROSPECTIVE_MAE_NOT_FAVORABLE"


def main() -> None:
    test_frozen_specs()
    test_capture_eligibility()
    test_official_selection()
    test_signal_labels()
    print("joint_core_shadow self-test passed")


if __name__ == "__main__":
    main()
