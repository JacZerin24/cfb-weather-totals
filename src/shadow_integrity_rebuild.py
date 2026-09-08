from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from . import joint_core_shadow_grade as joint
from . import orientation_shadow_grade as orientation


def _apply_shadow_eligibility_corrections(
    snapshots: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    if snapshots.empty:
        return snapshots.copy()
    work = snapshots.copy()
    work['shadow_eligibility_integrity_override'] = False
    work['shadow_eligibility_integrity_reason'] = ''

    corrections = (protocol.get('data_integrity_corrections', {}) or {}).get(
        'official_entry_eligibility_overrides', []
    )
    run_id = work.get('github_run_id', pd.Series('', index=work.index)).astype(str)
    event_name = work.get('github_event_name', pd.Series('', index=work.index)).astype(str)
    event_schedule = work.get('github_event_schedule', pd.Series('', index=work.index)).astype(str)

    for correction in corrections:
        mask = run_id.eq(str(correction.get('github_run_id', '')))
        expected_event = str(correction.get('github_event_name', '')).strip()
        expected_schedule = str(correction.get('github_event_schedule', '')).strip()
        if expected_event:
            mask &= event_name.eq(expected_event)
        if expected_schedule:
            mask &= event_schedule.eq(expected_schedule)
        if not mask.any():
            continue
        work.loc[mask, 'official_evaluation_eligible'] = True
        work.loc[mask, 'shadow_eligibility_integrity_override'] = True
        work.loc[mask, 'shadow_eligibility_integrity_reason'] = str(
            correction.get('reason', 'Documented shadow eligibility metadata correction.')
        )
    return work


def build_orientation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orientation.validate_protocol()
    protocol = orientation.load_protocol()
    snapshots = _apply_shadow_eligibility_corrections(
        orientation._read_shadow_snapshots(), protocol
    )
    entries = orientation.select_official_shadow_entries(snapshots)
    graded = orientation.grade_entries(entries)
    summary = orientation.build_summary(graded)
    orientation.write_outputs(entries, graded, summary)
    print(summary.to_string(index=False))
    return entries, graded, summary


def build_joint_core() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joint.validate_protocol()
    protocol = joint.load_protocol()
    snapshots = _apply_shadow_eligibility_corrections(
        joint._read_shadow_snapshots(), protocol
    )
    entries = joint.select_official_shadow_entries(snapshots)
    graded = joint.grade_entries(entries)
    summary = joint.build_summary(graded)
    joint.write_outputs(entries, graded, summary)
    print(summary.to_string(index=False))
    return entries, graded, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Apply documented scheduling-integrity corrections to research-only shadow evaluations.'
    )
    parser.add_argument('track', choices=['orientation', 'joint-core', 'all'])
    args = parser.parse_args()
    if args.track in {'orientation', 'all'}:
        build_orientation()
    if args.track in {'joint-core', 'all'}:
        build_joint_core()


if __name__ == '__main__':
    main()
