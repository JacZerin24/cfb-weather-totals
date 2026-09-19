from __future__ import annotations

from pathlib import Path

import yaml


PATH = Path('config/nfl_prospective_evaluation.yml')

EXPECTED = {
    ('evaluation', 'id'): 'nfl_totals_prospective_eval_v1',
    ('evaluation', 'protocol_id'): 'nfl_totals_paper_v1',
    ('evaluation', 'mode'): 'prospective_paper_only',
    ('evaluation', 'auto_validate_for_wagering'): False,
    ('sample_gate', 'minimum_graded_entries'): 25,
    ('sample_gate', 'minimum_graded_official_decisions'): 100,
    ('sample_gate', 'minimum_entries_with_clv'): 20,
    ('sample_gate', 'minimum_distinct_entry_weeks'): 8,
    ('sample_gate', 'minimum_monitor_coverage_fraction'): 0.80,
    ('performance_review', 'require_positive_captured_price_roi'): True,
    ('performance_review', 'require_mean_clv_points_greater_than'): 0.0,
    ('performance_review', 'require_median_clv_points_at_least'): 0.0,
    ('performance_review', 'require_positive_clv_fraction_at_least'): 0.50,
    ('performance_review', 'require_brier_score_at_most'): 0.25,
    ('performance_review', 'require_brier_not_worse_than_empirical_base_rate'): True,
    ('change_control', 'outcome_driven_changes_allowed'): False,
    ('change_control', 'require_new_evaluation_version_for_change'): True,
    ('change_control', 'human_review_required_before_any_wagering_status'): True,
}


def _get(data: dict, path: tuple[str, ...]):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise AssertionError(
                f'Missing frozen prospective-evaluation key: {".".join(path)}'
            )
        current = current[key]
    return current


def main() -> None:
    if not PATH.exists():
        raise AssertionError(f'Missing prospective evaluation config: {PATH}')
    data = yaml.safe_load(PATH.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise AssertionError('Prospective evaluation config must be a mapping.')

    errors = []
    for path, expected in EXPECTED.items():
        actual = _get(data, path)
        if actual != expected:
            errors.append(
                f'{".".join(path)} expected {expected!r}, found {actual!r}'
            )

    statuses = _get(data, ('reporting', 'statuses'))
    expected_statuses = {
        'insufficient': 'INSUFFICIENT_SAMPLE',
        'review': 'REVIEW_ELIGIBLE',
        'do_not_promote': 'DO_NOT_PROMOTE',
    }
    if statuses != expected_statuses:
        errors.append(
            f'reporting.statuses expected {expected_statuses!r}, found {statuses!r}'
        )

    if errors:
        raise AssertionError(
            'Frozen NFL prospective evaluation drift detected:\n- '
            + '\n- '.join(errors)
        )

    print('NFL prospective evaluation v1 validated successfully.')


if __name__ == '__main__':
    main()
