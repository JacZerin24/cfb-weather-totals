from __future__ import annotations

from pathlib import Path

import yaml


PROTOCOL_PATH = Path('config/nfl_frozen_protocol.yml')

EXPECTED = {
    ('protocol', 'id'): 'nfl_totals_paper_v1',
    ('protocol', 'version'): 1,
    ('protocol', 'status'): 'frozen_for_prospective_paper_tracking',
    ('decision', 'official_lead_hours'): 24,
    ('decision', 'side'): 'OVER',
    ('model', 'regression_model'): 'reg_rf_leaf25',
    ('model', 'classifier_model'): 'cls_rf_leaf25',
    ('model', 'forecast_source'): 'jma_gsm',
    ('model', 'retractable_roof_weather_policy'): 'withhold_weather',
    ('qualifier', 'minimum_regression_edge_points'): 3.0,
    ('qualifier', 'minimum_over_probability'): 0.60,
    ('strong', 'minimum_regression_edge_points'): 4.0,
    ('strong', 'minimum_over_probability'): 0.60,
    ('grading', 'mode'): 'paper',
    ('grading', 'stake_units'): 1.0,
    ('change_control', 'outcome_driven_retuning_allowed'): False,
    ('change_control', 'require_new_protocol_version_for_threshold_change'): True,
}


def _get(data: dict, path: tuple[str, ...]):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise AssertionError(f'Missing frozen protocol key: {".".join(path)}')
        current = current[key]
    return current


def main() -> None:
    if not PROTOCOL_PATH.exists():
        raise AssertionError(f'Missing frozen protocol: {PROTOCOL_PATH}')

    data = yaml.safe_load(PROTOCOL_PATH.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise AssertionError('Frozen NFL protocol must be a YAML mapping.')

    errors = []
    for path, expected in EXPECTED.items():
        actual = _get(data, path)
        if actual != expected:
            errors.append(
                f'{".".join(path)} expected {expected!r}, found {actual!r}'
            )

    game_types = _get(data, ('protocol', 'game_types'))
    if game_types != ['REG']:
        errors.append(
            f'protocol.game_types expected ["REG"], found {game_types!r}'
        )

    horizon_48 = _get(data, ('non_official_horizons', 48, 'status'))
    horizon_72 = _get(data, ('non_official_horizons', 72, 'status'))
    if horizon_48 != 'research_only':
        errors.append(
            f'48h horizon must remain research_only, found {horizon_48!r}'
        )
    if horizon_72 != 'disabled':
        errors.append(
            f'72h horizon must remain disabled, found {horizon_72!r}'
        )

    if errors:
        raise AssertionError(
            'Frozen NFL protocol drift detected:\n- ' + '\n- '.join(errors)
        )

    print('NFL frozen protocol v1 validated successfully.')


if __name__ == '__main__':
    main()
