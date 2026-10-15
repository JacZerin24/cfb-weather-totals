from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fcs_model import (
    FCS_CATEGORICAL_FEATURES,
    FCS_NUMERIC_FEATURES,
    _prepare_features,
    build_fcs_model,
    historical_fcs_training,
)

GAME_ID = 401868110
TARGET_PRED = -7.899322755487933
WEATHER_NUMS = ['wind_mph', 'temperature_f', 'humidity', 'precipitation', 'snowfall', 'dewpoint_f', 'pressure']


def _load_live_row() -> pd.DataFrame:
    snap = json.loads(Path('outputs/weekly_snapshot.json').read_text(encoding='utf-8'))
    row = next(x for x in snap['board'] if int(x['game_id']) == GAME_ID)
    # Fields used in live scoring but intentionally omitted from the public board output.
    # This matchup is a Big Sky conference game at Montana, not a neutral site.
    row['conference_game'] = True
    row['neutral_site'] = False
    row['game_indoors_bool'] = False
    row['pressure'] = np.nan
    return pd.DataFrame([row])


def _predict(model, row: pd.DataFrame) -> float:
    prepared = _prepare_features(row.copy())
    return float(model.predict(prepared[FCS_NUMERIC_FEATURES + FCS_CATEGORICAL_FEATURES])[0])


def main() -> None:
    hist = historical_fcs_training()
    model = build_fcs_model()
    model.fit(hist[FCS_NUMERIC_FEATURES + FCS_CATEGORICAL_FEATURES], hist['market_residual'])

    live = _load_live_row()
    baseline = _predict(model, live)
    print(f'PRODUCTION_TARGET={TARGET_PRED:.12f}')
    print(f'RECONSTRUCTED_BASELINE={baseline:.12f}')
    print(f'BASELINE_ERROR={baseline - TARGET_PRED:+.12f}')

    medians = {c: float(pd.to_numeric(hist[c], errors='coerce').median()) for c in WEATHER_NUMS}
    print('HISTORICAL_FCS_WEATHER_MEDIANS=' + json.dumps(medians, sort_keys=True))

    rows = []
    def add(name: str, changed: pd.DataFrame) -> None:
        pred = _predict(model, changed)
        rows.append({
            'scenario': name,
            'pred_market_residual': pred,
            'model_total_at_59_5': 59.5 + pred,
            'delta_vs_observed_prediction': pred - baseline,
            'under_edge': -pred,
        })

    add('observed', live.copy())

    typical = live.copy()
    for c, v in medians.items():
        typical[c] = v
    # Recompute categorical bins from changed continuous fields.
    typical = typical.drop(columns=['wind_bin', 'temp_bin'], errors='ignore')
    add('all_weather_to_historical_medians', typical)

    for c in WEATHER_NUMS:
        changed = live.copy()
        changed[c] = medians[c]
        if c == 'wind_mph':
            changed = changed.drop(columns=['wind_bin'], errors='ignore')
        if c == 'temperature_f':
            changed = changed.drop(columns=['temp_bin'], errors='ignore')
        add(f'{c}_to_historical_median', changed)

    # Intuitive local sensitivity scenarios. These are counterfactual probes, not causal estimates.
    for wind in [0.0, 10.0, 15.0, 20.0]:
        changed = live.copy()
        changed['wind_mph'] = wind
        changed = changed.drop(columns=['wind_bin'], errors='ignore')
        add(f'wind_{wind:g}_mph', changed)

    for temp in [55.0, 65.0, 80.0, 90.0]:
        changed = live.copy()
        changed['temperature_f'] = temp
        changed = changed.drop(columns=['temp_bin'], errors='ignore')
        add(f'temp_{temp:g}_f', changed)

    for rh in [30.0, 60.0, 80.0]:
        changed = live.copy()
        changed['humidity'] = rh
        add(f'rh_{rh:g}_pct', changed)

    for precip in [0.0, 0.03, 0.10, 0.25]:
        changed = live.copy()
        changed['precipitation'] = precip
        add(f'precip_{precip:g}', changed)

    out = pd.DataFrame(rows)
    print('=== LOCAL WEATHER COUNTERFACTUALS ===')
    print(out.to_string(index=False))
    Path('outputs/fcs_weather_attribution_401868110.csv').write_text(out.to_csv(index=False), encoding='utf-8')


if __name__ == '__main__':
    main()
