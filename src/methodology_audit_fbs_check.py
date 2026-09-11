from __future__ import annotations

import pandas as pd

from .methodology_audit import GENERAL_EDGE, GENERAL_TOTAL, GENERAL_VARIANTS, general_oof, grade_under
from .utils import read_df, write_df


def main() -> None:
    raw = read_df('data/processed/modeling_dataset.csv')
    if raw.empty:
        raise RuntimeError('modeling_dataset.csv is required.')

    rows = []
    for variant in GENERAL_VARIANTS:
        pred, _ = general_oof(raw, variant)
        if pred.empty:
            continue
        home = pred.get('home_classification', pd.Series('', index=pred.index)).astype(str).str.lower()
        away = pred.get('away_classification', pd.Series('', index=pred.index)).astype(str).str.lower()
        fbs = pred[home.eq('fbs') & away.eq('fbs')].copy()
        rows.append({
            'track': 'general',
            'scope': 'fbs_vs_fbs_independent_check',
            'variant': variant,
            **grade_under(fbs, GENERAL_EDGE, GENERAL_TOTAL),
        })

    out = pd.DataFrame(rows)
    write_df(out, 'outputs/methodology_audit_fbs_independent_check.csv')
    print('=== INDEPENDENT FBS-VS-FBS CHECK ===')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
