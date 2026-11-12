"""Offline historical HGB smoke-reproduction; reports, never changes production."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn

from src.deep_research import prep
from src.model_bakeoff import feature_lists, prep_features
from src.hgb_regime_audit import walk_forward, grade


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', required=True)
    parser.add_argument('--out', default='research/phase1/local_results')
    args = parser.parse_args()
    path = Path(args.games)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    df = prep(pd.read_csv(path, low_memory=False))
    nums, cats = feature_lists(df)
    df = prep_features(df, cats)
    pred, diagnostics = walk_forward(df, nums, cats, 'fixed_false', False)
    if pred.empty:
        raise RuntimeError('Not enough eligible historical folds to reproduce baseline.')
    fbs = pred.loc[pred['fbs_vs_fbs'].astype(bool)]
    plays = fbs[(fbs['pred_market_residual'] <= -4.0) & (fbs['closing_total'] >= 56)]
    actual = grade(plays)
    expected = {'games': 651, 'graded': 643, 'wins': 376, 'losses': 267, 'pushes': 8}
    result = {'input_sha256': digest, 'sklearn_version': sklearn.__version__,
              'rule': 'FBS-vs-FBS HGB, early_stopping=False, UNDER edge>=4, total>=56',
              'observed': actual, 'published_expected': expected,
              'matches_published_counts': all(actual[k] == v for k, v in expected.items()),
              'warning': 'Historical reproduction uses ex-post CFBD game weather and untimestamped CFBD selected totals; NOT a decision-time ROI validation.'}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out/'baseline_fold_diagnostics.csv', index=False)
    (out/'baseline_reproduction.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
