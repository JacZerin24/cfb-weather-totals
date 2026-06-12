from __future__ import annotations

import pandas as pd

from .utils import ensure_dir, write_df


def main() -> None:
    # Placeholder until live odds/forecast integrations are added.
    # This intentionally produces a paper-tracking file with no plays rather than pretending there are actionable picks.
    picks = pd.DataFrame([
        {
            'status': 'paper_tracking_placeholder',
            'note': 'Historical research/backtest must be validated before weekly suggested plays are enabled.',
        }
    ])
    out = write_df(picks, 'outputs/weekly_picks.csv')
    ensure_dir('outputs')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
