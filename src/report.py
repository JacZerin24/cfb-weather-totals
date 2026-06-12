from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .utils import ROOT, ensure_dir


def csv_preview(path: Path, max_rows: int = 10) -> str:
    if not path.exists():
        return '_Not available yet._'
    df = pd.read_csv(path)
    if df.empty:
        return '_File exists but has no rows._'
    return df.head(max_rows).to_markdown(index=False)


def main() -> None:
    outputs = ensure_dir('outputs')
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        '# Weekly CFB Weather Totals Report',
        '',
        f'Generated: {generated}',
        '',
        '> Paper tracking only. This report is not a guarantee of profit or betting advice.',
        '',
        '## Weekly picks',
        '',
        csv_preview(ROOT / 'outputs/weekly_picks.csv'),
        '',
        '## Backtest summary',
        '',
        csv_preview(ROOT / 'outputs/backtest_summary.csv'),
        '',
        '## Research summary',
        '',
    ]
    research = ROOT / 'outputs/research_summary.md'
    if research.exists():
        lines.append(research.read_text(encoding='utf-8'))
    else:
        lines.append('_Research summary not generated yet._')
    out = outputs / 'weekly_report.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
